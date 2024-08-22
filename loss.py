import torch
import torch.nn as nn
import numpy as np
from pytorch3d.ops import sample_farthest_points

from utils.loc_utils import ume_cdist, generate_ume_from_keypoints2, batch_estimate_transform_ume_old, ume_kp_layer
from utils.eval_utils import relative_rotation_error


class MyInfoNCELossNoSeg(nn.Module):
    def __init__(self, num_samples=2048, tau=0.1, match_r=0.1, neg_euclid_dist=5):
        super(MyInfoNCELossNoSeg, self).__init__()
        self.num_samples = num_samples
        self.tau = tau
        self.match_r = match_r
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
        self.neg_euclid_dist = neg_euclid_dist

    def forward(self, velo_feat, velo_pts, ref_feat, matches):
        # Get Anchor points from Velo and Pos from Ref
        bs, _, feat_dim = velo_feat.shape

        anchor_feat = torch.gather(velo_feat, 1, matches[..., :1].expand(-1, -1, feat_dim))  # (bs, N, d)
        anchor_pts = torch.gather(velo_pts, 1, matches[..., :1].expand(-1, -1, 3))  # (bs, N, 3)
        pos_feat = torch.gather(ref_feat, 1, matches[..., 1:2].expand(-1, -1, feat_dim))  # (bs, N, d)

        d_pos = self.cos_sim(anchor_feat, pos_feat).view(bs, -1, 1)  # (bs, N, 1)

        D = anchor_feat @ pos_feat.transpose(1, 2)
        D_cat = torch.cat((d_pos, D), dim=2)  # (bs, n_samples, n_samples+1) - [pos,negs]

        # Negatives
        D_euc = torch.cdist(anchor_pts, anchor_pts)
        far_from_anc_mask = D_euc > self.neg_euclid_dist  # (bs, n_samples, n_samples) - diagonal is always 0
        neg_mask = torch.cat([torch.ones_like(far_from_anc_mask[:, :, :1]), far_from_anc_mask],
                             dim=-1)  # (bs, n_samples, n_samples +1) instead of diagonal

        # softmax
        my_softmax = torch.exp(d_pos / self.tau) / (torch.exp(D_cat / self.tau) * neg_mask).sum(dim=-1,
                                                                                                keepdim=True)  # (bs, n_samples, 1)

        # CE
        loss = -torch.log(my_softmax)
        loss = loss.mean()

        return loss


class UMEContrastiveLoss(nn.Module):
    def __init__(self, num_samples=1024, max_nn=5000, min_nn=1000, nn_r=10, tau=0.1, tau_neg=0.1, hd_labels_flag=False,
                 flat_labels=[],
                 nn_intersection_r=0.6, svd_thr=1e-5):
        super(UMEContrastiveLoss, self).__init__()

        self.n_samples = num_samples
        self.max_nn = max_nn
        self.min_nn = min_nn
        self.nn_r = nn_r
        self.tau = tau
        self.tau_neg = tau_neg
        self.hd_labels_flag = hd_labels_flag
        self.flat_labels = flat_labels
        self.nn_intersection_r = nn_intersection_r
        self.svd_thr = svd_thr

    def forward(self, velo_pts, velo_seg, velo_feat, ref_pts, ref_feat, gt_tform):


        velo_ume, ref_ume, velo_keypoint_pts, ref_keypoint_pts, matched_nn_intersection_ratio, with_kpts_batch_cond = generate_ume_from_keypoints2(
            velo_pts, velo_seg,
            velo_feat,
            ref_pts, ref_feat,
            gt_tform,
            num_samples=self.n_samples,
            max_nn=self.max_nn,
            min_nn=self.min_nn,
            nn_r=self.nn_r,
            flat_labels=self.flat_labels,
            normalized_ume=True,
            nn_intersection_r=self.nn_intersection_r)

        # Valid UME Matrix check
        with torch.no_grad():
            velo_valid_ume_mask = ((torch.linalg.svdvals(velo_ume) > self.svd_thr).sum(dim=-1) == 4)
            ref_valid_ume_mask = ((torch.linalg.svdvals(ref_ume) > self.svd_thr).sum(dim=-1) == 4)
            velo_valid_ume_mask = velo_valid_ume_mask & ref_valid_ume_mask

            num_invalid = (~velo_valid_ume_mask).sum().item()
            # if num_invalid > 0:
            #     print(f"{num_invalid} - invalid UME Mats Found")
        #     # ref_valid_ume_mask = torch.linalg.matrix_rank(ref_ume) == 4
        #     # valid_ume_mask = velo_valid_ume_mask[..., None] & ref_valid_ume_mask[..., None].transpose(-1, -2)
        invalid_keypoints_velo = torch.zeros_like(velo_ume[0, :, 0, 0]).bool()
        invalid_keypoints_velo[torch.where(~velo_valid_ume_mask)[1]] = True
        velo_ume = velo_ume[:, ~invalid_keypoints_velo]
        ref_ume = ref_ume[:, ~invalid_keypoints_velo]
        matched_nn_intersection_ratio = matched_nn_intersection_ratio[:, ~invalid_keypoints_velo]
        D_ume = ume_cdist(velo_ume, ref_ume)  # (bs, n_samples, n_samples)
        ume_rank = velo_ume.shape[-1]

        sim_ume = (np.sqrt(ume_rank) - 2 * D_ume) / (np.sqrt(ume_rank))

        # Tau Calc
        tau_mat = self.tau_neg * torch.ones_like(sim_ume)
        pos_mask = torch.arange(D_ume.shape[-1], device=D_ume.device)[None] == \
                   torch.arange(D_ume.shape[-1], device=D_ume.device)[None].T
        pos_mask = pos_mask[None].expand(D_ume.shape[0], -1, -1)
        tau_mat[pos_mask] = self.tau

        exp_sim_ume = torch.exp(sim_ume / tau_mat)
        loss = exp_sim_ume / exp_sim_ume.sum(dim=-1, keepdim=True)  # (bs, n_samples, n_samples)
        loss = torch.diagonal(loss, dim1=-1, dim2=-2)  # (bs, n_samples)
        loss = -torch.log(loss)
        # loss = loss[velo_valid_ume_mask]  # to remove effect of UME matrices with defficent rank
        # loss_invalid = loss[~velo_valid_ume_mask].detach()
        loss = loss.mean()

        return loss, velo_keypoint_pts, ref_keypoint_pts, velo_ume, ref_ume, matched_nn_intersection_ratio, with_kpts_batch_cond


class CubeRegistrationLoss(nn.Module):
    def __init__(self, rtume_max_nn, rtume_r_nn, cube_scale=1.0, nn_inter_ratio_thr=0.75):
        super(CubeRegistrationLoss, self).__init__()
        self.rtume_estimator = ume_kp_layer(ume_knn=rtume_max_nn, ume_desc_rad=rtume_r_nn, diag_only=True)

        unit_cube = torch.tensor([[-1, 1, 1],
                                  [1, 1, 1],
                                  [-1, -1, 1],
                                  [1, -1, 1],
                                  [-1, 1, -1],
                                  [1, 1, -1],
                                  [-1, -1, -1],
                                  [1, -1, -1]])
        self.points_cube = unit_cube.float() * cube_scale
        self.nn_inter_ratio_thr = nn_inter_ratio_thr

    def forward(self, src_pts, src_ume, tgt_pts, tgt_ume, gt_tform, matched_nn_intersection_ratio, valid_batch_entries):
        # Keep valid batch entries
        # src_pts = src_pts[valid_batch_entries]
        # tgt_pts = tgt_pts[valid_batch_entries]
        gt_tform = gt_tform[valid_batch_entries]

        bs, n_hypotsises, _, _ = src_ume.shape
        device = src_ume.device
        self.points_cube = self.points_cube.to(device)

        # Estimate tfrom via RTUME
        G_kp = src_ume.unsqueeze(2)
        H_kp = tgt_ume.unsqueeze(1)
        G, H = G_kp, H_kp
        G = G.reshape(-1, *G.shape[3:])
        H = H.reshape(-1, *H.shape[3:])
        T, _ = batch_estimate_transform_ume_old(G, H)
        rtume_tform = T.view(bs, n_hypotsises, *T.shape[1:])

        R_rtume = rtume_tform[..., :3, :3]  # (bs, ume_n_samples, 3, 3)
        t_rtume = rtume_tform[..., :3, 3]  # (bs, ume_n_samples, 3)
        R_gt = gt_tform[:, :3, :3]
        t_gt = gt_tform[:, :3, 3]

        # src to tgt
        src_estimated_tform_pts = self.points_cube[None, None].expand(bs, n_hypotsises, -1, -1) @ R_rtume.transpose(-1,
                                                                                                                    -2) + t_rtume.unsqueeze(
            -2)  # (bs,n_hypotsises, num_pts, 3)
        src_gt_tform_pts = self.points_cube @ R_gt.transpose(-1, -2) + t_gt[:, None]  # (bs, num_pts, 3)
        src_gt_tform_pts = src_gt_tform_pts[:, None, ...].expand(-1, n_hypotsises, -1,
                                                                 -1)  # (bs,n_hypotsises, num_pts, 3)
        loss_src_to_tgt = (src_gt_tform_pts - src_estimated_tform_pts).norm(dim=-1)  # (bs, n_hypotsises, num_pts)
        loss_src_to_tgt = loss_src_to_tgt.mean(dim=-1)  # (bs, n_hypotsises)

        # Calc loss on keypoints with big NN intersection
        intersection_cond = matched_nn_intersection_ratio >= self.nn_inter_ratio_thr
        # Case no PC with wanted intersection THR, take median value
        if intersection_cond.sum() == 0:
            intersection_cond = matched_nn_intersection_ratio >= \
                                matched_nn_intersection_ratio.median(dim=-1, keepdim=True)[0]

        loss = loss_src_to_tgt[intersection_cond].mean()

        # Calc Rotation and translation errors
        with torch.no_grad():
            rre = relative_rotation_error(R_rtume.view(bs * n_hypotsises, 3, 3),
                                          R_gt.unsqueeze(1).expand(-1, n_hypotsises, -1, -1).reshape(
                                              bs * n_hypotsises, 3, 3)).view(bs, n_hypotsises)  # (bs, n_hypotsises)

            #
            rte = (t_rtume.view(bs * n_hypotsises, 3) - t_gt.unsqueeze(1).expand(-1, n_hypotsises, -1).reshape(
                bs * n_hypotsises, 3)).norm(dim=-1).view(bs, n_hypotsises)  # (bs, n_hypotsises)

        return loss, rre, rte
