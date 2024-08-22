import torch
import MinkowskiEngine as ME
import numpy as np
from pytorch3d.ops import ball_query, knn_gather, knn_points
import torch.nn as nn


def ume_cdist(ume1, ume2):
    Q1 = torch.linalg.qr(ume1, mode='reduced').Q
    P1 = Q1 @ Q1.transpose(-1, -2)
    Q2 = torch.linalg.qr(ume2, mode='reduced').Q
    P2 = Q2 @ Q2.transpose(-1, -2)
    D = torch.cdist(P1.flatten(2), P2.flatten(2)) / np.sqrt(2)

    return D  # (bs, n_samples, n_samples)


def generate_ume_from_keypoints(velo_pts, velo_seg, velo_feat, ref_pts, ref_feat, gt_tform, nn_r=10, max_nn=5000,
                                min_nn=1000, num_samples=1024, flat_labels=[9]):
    # KeyPoint Selection (from Velo)
    bs, velo_pc_size, dim_size = velo_feat.shape
    _, ref_pc_size, _ = ref_pts.shape

    # Select points from Velo PC that are not Flat
    # flat_label = 9
    # non_floor_mask = (velo_seg != flat_label).flatten(1)
    non_floor_mask = (velo_seg != torch.tensor(flat_labels, device=velo_seg.device)).all(dim=-1).flatten(1)
    mask_idxs = torch.where(non_floor_mask)
    mask_idxs_tensor = -1 * torch.ones_like(velo_seg[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    keypoints_velo_pts = torch.gather(velo_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))

    # Get NN of the selected points
    min_length = int(lengths.min())
    _, bq_idxs, bq_nn = ball_query(keypoints_velo_pts, velo_pts, lengths1=lengths, K=max_nn, radius=nn_r,
                                   return_nn=True)
    bq_idxs = bq_idxs[:, :min_length, :]  # (bs, min_length, max_nn)
    bq_nn = bq_nn[:, :min_length, :]  # (bs, min_length, max_nn, 3)

    # Find points with dense nn
    dense_cond = ((bq_idxs > -1).sum(dim=-1) >= min_nn)  # (bs, min_length)
    mask_idxs = torch.where(dense_cond)
    mask_idxs_tensor = -1 * torch.ones_like(bq_idxs[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths2 = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    min_length2 = int(lengths2.min())
    num_samples = min(min_length2, num_samples)
    mask_idxs_tensor = mask_idxs_tensor[:, :num_samples]
    velo_keypoint_pts = torch.gather(keypoints_velo_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))
    velo_valid_nn_idxs = torch.gather(bq_idxs, 1, mask_idxs_tensor[..., None].expand(-1, -1, max_nn))
    velo_valid_nn_idxs[velo_valid_nn_idxs == -1] = velo_pc_size
    velo_valid_nn_pts = torch.gather(bq_nn, 1, mask_idxs_tensor[..., None, None].expand(-1, -1, max_nn, 3))
    velo_valid_nn_pts = velo_valid_nn_pts - velo_keypoint_pts.unsqueeze(-2)
    velo_feat_pad = torch.cat([velo_feat, torch.zeros_like(velo_feat[:, :1, :])], dim=1)
    velo_valid_nn_feat = torch.gather(velo_feat_pad.unsqueeze(1).expand(-1, num_samples, -1, -1), 2,
                                      velo_valid_nn_idxs.unsqueeze(-1).expand(-1, -1, -1, dim_size))

    # UME
    F1_velo = velo_valid_nn_feat.transpose(-1, -2) @ velo_valid_nn_pts  # (bs, num_samples, dim_size, 3)
    F0_velo = velo_valid_nn_feat.transpose(-1, -2).sum(dim=-1, keepdim=True)  # (bs, num_samples, dim_size, 1)
    F_velo = torch.cat([F0_velo, F1_velo], dim=-1)  # (bs, num_samples, dim_size, 4)

    # Find NN in Ref PC
    ref_keypoint_pts = torch.cat([velo_keypoint_pts, torch.ones_like(velo_keypoint_pts[..., :1])], dim=-1)
    ref_keypoint_pts = ref_keypoint_pts @ gt_tform.transpose(-1, -2)
    ref_keypoint_pts = ref_keypoint_pts[..., :3] / ref_keypoint_pts[..., 3].unsqueeze(-1)
    _, ref_valid_nn_idxs, bq_nn = ball_query(ref_keypoint_pts, ref_pts, K=max_nn, radius=nn_r, return_nn=True)
    ref_valid_nn_pts = bq_nn - ref_keypoint_pts.unsqueeze(-2)
    ref_feat_pad = torch.cat([ref_feat, torch.zeros_like(ref_feat[:, :1, :])], dim=1)
    ref_valid_nn_idxs[ref_valid_nn_idxs == -1] = ref_pc_size
    ref_valid_nn_feat = torch.gather(ref_feat_pad.unsqueeze(1).expand(-1, num_samples, -1, -1), 2,
                                     ref_valid_nn_idxs.unsqueeze(-1).expand(-1, -1, -1, dim_size))

    F1_ref = ref_valid_nn_feat.transpose(-1, -2) @ ref_valid_nn_pts  # (bs, num_samples, dim_size, 3)
    F0_ref = ref_valid_nn_feat.transpose(-1, -2).sum(dim=-1, keepdim=True)  # (bs, num_samples, dim_size, 1)
    F_ref = torch.cat([F0_ref, F1_ref], dim=-1)  # (bs, num_samples, dim_size, 4)

    return F_velo, F_ref, velo_keypoint_pts, ref_keypoint_pts


def generate_ume_from_keypoints2(velo_pts, velo_seg, velo_feat, ref_pts, ref_feat, gt_tform,
                                 nn_r=10, max_nn=5000, min_nn=1000, num_samples=1024, flat_labels=[9],
                                 normalized_ume=False, nn_intersection_r=0.6):
    # KeyPoint Selection (from Velo)
    bs, velo_pc_size, dim_size = velo_feat.shape
    _, ref_pc_size, _ = ref_pts.shape

    # Select points from Velo PC that are not Flat
    non_floor_mask = (velo_seg != torch.tensor(flat_labels, device=velo_seg.device)).all(dim=-1).flatten(1)

    # Select points from PC intersection
    R_gt = gt_tform[:, :3, :3]
    t_gt = gt_tform[:, :3, 3]
    velo_pts_tform = velo_pts @ R_gt.transpose(-1, -2) + t_gt[:, None]
    _, tt, _ = ball_query(velo_pts_tform, ref_pts, K=1, radius=nn_intersection_r)
    inter_cond = tt[..., 0] > -1
    filter_cond = inter_cond & non_floor_mask

    mask_idxs = torch.where(filter_cond)
    mask_idxs_tensor = -1 * torch.ones_like(velo_seg[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    keypoints_velo_pts = torch.gather(velo_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))

    # Get NN of the selected points
    min_length = int(lengths.min())
    _, bq_idxs, bq_nn = ball_query(keypoints_velo_pts, velo_pts, lengths1=lengths, K=max_nn, radius=nn_r,
                                   return_nn=True)
    bq_idxs = bq_idxs[:, :min_length, :]  # (bs, min_length, max_nn)
    bq_nn = bq_nn[:, :min_length, :]  # (bs, min_length, max_nn, 3)

    # Find points with dense nn
    dense_cond = ((bq_idxs > -1).sum(dim=-1) >= min_nn)  # (bs, min_length)
    mask_idxs = torch.where(dense_cond)
    mask_idxs_tensor = -1 * torch.ones_like(bq_idxs[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths2 = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    min_length2 = int(lengths2.min())

    # Handle case some examples in batch does not have kpts
    with_kpts_batch_cond = lengths2 > 0
    if min_length2 == 0:
        mask_idxs_tensor = mask_idxs_tensor[with_kpts_batch_cond]
        min_length2 = int(lengths2[with_kpts_batch_cond].min())
        keypoints_velo_pts = keypoints_velo_pts[with_kpts_batch_cond]
        bq_idxs = bq_idxs[with_kpts_batch_cond]
        bq_nn = bq_nn[with_kpts_batch_cond]
        velo_feat = velo_feat[with_kpts_batch_cond]
        ref_feat = ref_feat[with_kpts_batch_cond]
        ref_pts = ref_pts[with_kpts_batch_cond]
        gt_tform = gt_tform[with_kpts_batch_cond]
        R_gt = gt_tform[:, :3, :3]
        t_gt = gt_tform[:, :3, 3]
        bs = R_gt.shape[0]

    num_samples = min(min_length2, num_samples)
    mask_idxs_tensor = mask_idxs_tensor[:, :num_samples]
    velo_keypoint_pts = torch.gather(keypoints_velo_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))
    velo_valid_nn_idxs = torch.gather(bq_idxs, 1, mask_idxs_tensor[..., None].expand(-1, -1, max_nn))
    velo_valid_nn_idxs[velo_valid_nn_idxs == -1] = velo_pc_size
    velo_valid_nn_pts = torch.gather(bq_nn, 1, mask_idxs_tensor[..., None, None].expand(-1, -1, max_nn, 3))
    # velo_valid_nn_pts = velo_valid_nn_pts - velo_keypoint_pts.unsqueeze(-2)
    velo_feat_pad = torch.cat([velo_feat, torch.zeros_like(velo_feat[:, :1, :])], dim=1)
    velo_valid_nn_feat = torch.gather(velo_feat_pad.unsqueeze(1).expand(-1, num_samples, -1, -1), 2,
                                      velo_valid_nn_idxs.unsqueeze(-1).expand(-1, -1, -1, dim_size))

    # UME
    F1_velo = velo_valid_nn_feat.transpose(-1, -2) @ velo_valid_nn_pts  # (bs, num_samples, dim_size, 3)
    F0_velo = velo_valid_nn_feat.transpose(-1, -2).sum(dim=-1, keepdim=True)  # (bs, num_samples, dim_size, 1)
    F_velo = torch.cat([F0_velo, F1_velo], dim=-1)  # (bs, num_samples, dim_size, 4)
    if normalized_ume:
        F_velo = F_velo / (F0_velo.sum(dim=-2, keepdim=True) + 1e-6)

    # Find NN in Ref PC
    ref_keypoint_pts = torch.cat([velo_keypoint_pts, torch.ones_like(velo_keypoint_pts[..., :1])], dim=-1)
    ref_keypoint_pts = ref_keypoint_pts @ gt_tform.transpose(-1, -2)
    ref_keypoint_pts = ref_keypoint_pts[..., :3] / ref_keypoint_pts[..., 3].unsqueeze(-1)
    _, ref_valid_nn_idxs, bq_nn = ball_query(ref_keypoint_pts, ref_pts, K=max_nn, radius=nn_r, return_nn=True)
    ref_valid_nn_pts = bq_nn  # - ref_keypoint_pts.unsqueeze(-2)
    ref_feat_pad = torch.cat([ref_feat, torch.zeros_like(ref_feat[:, :1, :])], dim=1)
    ref_valid_nn_idxs[ref_valid_nn_idxs == -1] = ref_pc_size
    ref_valid_nn_feat = torch.gather(ref_feat_pad.unsqueeze(1).expand(-1, num_samples, -1, -1), 2,
                                     ref_valid_nn_idxs.unsqueeze(-1).expand(-1, -1, -1, dim_size))

    F1_ref = ref_valid_nn_feat.transpose(-1, -2) @ ref_valid_nn_pts  # (bs, num_samples, dim_size, 3)
    F0_ref = ref_valid_nn_feat.transpose(-1, -2).sum(dim=-1, keepdim=True)  # (bs, num_samples, dim_size, 1)
    F_ref = torch.cat([F0_ref, F1_ref], dim=-1)  # (bs, num_samples, dim_size, 4)
    if normalized_ume:
        F_ref = F_ref / (F0_ref.sum(dim=-2, keepdim=True) + 1e-6)

    # Calc Matched NN Intersection Ratio
    velo_valid_nn_pts_tform = velo_valid_nn_pts @ R_gt[:, None].transpose(-1, -2) + t_gt[:, None, None]
    velo_valid_nn_pts_tform_flat = velo_valid_nn_pts_tform.flatten(0, 1)
    ref_valid_nn_pts_flat = ref_valid_nn_pts.flatten(0, 1)
    _, idx, _ = ball_query(velo_valid_nn_pts_tform_flat, ref_valid_nn_pts_flat, K=1, radius=nn_intersection_r)
    cond = idx > -1
    matched_nn_intersection_ratio = cond.view(bs, num_samples, -1).float().mean(dim=-1)  # (bs, num_samples)

    return F_velo, F_ref, velo_keypoint_pts, ref_keypoint_pts, matched_nn_intersection_ratio, with_kpts_batch_cond


def generate_ume_from_keypoints3(velo_pts, velo_seg, velo_feat, ref_pts, ref_feat, gt_tform, nn_r=10, max_nn=5000,
                                 min_nn=1000, num_samples=1024, flat_labels=[9], normalized_ume=False):
    # KeyPoint Selection (from Velo)
    bs, velo_pc_size, dim_size = velo_feat.shape
    _, ref_pc_size, _ = ref_pts.shape

    # Select points from Velo PC that are not Flat
    non_floor_mask = (velo_seg != torch.tensor(flat_labels, device=velo_seg.device)).all(dim=-1).flatten(1)

    R_gt = gt_tform[:, :3, :3]
    t_gt = gt_tform[:, :3, 3]
    velo_pts_tform = velo_pts @ R_gt.transpose(-1, -2) + t_gt[:, None]
    filter_cond = non_floor_mask

    mask_idxs = torch.where(filter_cond)
    mask_idxs_tensor = -1 * torch.ones_like(velo_seg[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    keypoints_velo_pts = torch.gather(velo_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))

    # Get NN of the selected points
    min_length = int(lengths.min())
    _, bq_idxs, bq_nn = ball_query(keypoints_velo_pts, velo_pts, lengths1=lengths, K=max_nn, radius=nn_r,
                                   return_nn=True)
    bq_idxs = bq_idxs[:, :min_length, :]  # (bs, min_length, max_nn)
    bq_nn = bq_nn[:, :min_length, :]  # (bs, min_length, max_nn, 3)

    # Find points with dense nn
    dense_cond = ((bq_idxs > -1).sum(dim=-1) >= min_nn)  # (bs, min_length)
    mask_idxs = torch.where(dense_cond)
    mask_idxs_tensor = -1 * torch.ones_like(bq_idxs[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths2 = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    min_length2 = int(lengths2.min())

    # Handle case some examples in batch does not have kpts
    with_kpts_batch_cond = lengths2 > 0
    if min_length2 == 0:
        mask_idxs_tensor = mask_idxs_tensor[with_kpts_batch_cond]
        min_length2 = int(lengths2[with_kpts_batch_cond].min())
        keypoints_velo_pts = keypoints_velo_pts[with_kpts_batch_cond]
        bq_idxs = bq_idxs[with_kpts_batch_cond]
        bq_nn = bq_nn[with_kpts_batch_cond]
        velo_feat = velo_feat[with_kpts_batch_cond]
        ref_feat = ref_feat[with_kpts_batch_cond]
        ref_pts = ref_pts[with_kpts_batch_cond]
        gt_tform = gt_tform[with_kpts_batch_cond]
        R_gt = gt_tform[:, :3, :3]
        t_gt = gt_tform[:, :3, 3]
        bs = R_gt.shape[0]

    num_samples = min(min_length2, num_samples)
    mask_idxs_tensor = mask_idxs_tensor[:, :num_samples]
    velo_keypoint_pts = torch.gather(keypoints_velo_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))
    velo_valid_nn_idxs = torch.gather(bq_idxs, 1, mask_idxs_tensor[..., None].expand(-1, -1, max_nn))
    velo_valid_nn_idxs[velo_valid_nn_idxs == -1] = velo_pc_size
    velo_valid_nn_pts = torch.gather(bq_nn, 1, mask_idxs_tensor[..., None, None].expand(-1, -1, max_nn, 3))
    # velo_valid_nn_pts = velo_valid_nn_pts - velo_keypoint_pts.unsqueeze(-2)
    velo_feat_pad = torch.cat([velo_feat, torch.zeros_like(velo_feat[:, :1, :])], dim=1)
    velo_valid_nn_feat = torch.gather(velo_feat_pad.unsqueeze(1).expand(-1, num_samples, -1, -1), 2,
                                      velo_valid_nn_idxs.unsqueeze(-1).expand(-1, -1, -1, dim_size))

    # UME
    F1_velo = velo_valid_nn_feat.transpose(-1, -2) @ velo_valid_nn_pts  # (bs, num_samples, dim_size, 3)
    F0_velo = velo_valid_nn_feat.transpose(-1, -2).sum(dim=-1, keepdim=True)  # (bs, num_samples, dim_size, 1)
    F_velo = torch.cat([F0_velo, F1_velo], dim=-1)  # (bs, num_samples, dim_size, 4)
    if normalized_ume:
        F_velo = F_velo / (F0_velo.sum(dim=-2, keepdim=True) + 1e-6)

    # Find NN in Ref PC
    ref_keypoint_pts = torch.cat([velo_keypoint_pts, torch.ones_like(velo_keypoint_pts[..., :1])], dim=-1)
    ref_keypoint_pts = ref_keypoint_pts @ gt_tform.transpose(-1, -2)
    ref_keypoint_pts = ref_keypoint_pts[..., :3] / ref_keypoint_pts[..., 3].unsqueeze(-1)
    _, ref_valid_nn_idxs, bq_nn = ball_query(ref_keypoint_pts, ref_pts, K=max_nn, radius=nn_r, return_nn=True)
    ref_valid_nn_pts = bq_nn  # - ref_keypoint_pts.unsqueeze(-2)
    ref_feat_pad = torch.cat([ref_feat, torch.zeros_like(ref_feat[:, :1, :])], dim=1)
    ref_valid_nn_idxs[ref_valid_nn_idxs == -1] = ref_pc_size
    ref_valid_nn_feat = torch.gather(ref_feat_pad.unsqueeze(1).expand(-1, num_samples, -1, -1), 2,
                                     ref_valid_nn_idxs.unsqueeze(-1).expand(-1, -1, -1, dim_size))

    F1_ref = ref_valid_nn_feat.transpose(-1, -2) @ ref_valid_nn_pts  # (bs, num_samples, dim_size, 3)
    F0_ref = ref_valid_nn_feat.transpose(-1, -2).sum(dim=-1, keepdim=True)  # (bs, num_samples, dim_size, 1)
    F_ref = torch.cat([F0_ref, F1_ref], dim=-1)  # (bs, num_samples, dim_size, 4)
    if normalized_ume:
        F_ref = F_ref / (F0_ref.sum(dim=-2, keepdim=True) + 1e-6)

    # Calc Matched NN Intersection Ratio
    velo_valid_nn_pts_tform = velo_valid_nn_pts @ R_gt[:, None].transpose(-1, -2) + t_gt[:, None, None]
    velo_valid_nn_pts_tform_flat = velo_valid_nn_pts_tform.flatten(0, 1)
    ref_valid_nn_pts_flat = ref_valid_nn_pts.flatten(0, 1)
    _, idx, _ = ball_query(velo_valid_nn_pts_tform_flat, ref_valid_nn_pts_flat, K=1, radius=0.6)
    cond = idx > -1
    matched_nn_intersection_ratio = cond.view(bs, num_samples, -1).float().mean(dim=-1)  # (bs, num_samples)

    return F_velo, F_ref, velo_keypoint_pts, ref_keypoint_pts, matched_nn_intersection_ratio, with_kpts_batch_cond


def batch_estimate_transform_ume_old(G, H):
    """
    Assuming G = UME_MAT(target_pc), H = UME_MAT(source_pc) such that target_pc = T(source_pc), and
    G= H @ D, estimates D_to_T(D), the estimated transformation between the point clouds
    :param H: source UME matrix
    :param G: target UME matrix
    :return: T, estimated transformation between the point clouds
    """
    bs = G.size(0)
    dev = G.device
    # weight vectors [bs x in_dim x 1]
    mg = G[:, :, 0].unsqueeze(2)
    mh = H[:, :, 0].unsqueeze(2)

    # first order moment matrices [bs x in_dim x 3]
    g = G[:, :, 1:]
    h = H[:, :, 1:]

    # pre compute some values
    mg_square = torch.sum(mg ** 2, dim=1, keepdim=True) + 1e-16
    mg_mh = torch.sum(mg * mh, dim=1, keepdim=True)
    gmg = torch.sum(g * mg, dim=1, keepdim=True)
    hmg = torch.sum(h * mg, dim=1, keepdim=True)

    # rotation estimate
    # [bs x in_dim x 3]
    wlc = gmg / (mg_square + 1e-16)
    wrc = hmg / (mg_mh + 1e-16)

    left = g - wlc * mg
    right = h - wrc * mh

    M = torch.transpose(right, 2, 1) @ left
    U, S, VH = torch.linalg.svd(torch.transpose(M, 2, 1))
    Q = torch.eye(3, device=dev).repeat(bs, 1, 1)
    Q[:, 2, 2] = torch.sign(torch.det(U @ VH))
    R = U @ Q @ VH

    # translation estimate
    b2 = wrc - wlc @ R

    D_2 = torch.eye(4, device=dev).repeat(bs, 1, 1)
    D_2[:, 0, 1:4] = b2.squeeze(1)
    D_2[:, 1:4, 1:4] = R
    D_R = D_2

    H_orth, _ = torch.linalg.qr(H, mode='reduced')  # torch.qr(data_tensor, some=True)
    H_HT = torch.matmul(H_orth, H_orth.transpose(1, 2))

    G_orth, _ = torch.linalg.qr(G, mode='reduced')  # torch.qr(data_tensor, some=True)
    G_GT = torch.matmul(G_orth, G_orth.transpose(1, 2))

    D = 0.707 * torch.linalg.norm(H_HT - G_GT, 'fro', dim=(1, 2))

    # errors
    T = torch.eye(4, device=dev).repeat(bs, 1, 1)
    T[:, :3, :3] = torch.transpose(D_R[:, 1:, 1:], 2, 1)
    T[:, :3, 3] = D_R[:, 0, 1:]
    return T, D


def ball_query_gather(pts, idx):
    return knn_gather(torch.cat((torch.zeros((pts.shape[0], 1, pts.shape[-1]), device=pts.device), pts), 1), idx + 1)


class ume_kp_layer(nn.Module):
    def __init__(self, ume_knn, ume_desc_rad, diag_only=False, n_rand=None):
        super().__init__()
        self.ume_knn = ume_knn
        self.ume_desc_rad = ume_desc_rad
        self.diag_only = diag_only
        self.n_rand = n_rand

    def ume_mat(self, points, features, bs, n_kp):
        m0 = torch.sum(features, dim=1, keepdim=True)
        m1 = features.transpose(2, 1) @ points  # [bs x feat_dim x 3]
        # ume_mat = torch.cat((m0.transpose(2, 1), m1), dim=2) / (torch.linalg.norm(m0, dim=2, keepdim=True) + 1e-6)
        # ume_mat = torch.cat((m0.transpose(2, 1), m1), dim=2) / (torch.sum(torch.sum(features, dim=-1, keepdim=True) != 0, dim=-2,keepdim=True) + 1e-6)
        ume_mat = torch.cat((m0.transpose(2, 1), m1), dim=2) / (torch.sum(m0, dim=-1, keepdim=True) + 1e-6)

        return ume_mat.view(bs, n_kp, *ume_mat.shape[1:])

    def batch_keypoints(self, points, idx):
        points_kp_batch = ball_query_gather(points, idx)
        points_kp_batch = points_kp_batch.view(-1, *points_kp_batch.shape[2:])

        return points_kp_batch

    def forward(self, source_points, source_features, source_kp,
                target_points, target_features, target_kp):
        bs, n_kp = source_kp.shape[0], source_kp.shape[1]
        source_dists = ball_query(source_kp, source_points, radius=self.ume_desc_rad, K=self.ume_knn, return_nn=False)
        target_dists = ball_query(target_kp, target_points, radius=self.ume_desc_rad, K=self.ume_knn, return_nn=False)
        # source_nn = knn_points(source_kp,source_points)
        # target_nn = knn_points(target_kp,target_points)
        source_kp_points_batch = self.batch_keypoints(source_points, source_dists.idx)
        source_kp_feat_batch = self.batch_keypoints(source_features, source_dists.idx)
        target_kp_points_batch = self.batch_keypoints(target_points, target_dists.idx)
        target_kp_feat_batch = self.batch_keypoints(target_features, target_dists.idx)

        G_kp = self.ume_mat(source_kp_points_batch, source_kp_feat_batch, bs, n_kp).unsqueeze(2)
        H_kp = self.ume_mat(target_kp_points_batch, target_kp_feat_batch, bs, n_kp).unsqueeze(1)
        if not self.diag_only:
            G, H = torch.broadcast_tensors(G_kp, H_kp)
            G = G.reshape(-1, *G.shape[3:])
            H = H.reshape(-1, *H.shape[3:])
        else:
            G, H = G_kp, H_kp
            G = G.reshape(-1, *G.shape[3:])
            H = H.reshape(-1, *H.shape[3:])
        # reg_mat = torch.zeros_like(G)
        # reg_mat[:,:4,:4] = 1e-6*torch.eye(4,device=G.device)
        # G = G + reg_mat
        # H = H + reg_mat
        if self.n_rand is not None:
            # only valid for batch size of one!!
            triplets = np.random.choice(np.arange(G.shape[0]), (self.n_rand, 3))
            G = G[triplets[:, 0]] + G[triplets[:, 1]] + G[triplets[:, 2]]
            H = H[triplets[:, 0]] + H[triplets[:, 1]] + H[triplets[:, 2]]
        T, D = batch_estimate_transform_ume_old(G, H)

        err_T = torch.tensor([[0, 1, 0, 100], [-1, 0, 0, -100], [0, 0, 1, 50], [0, 0, 0, 1]], device=H.device,
                             dtype=torch.float32)
        # T[torch.sum(torch.bitwise_not(H==0)[:,:,0],dim=-1)<3] = err_T
        # D[torch.sum(torch.bitwise_not(H==0)[:,:,0],dim=-1)<3] = torch.tensor(10.0,device=H.device)
        # T[torch.sum(torch.bitwise_not(G==0)[:,:,0],dim=-1)<3] = err_T
        # D[torch.sum(torch.bitwise_not(G==0)[:,:,0],dim=-1)<3] = torch.tensor(10.0,device=H.device)
        # T = batch_estimate_transform_ume(G, H)
        # D = torch.sqrt(2 - 2 * torch.sum(knn_gather(source_features,source_nn.idx) * knn_gather(target_features,target_nn.idx).squeeze(-2).unsqueeze(1),dim=-1))
        # D = torch.cdist(knn_gather(source_features,source_nn.idx).squeeze(-2), knn_gather(target_features,target_nn.idx).squeeze(-2))
        if not self.diag_only:
            T = T.view(bs, n_kp, n_kp, *T.shape[1:])
            D = D.view(bs, n_kp, n_kp)
        else:
            if self.n_rand is not None:
                n_kp = -1
            T = T.view(bs, n_kp, *T.shape[1:])
            D = D.view(bs, n_kp)

        return T, D, G_kp.squeeze(), H_kp.squeeze()


def create_local_ume_matrix(nn_pts, nn_feat):
    """
    Generate Local UME matrix for selected keypoints (um
    :param nn_pts: Tensor (bs, num_keypoints, pc_size, 3)
    :param nn_feat: Tensor (bs, num_keypoints, pc_size, dim_size)
    :return:
    """
    F1 = nn_feat.transpose(-1, -2) @ nn_pts  # (bs, num_samples, dim_size, 3)
    F0 = nn_feat.transpose(-1, -2).sum(dim=-1, keepdim=True)  # (bs, num_samples, dim_size, 1)
    ume = torch.cat([F0, F1], dim=-1)  # (bs, num_samples, dim_size, 4)

    return ume


def sample_smart_keypoints(pc_pts, pc_seg, nn_r=10, max_nn=5000, min_nn=1000, num_samples=1024):
    bs, pc_size, _ = pc_pts.shape
    num_potential_samples = num_samples * 8

    # Select points from PC that are not Flat
    flat_label = 9
    non_floor_mask = (pc_seg != flat_label).flatten(1)
    mask_idxs = torch.where(non_floor_mask)
    mask_idxs_tensor = -1 * torch.ones_like(pc_seg[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    keypoints_pts = torch.gather(pc_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))

    # Get NN of the selected points
    min_length = int(lengths.min())
    _, bq_idxs, bq_nn = ball_query(keypoints_pts, pc_pts, lengths1=lengths, K=max_nn, radius=nn_r, return_nn=True)
    bq_idxs = bq_idxs[:, :min_length, :]  # (bs, min_length, max_nn)

    # Find points with dense nn
    dense_cond = ((bq_idxs > -1).sum(dim=-1) >= min_nn)  # (bs, min_length)
    mask_idxs = torch.where(dense_cond)
    mask_idxs_tensor = -1 * torch.ones_like(bq_idxs[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths2 = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    min_length2 = int(lengths2.min())
    num_potential_samples = min(min_length2, num_potential_samples)
    mask_idxs_tensor = mask_idxs_tensor[:, :num_potential_samples]
    keypoints_pts = torch.gather(keypoints_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))
    keypoints_nn_idxs = torch.gather(bq_idxs, 1, mask_idxs_tensor[..., None].expand(-1, -1, max_nn))

    # # Sample far keypoint from each other
    # num_samples = min(num_samples, keypoints_pts.shape[1])
    # keypoints_pts, keep_keypoints_idxs = sample_farthest_points(keypoints_pts, K=num_samples)
    # keypoints_nn_idxs = torch.gather(keypoints_nn_idxs, 1, keep_keypoints_idxs.unsqueeze(-1).expand(-1, -1, max_nn))

    # Keep Keypoints that are far enough
    valid_idxs = -torch.ones_like(keypoints_pts[..., 0]).long()
    for batch_idx in range(keypoints_pts.shape[0]):
        _, idxs = ME.utils.quantization.sparse_quantize(keypoints_pts[batch_idx], return_index=True,
                                                        quantization_size=nn_r, device=keypoints_pts.device)
        valid_idxs[batch_idx, :idxs.shape[0]] = idxs

    new_num_samples = min(int((valid_idxs > -1).sum(dim=-1).min()), num_samples)

    keypoints_pts = torch.gather(keypoints_pts, 1, valid_idxs[:, :new_num_samples, None].expand(-1, -1, 3))
    keypoints_nn_idxs = torch.gather(keypoints_nn_idxs, 1,
                                     valid_idxs[:, :new_num_samples, None].expand(-1, -1, max_nn))

    return keypoints_pts, keypoints_nn_idxs


def sample_smart_keypoints2(pc_pts, pc_seg, nn_r=10, max_nn=5000, min_nn=1000, num_samples=1024, d_grid=4, dz_grid=2,
                            grid_clip_thr=0.75):
    bs, pc_size, _ = pc_pts.shape
    num_potential_samples = num_samples * 8

    # Create Grid of fat points
    batch_max_lim = pc_pts.max(dim=1)[0].max(dim=0)[0]
    batch_min_lim = pc_pts.min(dim=1)[0].min(dim=0)[0]

    x_vals = torch.arange(batch_min_lim[0], batch_max_lim[0], d_grid, device='cuda:0') - 0.5 * d_grid
    y_vals = torch.arange(batch_min_lim[1], batch_max_lim[1], d_grid, device='cuda:0') - 0.5 * d_grid
    z_vals = torch.arange(batch_min_lim[2], batch_max_lim[2], dz_grid, device='cuda:0') - 0.5 * dz_grid
    X, Y, Z = torch.meshgrid(x_vals, y_vals, z_vals)
    pc_grid = torch.stack([X, Y, Z], dim=-1).view(-1, 3)[None].expand(bs, -1, -1)  # (bs, n_grid,3)
    # _, k_idxs, nn = ball_query(pc_grid[None].expand(bs, -1, -1), pc_pts, K=1, radius=bq_thr_dist)
    # k_pts = nn.squeeze(2)

    # Select points that are not from floor
    flat_label = 9
    non_floor_mask = (pc_seg != flat_label).flatten(1)
    mask_idxs = torch.where(non_floor_mask)
    mask_idxs_tensor = -1 * torch.ones_like(pc_seg[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    keypoints_pts = torch.gather(pc_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))

    # for each grid points, find its nearst point that is not from floor (those our keypoints)
    _, k_idxs, _ = ball_query(pc_grid, keypoints_pts, lengths2=lengths, K=1, radius=grid_clip_thr)
    k_idxs = k_idxs[..., 0]
    lengths = (k_idxs > -1).sum(dim=-1)
    k_idxs = k_idxs.sort(dim=1, descending=True)[0]
    k_idxs[k_idxs == -1] = 0
    keypoints_pts = torch.gather(keypoints_pts, 1, k_idxs[..., None].expand(-1, -1, 3))

    # Keep keypoints that are from dense NNs
    min_length = int(lengths.min())
    _, bq_idxs, bq_nn = ball_query(keypoints_pts, pc_pts, lengths1=lengths, K=max_nn, radius=nn_r, return_nn=True)
    # bq_idxs = bq_idxs[:, :min_length, :]  # (bs, min_length, max_nn)

    # Find points with dense nn
    dense_cond = ((bq_idxs > -1).sum(dim=-1) >= min_nn)  # (bs, min_length)
    mask_idxs = torch.where(dense_cond)
    mask_idxs_tensor = -1 * torch.ones_like(bq_idxs[..., 0]).long()
    mask_idxs_tensor[mask_idxs] = mask_idxs[1]
    mask_idxs_tensor = mask_idxs_tensor.sort(dim=1, descending=True)[0]
    lengths2 = (mask_idxs_tensor > -1).sum(dim=-1)
    mask_idxs_tensor[mask_idxs_tensor == -1] = 0
    min_length2 = int(lengths2.min())
    num_samples = min(min_length2, num_samples)
    mask_idxs_tensor = mask_idxs_tensor[:, :num_samples]
    keypoints_pts = torch.gather(keypoints_pts, 1, mask_idxs_tensor[..., None].expand(-1, -1, 3))
    keypoints_nn_idxs = torch.gather(bq_idxs, 1, mask_idxs_tensor[..., None].expand(-1, -1, max_nn))

    return keypoints_pts, keypoints_nn_idxs


def get_matched_keypoints_and_nn(src_keypoints, tgt_pts, gt_tform, nn_r=10, max_nn=5000):
    """
    Get set of keypoints from src PC and find thier matched points in tgt PC (and thier NN idxes) using the GT transformation
    :param src_keypoints:
    :param tgt_pts:
    :param gt_tform:
    :param nn_r:
    :param max_nn:
    :return:
    """
    tgt_keypoint_pts = torch.cat([src_keypoints, torch.ones_like(src_keypoints[..., :1])], dim=-1)
    tgt_keypoint_pts = tgt_keypoint_pts @ gt_tform.transpose(-1, -2)
    tgt_keypoint_pts = tgt_keypoint_pts[..., :3] / tgt_keypoint_pts[..., 3].unsqueeze(-1)
    _, tgt_keypoints_nn_idxs, _ = ball_query(tgt_keypoint_pts, tgt_pts, K=max_nn, radius=nn_r)

    return tgt_keypoint_pts, tgt_keypoints_nn_idxs


def feature_spatial_var(pts, feat, knn=10):
    q_nn_to_p = knn_points(pts, pts, K=knn)
    nn_feat = knn_gather(feat, q_nn_to_p.idx[:, :, 1:])
    nn_feat_diff = feat.unsqueeze(-2) - nn_feat
    nn_feat_diff_norm = torch.linalg.norm(nn_feat_diff, dim=-1)
    nn_feat_mean_diff_norm = torch.mean(nn_feat_diff_norm, dim=-1)
    return nn_feat_mean_diff_norm


def cauchy_kernel(e, k=0.1):
    return 1 / (1 + (e / k) ** 2)


def pc_corr(p, q, q_nn_to_p, vals_p, vals_q, sigma, P=None, use_norm=False, src_norm=None, tgt_norm=None):
    dist_mat = torch.linalg.norm(p.unsqueeze(2) - q[q_nn_to_p], axis=-1)
    # dist_mat = torch.abs(torch.sum((p.unsqueeze(1) - q[q_nn_to_p])*q_norm[q_nn_to_p], axis=2))

    weight_mat = cauchy_kernel(dist_mat, sigma)
    # dist_weight_mat = torch.clamp(1 - (dist_mat / inlier_th), min=0.0)
    # n_neighbors = torch.maximum(torch.sum(torch.gt(dist_weight_mat,0),dim=-1),torch.ones((1,),device=q.device))
    if P is not None:
        val_product_mat = torch.einsum("Nj,jk->Nk", vals_p[:, :], P)
        val_product_mat = torch.einsum("BNj,BNMj->BNM", val_product_mat[None], vals_q[q_nn_to_p])
    else:
        val_product_mat = torch.sum(vals_p[None, :, None, :] * vals_q[q_nn_to_p], axis=-1)
    if use_norm:
        norm_product_mat = torch.abs(torch.sum(src_norm[:, :, None, :] * tgt_norm[q_nn_to_p], axis=-1))
        val_product_mat = val_product_mat * norm_product_mat
    # normalize features
    # val_product_mat = val_product_mat / (
    #         torch.linalg.norm(vals_p, dim=-1)[None,:,None] * torch.linalg.norm(vals_q[q_nn_to_p], dim=-1) + 1e-6)

    # val_out = torch.sum(torch.sum(weight_mat * val_product_mat,dim=2)/n_neighbors,dim=-1)
    val_out = torch.sum(weight_mat * val_product_mat, dim=(1, 2))

    val_out = val_out / vals_p.shape[0]
    # val_out = val_out / torch.sum(weight_mat,dim=(1,2))

    # val_out = torch.sum(weight_mat * val_product_mat,dim=(1,2))/torch.sum(weight_mat,dim=(1,2))

    return val_out


def pc_corr_pytorch3d(p, q, k, vals_p, vals_q, sigma, P=None, use_norm=False, src_norm=None, tgt_norm=None):
    q_nn_to_p = knn_points(p, q.expand(p.shape[0], -1, 3), K=k)  #
    return pc_corr(p, q, q_nn_to_p.idx, vals_p, vals_q, sigma, P, use_norm, src_norm, tgt_norm)


def pc_corr_cost_pytorch3d(x1, x2, source_points, target_points, k, source_vals, target_vals, sigma, P=None,
                           use_norm=False, src_norm=None, tgt_norm=None, dev='cpu'):
    R = x1.to(dev)
    t = x2.to(dev)
    source_transformed = source_points.unsqueeze(0) @ torch.transpose(R, 1, 2) + t.unsqueeze(1)
    if use_norm:
        src_norm_transformed = src_norm.unsqueeze(0) @ torch.transpose(R, 1, 2)
    else:
        src_norm_transformed = src_norm
    return pc_corr_pytorch3d(source_transformed, target_points, k, source_vals, target_vals, sigma, P, use_norm,
                             src_norm_transformed, tgt_norm)


class FeatureCorrelator:
    def __init__(self,
                 n_clusters=8,
                 batch=1,
                 n_hypotheses=1,
                 sigma=0.05,
                 P=None,
                 corr_num_nn=20):
        self.n_clusters = n_clusters
        self.batch = batch
        self.sigma = sigma
        self.n_hypotheses = n_hypotheses
        self.P = P
        self.corr_num_nn = corr_num_nn

    def feature_corr_hypothesis_test(self, source_pc, target_pc, source_feat, target_feat,
                                     T_kp, src_norm=None, tgt_norm=None):
        T_clustered = T_kp
        T_clustered_batched = list(torch.split(T_clustered, self.batch))
        mmf_score = []
        m = torch.mean(torch.concat((source_feat, target_feat), dim=1), dim=1)
        src_feat_weight = feature_spatial_var(source_pc, source_feat, knn=50)
        tgt_feat_weight = feature_spatial_var(target_pc, target_feat, knn=50)
        weighted_source_feat = (source_feat - m) * src_feat_weight.unsqueeze(-1)
        weighted_target_feat = (target_feat - m) * tgt_feat_weight.unsqueeze(-1)
        for i in range(T_clustered_batched.__len__()):
            a = pc_corr_cost_pytorch3d(T_clustered_batched[i][:, :3, :3], T_clustered_batched[i][:, :3, 3],
                                       source_pc.squeeze(), target_pc.squeeze(),
                                       self.corr_num_nn, weighted_source_feat.squeeze(),
                                       weighted_target_feat.squeeze(), self.sigma, self.P,
                                       use_norm=False, src_norm=src_norm, tgt_norm=tgt_norm, dev=source_pc.device)
            mmf_score.append(a)
        mmf_score = torch.cat(mmf_score)
        T_clustered = torch.cat(T_clustered_batched, dim=0)

        best_T_list_order = torch.argsort(mmf_score, descending=True)
        return_T_list = T_clustered[best_T_list_order[:self.n_hypotheses]]

        return_mmf_score = mmf_score[best_T_list_order[:self.n_hypotheses]]
        best_T = return_T_list[torch.argmax(return_mmf_score)]

        return best_T
