import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.loc_utils import generate_ume_from_keypoints2, ume_cdist


def calc_inliear_ratio(src_inputs, tgt_inputs, src_pts_tform, gt_tform, ume_r_nn, ume_max_nn, ume_min_nn, eval_num_kpts,
                       keypoints_ignore_segments=[], inlear_thr=0.6, nn_inter_thr=0.6, svd_thr=1e-5):
    device = gt_tform.device
    src_pts = src_inputs['pts']
    src_seg = src_inputs['seg']
    src_feat = src_inputs['feat']
    tgt_pts = tgt_inputs['pts']
    tgt_seg = tgt_inputs['seg']
    tgt_feat = tgt_inputs['feat']

    ume_src, ume_tgt, src_keypoint_pts, tgt_keypoint_pts, _, _ = generate_ume_from_keypoints2(src_pts, src_seg,
                                                                                              src_feat,
                                                                                              tgt_pts, tgt_feat,
                                                                                              gt_tform,
                                                                                              nn_r=ume_r_nn,
                                                                                              max_nn=ume_max_nn,
                                                                                              min_nn=ume_min_nn,
                                                                                              num_samples=eval_num_kpts,
                                                                                              flat_labels=keypoints_ignore_segments,
                                                                                              nn_intersection_r=nn_inter_thr)

    # Filter Invalid UME Matrices
    with torch.no_grad():
        src_valid_ume_mask = ((torch.linalg.svdvals(ume_src) > svd_thr).sum(dim=-1) == 4)
        tgt_valid_ume_mask = ((torch.linalg.svdvals(ume_tgt) > svd_thr).sum(dim=-1) == 4)
        src_valid_ume_mask = src_valid_ume_mask & tgt_valid_ume_mask

    invalid_keypoints_src = torch.zeros_like(ume_src[0, :, 0, 0]).bool()
    invalid_keypoints_src[torch.where(~src_valid_ume_mask)[1]] = True
    ume_src = ume_src[:, ~invalid_keypoints_src]
    ume_tgt = ume_tgt[:, ~invalid_keypoints_src]

    D = ume_cdist(ume_src, ume_tgt).cpu().numpy()
    bs = D.shape[0]
    m = np.zeros((bs, min(D.shape[1], D.shape[2]), 2))
    for b_idx in range(bs):
        src_m_idxs, tgt_m_idxs = linear_sum_assignment(D[b_idx])
        m[b_idx, :, 0] = src_m_idxs
        m[b_idx, :, 1] = tgt_m_idxs
    m = torch.from_numpy(m).long().to(device)
    tgt_matches_keypoint_pts = torch.gather(tgt_keypoint_pts, 1, m[..., 1].unsqueeze(-1).expand(-1, -1, 3))
    src_matches_keypoint_pts = torch.gather(src_keypoint_pts, 1, m[..., 0].unsqueeze(-1).expand(-1, -1, 3))

    R_gt = gt_tform[:, :3, :3]
    t_gt = gt_tform[:, :3, 3]
    src_matches_keypoint_pts_tform = (src_matches_keypoint_pts @ R_gt.transpose(-1, -2) + t_gt[:, None])
    my_re_proj = (tgt_matches_keypoint_pts - src_matches_keypoint_pts_tform).norm(dim=-1)
    inliear_ratio = (my_re_proj <= inlear_thr).float().mean(dim=-1)  #

    return inliear_ratio


def relative_rotation_error(R, R_hat):
    # Compute the rotation matrix difference
    delta_R = torch.matmul(R_hat, torch.transpose(R, 1, 2))

    # Calculate the trace of the matrix difference
    trace_delta_R = torch.einsum('bii->b', delta_R)

    # Clamp the trace to the valid range [-1, 3] to avoid numerical errors
    trace_delta_R = torch.clamp(trace_delta_R, -1, 3)

    # Calculate the relative rotation error in radians
    error_radians = torch.acos((trace_delta_R - 1) / 2)

    # Convert the error to degrees
    error_degrees = error_radians * (180 / torch.tensor(3.141592653589793))

    return error_degrees
