import argparse
import torch
import MinkowskiEngine as ME
import numpy as np
from pytorch3d.ops import ball_query, knn_points, knn_gather
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets.kitti.kitti_dataset import SemanticKITTIDataset, batch_collate_fn_dset
from datasets.nuscenes.nuscenes_dataset import NuscenesDataset
from models import ResUNetSmall2
from functools import partial
from scipy.optimize import linear_sum_assignment
import open3d as o3d
from utils.general_utils import update_namespace_from_yaml
from utils.loc_utils import ume_cdist, FeatureCorrelator, ume_kp_layer, batch_estimate_transform_ume_old
from utils.eval_utils import relative_rotation_error



def pc_fcht(pc1_pts, pc2_pts, pc1_feat, pc2_feat, rtume_hypotises, gt_tform, corr_sigma, args):
    hypotisis_matcher = FeatureCorrelator(
        sigma=corr_sigma,
        batch=args.corr_batch_size,
        n_hypotheses=10)

    tform_hat = torch.tensor([], device=args.device)
    for b_idx in range(args.batch_size):
        tt = hypotisis_matcher.feature_corr_hypothesis_test(
            source_pc=pc1_pts[b_idx][None],
            target_pc=pc2_pts[b_idx][None],
            source_feat=pc1_feat[b_idx][None],
            target_feat=pc2_feat[b_idx][None],
            T_kp=rtume_hypotises[b_idx],
            src_norm=None,
            tgt_norm=None,
        )
        tform_hat = torch.cat([tform_hat, tt[None]], dim=0)

    R_hat = tform_hat[:, :3, :3]
    t_hat = tform_hat[:, :3, 3]

    R_gt = gt_tform[:, :3, :3]
    t_gt = gt_tform[:, :3, 3]
    R_err = relative_rotation_error(R_hat, R_gt).cpu()
    t_err = (t_hat - t_gt).norm(dim=-1).cpu()

    return R_err, t_err, R_hat, t_hat


def my_ume_generation(pts, kpts, feat, args):
    _, bq_idxs, bq_nn = ball_query(kpts, pts, K=args.ume_max_nn, radius=args.ume_r_nn, return_nn=True)
    bq_idxs[bq_idxs == -1] = pts.shape[1]
    feat_pad = torch.cat([feat, torch.zeros_like(feat[:, :1, :])], dim=1)
    nn_feat = torch.gather(feat_pad.unsqueeze(1).expand(-1, kpts.shape[1], -1, -1), 2,
                           bq_idxs.unsqueeze(-1).expand(-1, -1, -1, 32))
    F1 = nn_feat.transpose(-1, -2) @ bq_nn  # (bs, num_samples, dim_size, 3)
    F0 = nn_feat.transpose(-1, -2).sum(dim=-1, keepdim=True)  # (bs, num_samples, dim_size, 1)
    F = torch.cat([F0, F1], dim=-1)  # (bs, num_samples, dim_size, 4)
    F = F / (F0.sum(dim=-2, keepdim=True) + 1e-6)
    return F


def refine_registration(R_hat, t_hat,  args):
    
    if args.dataset == 'kitti':
        dset = SemanticKITTIDataset(data_path=args.data_path,
                                    split=args.split,
                                    cache_data_path=args.cache_data_path,
                                    convert_points_to_grid=False,
                                    skip_invalid_entries=args.skip_invalid_entries_flag,
                                    overied_cache=True)
    else:  # nuscenes
        dset = NuscenesDataset(data_path=args.data_path,
                               split=args.split,
                               cache_data_path=args.cache_data_path,
                               convert_points_to_grid=False,
                               skip_invalid_entries=args.skip_invalid_entries_flag,
                               overied_cache=False)
   
    rre_arr = []
    rte_arr = []
    T_est_arr = []

    for itr in tqdm(range(len(dset))):
        src_pts_raw, _, _, tgt_pts_raw, _, _, _, gt_tform, _ = dset[itr]
        tform_hat = np.zeros((4, 4))
        tform_hat[:3, :3] = R_hat[itr]
        tform_hat[:3, 3] = t_hat[itr]
        tform_hat[3, 3] = 1

        pcd0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts_raw.numpy()))
        pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts_raw.numpy()))
        reg = o3d.pipelines.registration.registration_icp(
            pcd0, pcd1, 0.2, tform_hat,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        new_tform = torch.from_numpy(reg.transformation).float()
        T_est_arr.append(new_tform)
        
        rre = relative_rotation_error(new_tform[:3, :3][None], gt_tform[:3, :3][None])
        rte = (new_tform[:3, 3] - gt_tform[:3, 3]).norm(dim=-1)
        rre_arr.append(rre)
        rte_arr.append(rte)

    rre_tensor = torch.cat(rre_arr)
    rte_tensor = torch.stack(rte_arr)
    T_est_tensor = torch.stack(T_est_arr)
    
    return T_est_tensor, rre_tensor, rte_tensor
    


if __name__ == '__main__':
    # Load Configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, choices=['kitti_test', 'lokitti', 'rotkitti', 'nuscenes_test', 'lonuscenes', 'rotnuscenes'], default="kitti_test")
    args = parser.parse_args()
    config_dict = {'kitti_test': 'configs/benchmarks/test_kitti_config.yaml',
                   'lokitti': 'configs/benchmarks/lokitti_config.yaml',
                   'rotkitti': 'configs/benchmarks/rotkitti_config.yaml',
                   'nuscenes_test': 'configs/benchmarks/test_nuscenes_config.yaml',
                   'lonuscenes': 'configs/benchmarks/lonuscenes_config.yaml',
                   'rotnuscenes': 'configs/benchmarks/rotnuscenes_config.yaml'}
    config_path = config_dict[args.benchmark]
    args = update_namespace_from_yaml(args, config_path)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Evaluate {args.dataset} Benchmark: {args.benchmark} config file: {config_path}")

    # Datasets
    if args.dataset == 'kitti':
        dset = SemanticKITTIDataset(data_path=args.data_path,
                                    split=args.split,
                                    cache_data_path=args.cache_data_path,
                                    skip_invalid_entries=args.skip_invalid_entries_flag)

        dset_no_nksr = SemanticKITTIDataset(data_path=args.data_path,
                                            split=args.split,
                                            cache_data_path=args.cache_data_path,
                                            convert_points_to_grid=False,
                                            skip_invalid_entries=args.skip_invalid_entries_flag,
                                            overied_cache=args.corr_no_nksr)
    else:  # nuscenes
        dset = NuscenesDataset(data_path=args.data_path,
                               split=args.split,
                               cache_data_path=args.cache_data_path,
                               skip_invalid_entries=args.skip_invalid_entries_flag)
        dset_no_nksr = NuscenesDataset(data_path=args.data_path,
                                       split=args.split,
                                       cache_data_path=args.cache_data_path,
                                       convert_points_to_grid=False,
                                       skip_invalid_entries=args.skip_invalid_entries_flag,
                                       overied_cache=args.corr_no_nksr)

    collate_fn = partial(batch_collate_fn_dset, num_matches=args.num_samples, max_pc_size=args.max_pc_size)
    dloader = torch.utils.data.DataLoader(dset,
                                          shuffle=False,
                                          num_workers=args.num_workers,
                                          batch_size=args.batch_size,
                                          collate_fn=collate_fn, drop_last=True)

    # Model
    model = ResUNetSmall2(in_channels=1, out_channels=args.out_ch).to(args.device)
    model.load_state_dict(torch.load(args.model_checkpoint_path)['model_state_dict'])
    model.eval()

    # Regression
    rtume_estimator = ume_kp_layer(ume_knn=args.rtume_nn_max, ume_desc_rad=args.rtume_r_nn, diag_only=True)

    # Loggers
    R_sel_arr = torch.tensor([])
    t_sel_arr = torch.tensor([])

    # Eval
    for itr, data in enumerate(tqdm(dloader)):
        # Fetch
        src_pts, src_seg, src_coords, src_feat, tgt_pts, tgt_seg, tgt_coords, tgt_feat, src_pts_tform, gt_tform, _ = data
        src_stensor = ME.SparseTensor(src_feat, coordinates=src_coords, device=args.device)
        tgt_stensor = ME.SparseTensor(tgt_feat, coordinates=tgt_coords, device=args.device)
        src_seg = src_seg.to(args.device)[..., None]
        tgt_seg = tgt_seg.to(args.device)[..., None]
        src_pts = src_pts.to(args.device)
        tgt_pts = tgt_pts.to(args.device)
        gt_tform = gt_tform.to(args.device)
        src_pts_tform = src_pts_tform.to(args.device)
        R_gt = gt_tform[:, :3, :3]
        t_gt = gt_tform[:, :3, 3]

        # Forward
        with torch.no_grad():
            src_feat = torch.stack(model(src_stensor).decomposed_features, dim=0)
            tgt_feat = torch.stack(model(tgt_stensor).decomposed_features, dim=0)

        # Sample Keypoints + UME matrices
        if args.filter_by_ume_dist_cond:
            num_init_sel = min(10000, min(len(src_pts[0]), len(tgt_pts[0])))
        else:
            num_init_sel = min(min(src_pts.shape[1], tgt_pts.shape[1]), args.ume_n_samples)
        src_inds = np.random.choice(len(src_pts[0]), num_init_sel, replace=False)
        tgt_inds = np.random.choice(len(tgt_pts[0]), num_init_sel, replace=False)
        src_pts_ds = src_pts[0, src_inds][None]
        tgt_pts_ds = tgt_pts[0, tgt_inds][None]
        src_feat_ds = src_feat[:, src_inds]
        tgt_feat_ds = tgt_feat[:, tgt_inds]

        ume_src = my_ume_generation(src_pts, src_pts_ds, src_feat, args)
        ume_tgt = my_ume_generation(tgt_pts, tgt_pts_ds, tgt_feat, args)
        num_kpts = min(ume_src.shape[1], ume_tgt.shape[1])
        ume_src = ume_src[:, :num_kpts]
        src_keypoint_pts = src_pts_ds[:, :num_kpts]
        ume_tgt = ume_tgt[:, :num_kpts]
        tgt_keypoint_pts = tgt_pts_ds[:, :num_kpts]

        # Find Matches
        D = ume_cdist(ume_src, ume_tgt)
        if args.hungarian_matching_flag:
            m = np.zeros((args.batch_size, min(D.shape[1], D.shape[2]), 2))
            for b_idx in range(D.shape[0]):
                src_m_idxs, tgt_m_idxs = linear_sum_assignment(D[b_idx].cpu().numpy())
                m[b_idx, :, 0] = src_m_idxs
                m[b_idx, :, 1] = tgt_m_idxs
            m = torch.from_numpy(m).long().to(args.device)
        else:
            m = D.min(dim=-1)[1]
            m = torch.cat([torch.arange(D.shape[1])[None, :, None].cuda(), m[..., None]], dim=-1)

        num_matches = m.shape[1]
        tgt_matches_keypoint_pts = torch.gather(tgt_keypoint_pts, 1, m[..., 1].unsqueeze(-1).expand(-1, -1, 3))
        src_matches_keypoint_pts = torch.gather(src_keypoint_pts, 1, m[..., 0].unsqueeze(-1).expand(-1, -1, 3))
        ume_tgt = torch.gather(ume_tgt, 1, m[..., 1][..., None, None].expand(-1, -1, 32, 4))
        ume_src = torch.gather(ume_src, 1, m[..., 0][..., None, None].expand(-1, -1, 32, 4))

        if args.filter_by_ume_dist_cond:
            ume_d = D[0, m[0][:, 0], m[0][:, 1]]
            a = (torch.exp((1 - ume_d) / args.tau))
            prob = a / a.sum()
            num_matches = min(src_matches_keypoint_pts.shape[1], args.ume_n_samples)
            cond = np.random.choice(src_matches_keypoint_pts.shape[1], num_matches, replace=False,
                                    p=prob.cpu().numpy())
            src_matches_keypoint_pts = src_matches_keypoint_pts[:, cond]
            tgt_matches_keypoint_pts = tgt_matches_keypoint_pts[:, cond]

            ume_src = ume_src[:, cond]
            ume_tgt = ume_tgt[:, cond]
            num_matches = ume_src.shape[1]

        # Estimate Hypothesis
        G_kp = ume_src.unsqueeze(2)
        H_kp = ume_tgt.unsqueeze(1)
        G, H = G_kp, H_kp
        G = G.reshape(-1, *G.shape[3:])
        H = H.reshape(-1, *H.shape[3:])
        T, _ = batch_estimate_transform_ume_old(G, H)
        rtume_tform = T.view(args.batch_size, num_matches, *T.shape[1:])

        R_rtume = rtume_tform[..., :3, :3]  # (bs, ume_n_samples, 3, 3)
        t_rtume = rtume_tform[..., :3, 3]  # (bs, ume_n_samples, 3)

        # Hypothesis Selection (By Correlator)
        src_pts_raw, _, _, tgt_pts_raw, _, _, _, gt_tform, _ = dset_no_nksr[itr]
        src_coords_raw, src_inds = ME.utils.sparse_quantize(coordinates=src_pts_raw, return_index=True,
                                                            quantization_size=args.corr_ds)
        tgt_coords_raw, tgt_inds = ME.utils.sparse_quantize(coordinates=tgt_pts_raw, return_index=True,
                                                            quantization_size=0.3)

        src_pts_raw = src_pts_raw[src_inds]
        tgt_pts_raw = tgt_pts_raw[tgt_inds]
        src_pts_raw = src_pts_raw[None].cuda()
        tgt_pts_raw = tgt_pts_raw[None].cuda()
        gt_tform = gt_tform[None].cuda()

        ind = knn_points(src_pts_raw, src_pts, K=1)
        src_feat_corr = knn_gather(src_feat, ind[1])[:, :, 0, :]
        ind = knn_points(tgt_pts_raw, tgt_pts, K=1)
        tgt_feat_corr = knn_gather(tgt_feat, ind[1])[:, :, 0, :]

        # DownSample
        num_pts = min(args.pc_corr_max_size, src_pts_raw.shape[1])
        rand_idxs = np.random.choice(src_pts_raw.shape[1], num_pts, replace=False)
        src_pts_raw = src_pts_raw[:, rand_idxs]
        src_feat_corr = src_feat_corr[:, rand_idxs]
        num_pts = min(args.pc_corr_max_size, tgt_pts_raw.shape[1])
        rand_idxs = np.random.choice(tgt_pts_raw.shape[1], num_pts, replace=False)
        tgt_pts_raw = tgt_pts_raw[:, rand_idxs]
        tgt_feat_corr = tgt_feat_corr[:, rand_idxs]

        with torch.no_grad():
            _, _, R_hat_corr, t_hat_corr = pc_fcht(
                pc1_pts=src_pts_raw,
                pc2_pts=tgt_pts_raw,
                pc1_feat=src_feat_corr,
                pc2_feat=tgt_feat_corr,
                rtume_hypotises=rtume_tform,
                gt_tform=gt_tform,
                corr_sigma=args.corr_kernel_sigma,
                args=args)

        R_sel_arr = torch.cat([R_sel_arr, R_hat_corr.cpu()], dim=0)
        t_sel_arr = torch.cat([t_sel_arr, t_hat_corr.cpu()], dim=0)

    T_est_tensor, final_rre_tensor, final_rte_tensor = refine_registration(R_sel_arr, t_sel_arr, args)
    
    # Results
    rr_np = ((final_rre_tensor <= 1.5) & (final_rte_tensor <= 0.6)).float().mean()
    rr_sp = ((final_rre_tensor <= 1) & (final_rte_tensor <= 0.1)).float().mean()

    print(f"Evaluate {args.dataset} Benchmark: {args.benchmark} Results:")
    print(f"N.P: {100 * rr_np:.03f} | S.P: {100 * rr_sp:.03f}")
    print(f"mRRE: {float(final_rre_tensor.mean()):.03f} | mRTE: {float(final_rte_tensor.mean()):.03f}")
    quit()
