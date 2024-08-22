import pickle
import yaml
from scipy.spatial import KDTree
import numpy as np
import torch


def str2bool(v):
    return v.lower() in ('true', '1')


def str2eval(v):
    return eval(v)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(filename, file):
    with open(filename, "wb") as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_coords_to_grid_pts(pts, coords, ds):
    a = pts.max(dim=0)[0] - 0.5 * ds
    b = coords.max(dim=0)[0]
    c = pts.min(dim=0)[0] + 0.5 * ds
    d = coords.min(dim=0)[0]
    alpha = (a - c) / (b - d)
    beta = (b * c - a * d) / (b - d)
    grid_pts = coords * alpha + beta
    return grid_pts.float()


def one_side_ball_query_matches(src_pts, tgt_pts, trans, search_voxel_size):
    src_trans_pts = src_pts @ trans[:3, :3].T + trans[:3, 3]
    tree = KDTree(tgt_pts)
    dist, idx = tree.query(src_trans_pts, 1)
    match_inds = np.concatenate((np.arange(src_trans_pts.shape[0])[dist < search_voxel_size, None],
                                 idx[dist < search_voxel_size, None]), axis=-1)
    return match_inds


def mutual_ball_query_matches(src_pts, tgt_pts, tform, voxel_size):
    matches_s2t = one_side_ball_query_matches(src_pts, tgt_pts, tform, voxel_size)
    matches_t2s = one_side_ball_query_matches(tgt_pts, src_pts, torch.linalg.inv(tform), voxel_size)

    matches = []
    for itr in range(len(matches_s2t)):
        i = matches_s2t[itr, 0]
        j = matches_s2t[itr, 1]
        jj = np.where(matches_t2s[:, 0] == j)[0]
        if matches_t2s[jj, 1] == i:
            matches.append((i, j))

    return np.array(matches)


def update_namespace_from_yaml(args, yaml_path):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    for key, value in yaml_data.items():
        setattr(args, key, value)

    return args
