import argparse
import os
import pickle
from tqdm import tqdm
from datasets.kitti.kitti_dataset import SemanticKITTIDataset
from datasets.nuscenes.nuscenes_dataset import NuscenesDataset

if __name__ == '__main__':
    """
    Saves a preprocessed version of the dataset after applying SEM.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default="/francosws/haitman/KITTI/data_odometry_velodyne/dataset/sequences")
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--split', type=str, default="train", choices=['train', 'test', 'val'])
    parser.add_argument('--nksr', type=eval, default=True)
    parser.add_argument('--dataset_mode', type=str, default="kitti", choices=['kitti', 'nuscenes'])
    parser.add_argument('--convert_points_to_grid', type=eval, default=True)
    parser.add_argument('--voxel_size', type=float, default=0.3)
    parser.add_argument('--range_idxs', type=eval, default=[])

    args = parser.parse_args()

    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    SPLIT = args.split
    USE_PC_COMPLETION = args.nksr
    DATASET_MODE = args.dataset_mode
    CONVERT_POINTS_TO_GRID = args.convert_points_to_grid
    VOXEL_SIZE = args.voxel_size
    RANGE_IDXS = args.range_idxs

    if DATASET_MODE == 'nuscenes':
        dset = NuscenesDataset(data_path=DATA_PATH,
                               split=SPLIT,
                               use_pc_completion=USE_PC_COMPLETION,
                               skip_invalid_entries=False,
                               convert_points_to_grid=CONVERT_POINTS_TO_GRID,
                               voxel_size=VOXEL_SIZE)
    else:
        dset = SemanticKITTIDataset(data_path=DATA_PATH,
                                    split=SPLIT,
                                    use_pc_completion=USE_PC_COMPLETION,
                                    skip_invalid_entries=False,
                                    convert_points_to_grid=CONVERT_POINTS_TO_GRID,
                                    voxel_size=VOXEL_SIZE)



    if RANGE_IDXS == []:
        range_vals = range(len(dset))
    else:
        range_vals = range(RANGE_IDXS[0], RANGE_IDXS[1])
    for itr in tqdm(range_vals):
        seq_id, frame0_id, frame1_id = dset.files[itr]
        if DATASET_MODE == 'nuscenes':
            save_dir_path = os.path.join(OUTPUT_PATH, SPLIT, seq_id)
        else:
            save_dir_path = os.path.join(OUTPUT_PATH, SPLIT, f"{seq_id:02d}")
        save_name = f"{frame0_id:06d}_{frame1_id:06d}"

        save_path = os.path.join(save_dir_path, save_name+".pickle")
        if os.path.isfile(save_path):
            print(f'{save_path} - EXIST (Skip)')
            continue

        os.makedirs(save_dir_path, exist_ok=True)

        src_grid_pts, src_sem, src_coords, tgt_grid_pts, tgt_sem, tgt_coords, src_pts_tform, gt_tform, matches = dset[itr]

        save_dict = {"src_pts": src_grid_pts,
                     "src_seg": src_sem,
                     "src_coords": src_coords,
                     "tgt_pts": tgt_grid_pts,
                     "tgt_seg": tgt_sem,
                     "tgt_coords": tgt_coords,
                     "src_pts_tform": src_pts_tform,
                     "gt_tform": gt_tform,
                     "matches": matches}
        with open(os.path.join(save_dir_path, save_name+".pickle"), "wb") as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    quit()
