import os
import numpy as np
import torch
import yaml
from scipy.spatial import KDTree
from torch.utils.data import Dataset
import pickle
import MinkowskiEngine as ME
import nksr
from pycg import vis
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.general_utils import load_pickle, convert_coords_to_grid_pts, one_side_ball_query_matches, \
    mutual_ball_query_matches


CFG = yaml.safe_load(open("./datasets/kitti/kitti_config.yaml", 'r'))
semantic_kitti_cfg = CFG


class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
    """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
    """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points  # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / (depth + 1e-8))

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)


class SemLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, sem_color_dict=CFG['color_map'], project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
        self.reset()

        # make semantic colors
        max_sem_key = 0
        for key, data in sem_color_dict.items():
            if key + 1 > max_sem_key:
                max_sem_key = key + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for key, value in sem_color_dict.items():
            self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

        # make instance colors
        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0,
                                                high=1.0,
                                                size=(max_inst_id, 3))
        # force zero to a gray-ish color
        self.inst_color_lut[0] = np.full((3), 0.1)

    def reset(self):
        """ Reset scan members. """
        super(SemLaserScan, self).reset()

        # semantic labels
        self.sem_label = np.zeros((0, 1), dtype=np.uint32)  # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # instance labels
        self.inst_label = np.zeros((0, 1), dtype=np.uint32)  # [m, 1]: label
        self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # projection color with semantic labels
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                       dtype=np.int32)  # [H,W]  label
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                       dtype=float)  # [H,W,3] color

        # projection color with instance labels
        self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                        dtype=np.int32)  # [H,W]  label
        self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                        dtype=float)  # [H,W,3] color

    def open_label(self, filename):
        """ Open raw scan and fill in attributes
    """
        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(filename, dtype=np.uint32)
        label = label.reshape((-1))

        # set it
        self.set_label(label)

    def set_label(self, label):
        """ Set points for label not from file but from np
    """
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF  # semantic label in lower half
            self.inst_label = label >> 16  # instance id in upper half
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        # sanity check
        assert ((self.sem_label + (self.inst_label << 16) == label).all())

        if self.project:
            self.do_label_projection()

    def colorize(self):
        """ Colorize pointcloud with the color of each semantic label
    """
        self.sem_label_color = self.sem_color_lut[self.sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

        self.inst_label_color = self.inst_color_lut[self.inst_label]
        self.inst_label_color = self.inst_label_color.reshape((-1, 3))

    def do_label_projection(self):
        # only map colors to labels that exist
        mask = self.proj_idx >= 0

        # semantics
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

        # instances
        self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
        self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]


SEMANTIC_KITTI_COLORMAP = SemLaserScan().sem_color_lut


def load_semantic_kitti_point_cloud(base_path, seq_id, frame_id):
    velo_path = os.path.join(base_path, f"{seq_id:02d}", "velodyne", f"{frame_id:06d}.bin")
    label_path = os.path.join(base_path, f"{seq_id:02d}", "labels", f"{frame_id:06d}.label")

    kitti_scan_loader = SemLaserScan()
    kitti_scan_loader.open_scan(velo_path)
    kitti_scan_loader.open_label(label_path)

    pts = kitti_scan_loader.points
    seg = kitti_scan_loader.sem_label

    # Map to learning labels
    seg = np.array([CFG['learning_map'][l] for l in seg])

    return pts, seg


class SemanticKITTIDataset(Dataset):
    """
    Our Kitti Registration dataset, uses semantic-kitti poses, cna use also semantic information
    """

    # Point Cloud completion Params
    NKSR_DEVICE = "cuda:0"
    NKSR_KNN = 128
    NKSR_FOV_DEG = 90.0
    NKSR_DETAIL_LEVEL = None
    LABEL_COPY_DIST_THR = 3
    NKSR_NUM_SAMPLED_POINTS = 125000
    IN_VALID_IDXS = {'train': [489, 3770, 5132, 5184, 7559, 9080, 9344, 11627],
                     'val': [623],
                     'test': [9],
                     'lokitti': [241, 392, 530],
                     'rotkitti': [394, 441]}

    def __init__(self, data_path, split, voxel_size=0.3,
                 use_pc_completion=False, cache_data_path="", dataset_size=-1, use_augmentations=False,
                 convert_points_to_grid=True, skip_invalid_entries=True, overied_cache=False):
        super(SemanticKITTIDataset, self).__init__()

        self.data_path = data_path
        self.voxel_size = voxel_size
        self.use_pc_completion = use_pc_completion
        self.cache_data_path = cache_data_path
        self.use_augmentations = use_augmentations
        self.convert_points_to_grid = convert_points_to_grid
        self.skip_invalid_entries = skip_invalid_entries
        self.split = split

        # Initiate containers
        self.files = []
        self.kitti_cache = {}
        self.files = np.load(f"datasets/kitti/metadata/{split}_metadata.npy").tolist()
        self.gt_tforms = np.load(f"datasets/kitti/metadata/{split}_gt_tforms.npy")
        not_too_far_cond = (np.linalg.norm(self.gt_tforms[:, :3, 3], axis=-1) <= 50)
        self.files = np.array(self.files)[not_too_far_cond].tolist()
        self.gt_tforms = self.gt_tforms[not_too_far_cond]
        print(f'Num_{split}: {len(self.files)}')

        # Remove pairs without matches
        if self.skip_invalid_entries and self.cache_data_path != "":
            valid_idxs = np.setdiff1d(np.arange(len(self.files)), np.array(self.IN_VALID_IDXS[split]))
            self.files = np.array(self.files)[valid_idxs].tolist()
            self.gt_tforms = self.gt_tforms[valid_idxs]
        if overied_cache:
            self.cache_data_path = ""

        # Use not all dataset
        if dataset_size != -1:
            self.files = self.files[:dataset_size]
            self.gt_tforms = self.gt_tforms[:dataset_size]



    def __len__(self):
        # return len(self.files)
        return len(self.files)

    def __getitem__(self, idx):

        if self.cache_data_path != "":
            if self.use_augmentations:
                return self.cached_getitem_augmented(idx)
            else:
                return self.cached_getitem(idx)
        else:
            return self.preprocess_getitem(idx)

    def preprocess_getitem(self, idx):

        seq_id, frame0_id, frame1_id = self.files[idx]

        # Load Point Clouds
        src_pts, src_sem = load_semantic_kitti_point_cloud(self.data_path, seq_id, frame0_id)
        src_pts = torch.from_numpy(src_pts).float()
        src_sem = torch.from_numpy(src_sem).long()
        tgt_pts, tgt_sem = load_semantic_kitti_point_cloud(self.data_path, seq_id, frame1_id)
        tgt_pts = torch.from_numpy(tgt_pts).float()
        tgt_sem = torch.from_numpy(tgt_sem).long()

        # Load GT Tform
        gt_tform = torch.from_numpy(self.gt_tforms[idx]).float()

        if self.use_pc_completion:
            src_pts, src_sem = self.lidar_point_cloud_completion(src_pts, src_sem)
            tgt_pts, tgt_sem = self.lidar_point_cloud_completion(tgt_pts, tgt_sem)

        # Remove unlabeled points
        unlabeled_mask_src = src_sem == 0
        src_pts = src_pts[~unlabeled_mask_src]
        src_sem = src_sem[~unlabeled_mask_src]
        unlabeled_mask_tgt = tgt_sem == 0
        tgt_pts = tgt_pts[~unlabeled_mask_tgt]
        tgt_sem = tgt_sem[~unlabeled_mask_tgt]

        # Voxlize point clouds
        src_coords, src_inds = ME.utils.sparse_quantize(coordinates=src_pts, return_index=True,
                                                        quantization_size=self.voxel_size)
        tgt_coords, tgt_inds = ME.utils.sparse_quantize(coordinates=tgt_pts, return_index=True,
                                                        quantization_size=self.voxel_size)
        src_sem = src_sem[src_inds]
        if self.convert_points_to_grid:
            src_grid_pts = convert_coords_to_grid_pts(src_pts, src_coords, self.voxel_size)
        else:
            src_grid_pts = src_pts[src_inds]
        tgt_sem = tgt_sem[tgt_inds]
        if self.convert_points_to_grid:
            tgt_grid_pts = convert_coords_to_grid_pts(tgt_pts, tgt_coords, self.voxel_size)
        else:
            tgt_grid_pts = tgt_pts[tgt_inds]

        # Get Matches
        # matches = one_side_ball_query_matches(src_grid_pts, tgt_grid_pts, gt_tform, self.voxel_size/2)
        matches = mutual_ball_query_matches(src_grid_pts, tgt_grid_pts, gt_tform, self.voxel_size / 2)
        matches = torch.from_numpy(matches).long()

        # Create Src Tform
        src_pts_tform = src_grid_pts @ gt_tform[:3, :3].T + gt_tform[:3, 3]

        return src_grid_pts, src_sem, src_coords, tgt_grid_pts, tgt_sem, tgt_coords, src_pts_tform, gt_tform, matches

    def cached_getitem(self, idx):
        seq_id, frame0_id, frame1_id = self.files[idx]

        file_path = os.path.join(self.cache_data_path, self.split, f"{seq_id:02d}",
                                 f"{frame0_id:06d}_{frame1_id:06d}.pickle")

        data_dict = load_pickle(file_path)
        src_pts = data_dict['src_pts']
        src_seg = data_dict['src_seg']
        src_coords = data_dict['src_coords']
        tgt_pts = data_dict['tgt_pts']
        tgt_seg = data_dict['tgt_seg']
        tgt_coords = data_dict['tgt_coords']
        src_pts_tform = data_dict['src_pts_tform']
        gt_tform = data_dict['gt_tform']
        matches = data_dict['matches']

        return src_pts, src_seg, src_coords, tgt_pts, tgt_seg, tgt_coords, src_pts_tform, gt_tform, matches

    def cached_getitem_augmented(self, idx):
        """
        Use Rotate Around Z augmentation
        :param idx:
        :return:
        """

        src_pts, src_seg, src_coords, tgt_pts, tgt_seg, tgt_coords, src_pts_tform, gt_tform, matches = self.cached_getitem(
            idx)

        # Create Random Rotation Over Z-Axis
        rand_z_angle_deg_src = np.random.uniform(low=-180, high=180)
        rand_z_angle_deg_tgt = np.random.uniform(low=-180, high=180)
        rand_z_rot_src = torch.from_numpy(R.from_euler('z', rand_z_angle_deg_src, degrees=True).as_matrix()).float()
        rand_z_rot_tgt = torch.from_numpy(R.from_euler('z', rand_z_angle_deg_tgt, degrees=True).as_matrix()).float()

        src_pts_aug = src_pts @ rand_z_rot_src
        tgt_pts_aug = tgt_pts @ rand_z_rot_tgt

        # Quantizie augmented PCs
        src_coords_aug, src_inds_aug = ME.utils.sparse_quantize(coordinates=src_pts_aug,
                                                                return_index=True,
                                                                quantization_size=self.voxel_size)
        src_grid_pts_aug = convert_coords_to_grid_pts(src_pts_aug, src_coords_aug, self.voxel_size)
        src_seg_aug = src_seg[src_inds_aug]
        tgt_coords_aug, tgt_inds_aug = ME.utils.sparse_quantize(coordinates=tgt_pts_aug,
                                                                return_index=True,
                                                                quantization_size=self.voxel_size)
        tgt_grid_pts_aug = convert_coords_to_grid_pts(tgt_pts_aug, tgt_coords_aug, self.voxel_size)
        tgt_seg_aug = tgt_seg[tgt_inds_aug]

        # New tform after augmentation
        R_gt = gt_tform[:3, :3]
        t_gt = gt_tform[:3, 3]
        R_gt_aug = (rand_z_rot_src.T @ R_gt.T @ rand_z_rot_tgt).T
        t_tgt_aug = t_gt @ rand_z_rot_tgt
        gt_tform_aug = torch.zeros_like(gt_tform)
        gt_tform_aug[:3, :3] = R_gt_aug
        gt_tform_aug[:3, 3] = t_tgt_aug
        gt_tform_aug[3, 3] = 1

        src_pts_tform_aug = src_grid_pts_aug @ R_gt_aug.T + t_tgt_aug

        # Create New Matches
        matches_aug = torch.from_numpy(
            one_side_ball_query_matches(src_grid_pts_aug, tgt_grid_pts_aug, gt_tform_aug, self.voxel_size / 2)).long()

        return (src_grid_pts_aug, src_seg_aug, src_coords_aug,
                tgt_grid_pts_aug, tgt_seg_aug, tgt_coords_aug,
                src_pts_tform_aug, gt_tform_aug, matches_aug)

    def lidar_point_cloud_completion(self, pts, seg):

        # Use NKSR for point cloud reconstruction
        reconstructor = nksr.Reconstructor(self.NKSR_DEVICE)
        reconstructor.chunk_tmp_device = torch.device("cpu")

        input_xyz = pts.to(self.NKSR_DEVICE)
        input_sensor = torch.zeros_like(input_xyz)

        field = reconstructor.reconstruct(
            input_xyz, sensor=input_sensor, detail_level=self.NKSR_DETAIL_LEVEL,
            # Minor configs for better efficiency (not necessary)
            approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True,
            # Chunked reconstruction (if OOM)
            # chunk_size=51.2,
            preprocess_fn=nksr.get_estimate_normal_preprocess_fn(self.NKSR_KNN, self.NKSR_FOV_DEG)
        )

        mesh = field.extract_dual_mesh(mise_iter=1)
        mesh = vis.mesh(mesh.v, mesh.f)
        pc_from_mesh = mesh.sample_points_uniformly(number_of_points=self.NKSR_NUM_SAMPLED_POINTS,
                                                    use_triangle_normal=True)
        new_pts = np.asarray(pc_from_mesh.points).astype("float32")

        # Set labels to new point cloud
        new_seg = np.zeros_like(new_pts[:, 0]).astype(int)
        tree = KDTree(pts)
        dist, idx = tree.query(new_pts, 1)
        valid_label_mask = dist <= self.LABEL_COPY_DIST_THR
        new_seg[valid_label_mask] = seg[idx[valid_label_mask]]

        return torch.from_numpy(new_pts).float(), torch.from_numpy(new_seg).long()



def batch_collate_fn_dset(data, num_matches, max_pc_size=100000):
    """

    :param data: [src_pts, src_sem, src_coords, tgt_pts, tgt_sem, tgt_coords, src_pts_tform, gt_tform, matches]
    :param num_matches:
    :param max_pc_size:
    :return:
    """

    bs = len(data)
    src_pts = []
    src_seg = []
    src_coords = []
    src_feat = []
    tgt_pts = []
    tgt_seg = []
    tgt_coords = []
    tgt_feat = []
    src_pts_tform = []
    matches = []

    # Get Minimal number of points
    src_num_pts = min(min([len(d[0]) for d in data]), max_pc_size)
    tgt_num_pts = min(min([len(d[3]) for d in data]), max_pc_size)

    for b_idx in range(bs):
        d = data[b_idx]
        # Src
        src_rand_idx = np.random.choice(len(d[0]), src_num_pts, replace=False)
        src_pts.append(d[0][src_rand_idx])
        src_seg.append(d[1][src_rand_idx])
        src_coords.append(d[2][src_rand_idx])
        src_feat.append(torch.ones_like(d[0][src_rand_idx, :1]).float())
        src_pts_tform.append(d[6][src_rand_idx])

        # Tgt
        tgt_rand_idx = np.random.choice(len(d[3]), tgt_num_pts, replace=False)
        tgt_pts.append(d[3][tgt_rand_idx])
        tgt_seg.append(d[4][tgt_rand_idx])
        tgt_coords.append(d[5][tgt_rand_idx])
        tgt_feat.append(torch.ones_like(d[3][tgt_rand_idx, :1]).float())

        # Matches (Keep matches after point cloud dilution)
        m = d[8]
        _, m1, idxs1 = np.intersect1d(src_rand_idx, m[:, 0], return_indices=True)
        _, m2, idxs2 = np.intersect1d(tgt_rand_idx, m[idxs1, 1], return_indices=True)
        mm = np.concatenate([m1[idxs2, None], m2[:, None]], axis=1)
        matches.append(mm)

    # Batch - Src
    src_coords, src_feat = ME.utils.sparse_collate(coords=src_coords, feats=src_feat)
    src_pts = torch.stack(src_pts, dim=0)
    src_seg = torch.stack(src_seg, dim=0)
    src_pts_tform = torch.stack(src_pts_tform, dim=0)

    # Batch - Tgt
    tgt_coords, tgt_feat = ME.utils.sparse_collate(coords=tgt_coords, feats=tgt_feat)
    tgt_pts = torch.stack(tgt_pts, dim=0)
    tgt_seg = torch.stack(tgt_seg, dim=0)

    # GT Data
    gt_tform = torch.stack([d[7] for d in data], dim=0)
    # matches = [d[8] for d in data]
    min_num_matches = min([len(m) for m in matches])
    num_matches = min(min_num_matches, num_matches)
    matches = torch.stack([torch.from_numpy(m[np.random.choice(len(m), num_matches, replace=False)]) for m in matches],
                          dim=0)

    return (src_pts, src_seg, src_coords, src_feat,
            tgt_pts, tgt_seg, tgt_coords, tgt_feat,
            src_pts_tform, gt_tform, matches)


if __name__ == '__main__':
    DATA_PATH = "/francosws/haitman/KITTI/data_odometry_velodyne/dataset/sequences"
    POSES_PATH = "/francosws/haitman/KITTI/semmatic_kitti_poses"

    SPLIT = "test"  # "train"

    dset = SemanticKITTIDataset(split=SPLIT, cache_data_path="", dataset_mode='new_v2', refined_gt=False)

    CACHE_DATA_PATH = "/users_data/haitman/Datasets/my_semantic_kitti/dataset_v1_210923_1250"
    dset_train = SemanticKITTIDataset(split="val", dataset_mode='new',
                                      use_pc_completion=True)
    dset_test = SemanticKITTIDataset(split="test", dataset_mode='new',
                                     cache_data_path=CACHE_DATA_PATH)

    dset_test[100]

    output_path = "/users_data/haitman/Datasets/my_semantic_kitti/dataset_v1_210923_1250"
    split = 'test'
    dset = dset_train if split == 'train' else dset_test
    for itr in tqdm(range(len(dset))):
        seq_id, frame0_id, frame1_id = dset.files[itr]

        save_dir_path = os.path.join(output_path, split, f"{seq_id:02d}")
        os.makedirs(save_dir_path, exist_ok=True)
        save_name = f"{frame0_id:06d}_{frame1_id:06d}"
        src_grid_pts, src_sem, src_coords, tgt_grid_pts, tgt_sem, tgt_coords, src_pts_tform, gt_tform, matches = dset[
            itr]

        save_dict = {"src_pts": src_grid_pts,
                     "src_seg": src_sem,
                     "src_coords": src_coords,
                     "tgt_pts": tgt_grid_pts,
                     "tgt_seg": tgt_sem,
                     "tgt_coords": tgt_coords,
                     "src_pts_tform": src_pts_tform,
                     "gt_tform": gt_tform,
                     "matches": matches}
        with open(os.path.join(save_dir_path, save_name + ".pickle"), "wb") as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # t = torch.from_numpy(dset_train.gt_tforms[:, :3, 3])
    # d_train = t.norm(dim=-1)
    # R = torch.from_numpy(dset_train.gt_tforms[:, :3, :3])
    # ang_train = torch.rad2deg(torch.arccos((R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2))
    #
    # t = torch.from_numpy(dset_test.gt_tforms[:, :3, 3])
    # d_test = t.norm(dim=-1)
    # R = torch.from_numpy(dset_test.gt_tforms[:, :3, :3])
    # ang_test = torch.rad2deg(torch.arccos((R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2))
    #
    # fig, ax = plt.subplots(1, 2)
    #
    # ax[0].hist(d_train, 128, histtype='step', label='train', density=True)
    # ax[0].hist(d_test, 128, histtype='step', label='test', density=True)
    # ax[0].set_xlabel('||t|| [m]')
    #
    # ax[1].hist(ang_train, 128, histtype='step', label=f'train(N={len(dset_train)})', density=True)
    # ax[1].hist(ang_test, 128, histtype='step', label=f'test(N={len(dset_test)})', density=True)
    # ax[1].set_xlabel('|theta| [deg]')
    # ax[1].legend()
    # plt.show()
    #
    # seq_id = 0
    # frame_id = 1000
    # velo_path = os.path.join(DATA_PATH, f"{seq_id:02d}", "velodyne", f"{frame_id:06d}.bin")
    # label_path = os.path.join(DATA_PATH, f"{seq_id:02d}", "labels", f"{frame_id:06d}.label")
    #
    # kitti_scan_loader = SemLaserScan()
    # kitti_scan_loader.open_scan(velo_path)
    # kitti_scan_loader.open_label(label_path)
    #
    # pl = pv.Plotter()
    # pl.add_mesh(kitti_scan_loader.points, render_points_as_spheres=True,
    #             scalars=kitti_scan_loader.sem_color_lut[kitti_scan_loader.sem_label], rgba=True)
    # pl.view_xy()
    # pl.show()
    # print(1)

    quit()
