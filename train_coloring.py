import argparse
import json
import torch
import MinkowskiEngine as ME
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
from functools import partial
from datasets.nuscenes.nuscenes_dataset import NuscenesDataset
from utils.general_utils import update_namespace_from_yaml
from datasets.kitti.kitti_dataset import SemanticKITTIDataset, batch_collate_fn_dset
from utils.eval_utils import calc_inliear_ratio
from loss import MyInfoNCELossNoSeg, UMEContrastiveLoss, CubeRegistrationLoss
from models import ResUNetSmall2


def train_one_epoch(epoch, data_loader, model, loss_func, optimizer, summary_writer):
    # Loggers
    loss_acum = 0.0
    pointwise_loss_acum = 0.0
    ume_loss_acum = 0.0
    reg_loss_acum = 0.0
    num_iters = len(data_loader)

    for itr, data in enumerate(tqdm(data_loader)):
        src_pts, src_seg, src_coords, src_feat, tgt_pts, tgt_seg, tgt_coords, tgt_feat, src_pts_tform, gt_tform, matches = data
        if matches.shape[1] == 0:  # Case no matches
            print(f'No Matches in this batch - Skip')
            continue
        src_stensor = ME.SparseTensor(src_feat, coordinates=src_coords, device=DEVICE)
        tgt_stensor = ME.SparseTensor(tgt_feat, coordinates=tgt_coords, device=DEVICE)
        src_seg = src_seg.to(DEVICE)[..., None]
        tgt_seg = tgt_seg.to(DEVICE)[..., None]
        matches = matches.to(DEVICE)
        src_pts = src_pts.to(DEVICE)
        tgt_pts = tgt_pts.to(DEVICE)
        gt_tform = gt_tform.to(DEVICE)

        src_feat = torch.stack(model(src_stensor).decomposed_features, dim=0)
        tgt_feat = torch.stack(model(tgt_stensor).decomposed_features, dim=0)

        pointwise_loss = loss_func(src_feat, src_pts, tgt_feat, matches)

        if USE_UME_LOSS:
            ume_loss, src_keypoint_pts, tgt_keypoint_pts, src_ume, tgt_ume, matched_nn_intersection_ratio, valid_batch_entries = ume_loss_fn(
                src_pts, src_seg, src_feat, tgt_pts, tgt_feat, gt_tform)
            if src_ume.shape[1] == 0:
                print(f'No UME Keypoints where found - skip itr')
                continue
            ume_loss_acum += float(ume_loss)
            if REG_LOSS:
                reg_loss, _, _ = registration_loss_fn(src_pts, src_ume, tgt_pts, tgt_ume, gt_tform,
                                                      matched_nn_intersection_ratio, valid_batch_entries)
                reg_loss_acum += float(reg_loss)
                total_loss = PW_LOSS_WEIGHT * pointwise_loss + UME_LOSS_WEIGHT * ume_loss + REG_LOSS_WEIGHT * reg_loss
            else:
                total_loss = PW_LOSS_WEIGHT * pointwise_loss + UME_LOSS_WEIGHT * ume_loss

        else:
            total_loss = pointwise_loss

        loss_acum += float(total_loss)
        pointwise_loss_acum += float(pointwise_loss)

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        # max_norm = 5.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Loggs
        summary_writer.add_scalar('train/total_loss', float(total_loss.item()), epoch * num_iters + itr)
        summary_writer.add_scalar('train/pointwise_loss', float(pointwise_loss.item()), epoch * num_iters + itr)
        if USE_UME_LOSS:
            summary_writer.add_scalar('train/ume_loss', float(ume_loss.item()), epoch * num_iters + itr)
        if REG_LOSS:
            summary_writer.add_scalar('train/reg_loss', float(reg_loss.item()), epoch * num_iters + itr)

        if (itr + 1) % 10 == 0:
            print(f'Total Loss: {loss_acum / 10:.04f} |'
                  f' Pointwise Loss: {pointwise_loss_acum / 10:.04f} |'
                  f' UME Loss: {ume_loss_acum / 10:.04f} |'
                  f' REG Loss {reg_loss_acum / 10: .04f}')
            loss_acum = 0.0
            pointwise_loss_acum = 0.0
            ume_loss_acum = 0.0
            reg_loss_acum = 0.0

    print(f'Done Train Epoch: {epoch + 1}')


def eval_one_epoch(epoch, data_loader, model, loss_func, summary_writer):
    # Loggers
    loss_acum = 0.0
    pointwise_loss_acum = 0.0
    ume_loss_acum = 0.0
    inlear_ratio_acum = 0.0
    reg_loss_acum = 0.0
    rre_acum = 0.0
    rte_acum = 0.0
    reg_acc_acum = 0.0
    num_iters = len(data_loader)

    for itr, data in enumerate(tqdm(data_loader)):

        src_pts, src_seg, src_coords, src_feat, tgt_pts, tgt_seg, tgt_coords, tgt_feat, src_pts_tform, gt_tform, matches = data
        if matches.shape[1] == 0:  # Case no matches
            print(f'No Matches in this batch - Skip')
            continue
        src_stensor = ME.SparseTensor(src_feat, coordinates=src_coords, device=DEVICE)
        tgt_stensor = ME.SparseTensor(tgt_feat, coordinates=tgt_coords, device=DEVICE)
        src_seg = src_seg.to(DEVICE)[..., None]
        tgt_seg = tgt_seg.to(DEVICE)[..., None]
        matches = matches.to(DEVICE)
        src_pts = src_pts.to(DEVICE)
        tgt_pts = tgt_pts.to(DEVICE)
        gt_tform = gt_tform.to(DEVICE)

        with torch.no_grad():
            velo_feat = torch.stack(model(src_stensor).decomposed_features, dim=0)
            ref_feat = torch.stack(model(tgt_stensor).decomposed_features, dim=0)

        pointwise_loss = loss_func(velo_feat, src_pts, ref_feat, matches)

        if USE_UME_LOSS:
            ume_loss, src_keypoint_pts, tgt_keypoint_pts, src_ume, tgt_ume, matched_nn_intersection_ratio, valid_batch_entries = ume_loss_fn(
                src_pts, src_seg, velo_feat, tgt_pts,
                ref_feat, gt_tform)
            if src_ume.shape[1] == 0:
                print(f'No UME Keypoints where found - skip itr')
                continue
            ume_loss_acum += float(ume_loss)

            if REG_LOSS:
                reg_loss, rre, rte = registration_loss_fn(src_pts, src_ume, tgt_pts, tgt_ume, gt_tform,
                                                          matched_nn_intersection_ratio, valid_batch_entries)
                reg_acc = ((rre <= REG_ROT_THR_DEG) & (rte <= REG_TRANS_THR_M)).float().mean()
                reg_loss_acum += float(reg_loss)
                rre_acum += float(rre.median(dim=-1)[0].mean())
                rte_acum += float(rte.median(dim=-1)[0].mean())
                reg_acc_acum += float(reg_acc)

                total_loss = PW_LOSS_WEIGHT * pointwise_loss + UME_LOSS_WEIGHT * ume_loss + REG_LOSS_WEIGHT * reg_loss
            else:
                total_loss = PW_LOSS_WEIGHT * pointwise_loss + UME_LOSS_WEIGHT * ume_loss
        else:
            total_loss = pointwise_loss
            valid_batch_entries = torch.ones_like(gt_tform[:, 0, 0]).bool()

        pointwise_loss_acum += float(pointwise_loss)
        loss_acum += float(total_loss)

        # Calc Inlear Ratio
        if CALC_INLEAR_RATIO_EVAL:
            gt_tform = gt_tform[valid_batch_entries]
            src_inputs = {'pts': src_pts[valid_batch_entries], 'seg': src_seg[valid_batch_entries],
                          'feat': velo_feat[valid_batch_entries]}
            tgt_inputs = {'pts': tgt_pts[valid_batch_entries], 'seg': [valid_batch_entries],
                          'feat': ref_feat[valid_batch_entries]}
            inlier_ratio = calc_inliear_ratio(src_inputs, tgt_inputs,
                                              src_pts_tform, gt_tform,
                                              UME_R_NN, UME_MAX_NN, UME_MIN_NN,
                                              eval_num_kpts=EVAL_NUM_KPTS,
                                              inlear_thr=EVAL_INLEAR_THR)
            inlear_ratio_acum += float(inlier_ratio.mean())
        else:
            inlear_ratio_acum = 0.0

        if (itr + 1) % 10 == 0:
            print(f'Total Loss: {loss_acum / (itr + 1):.04f} |'
                  f' Pointwise Loss: {pointwise_loss_acum / (itr + 1):.04f} |'
                  f' UME Loss: {ume_loss_acum / (itr + 1):.04f} |'
                  f' Reg Loss: {reg_loss_acum / (itr + 1):.04f} |'
                  f' Inlear Ratio: {100 * inlear_ratio_acum / (itr + 1):.02f}')

    # Loggs
    valid_loss = float(loss_acum) / num_iters
    valid_pw_loss = float(loss_acum) / num_iters
    valid_inlear_ratio = float(inlear_ratio_acum) / num_iters

    summary_writer.add_scalar('valid/total_loss', valid_loss, epoch)
    summary_writer.add_scalar('valid/pointwise_loss', valid_pw_loss, epoch)
    summary_writer.add_scalar('valid/inlear_ratio', valid_inlear_ratio, epoch)
    if USE_UME_LOSS:
        valid_ume_loss = float(loss_acum) / num_iters
        summary_writer.add_scalar('valid/ume_loss', valid_ume_loss, epoch)
    else:
        valid_ume_loss = 0.0

    if REG_LOSS:
        valid_reg_loss = float(reg_loss_acum) / num_iters
        valid_rre = float(rre_acum) / num_iters
        valid_rte_loss = float(rte_acum) / num_iters
        valid_reg_acc = float(reg_acc_acum) / num_iters
        summary_writer.add_scalar('valid/reg_loss', valid_reg_loss, epoch)
        summary_writer.add_scalar('valid/rre', valid_rre, epoch)
        summary_writer.add_scalar('valid/rte', valid_rte_loss, epoch)
        summary_writer.add_scalar('valid/chr', valid_reg_acc, epoch)
    else:
        valid_reg_loss = 0.0

    print(f'Done Valid Epoch: {epoch + 1}')
    return valid_loss, valid_pw_loss, valid_ume_loss, valid_reg_loss, valid_inlear_ratio, valid_reg_acc


def save_model(model, save_path, save_name):
    full_save_path = os.path.join(save_path, save_name)
    torch.save(model.state_dict(), full_save_path)


def save_checkpoint(epoch, total_loss, model, optimizer, save_path, save_name):
    full_save_path = os.path.join(save_path, save_name)
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'total_loss': total_loss}
    full_save_path = full_save_path.replace('.pth', '_checkpoint.pth')
    torch.save(checkpoint, full_save_path)


def create_params_dict():
    config_dict = {'run_name': run_name,
                   'seed': SEED,
                   'device': DEVICE,
                   'num_workers': NUM_WORKERS,
                   'data_path': CACHE_DATA_PATH,
                   'checkpoint_path': out_path,
                   'num_epochs': NUM_EPOCHS,
                   'num_samples': NUM_SAMPLES,
                   'batch_size': BATCH_SIZE,
                   'num_out_ch': OUT_CH,
                   'tau': TAU,
                   'USE_UME_LOSS': USE_UME_LOSS,
                   'ume_n_samples': UME_N_SAMPLES,
                   'UME_MAX_NN': UME_MAX_NN,
                   'UME_MIN_NN': UME_MIN_NN,
                   'UME_R_NN': UME_R_NN,
                   'PW_LOSS_WEIGHT': PW_LOSS_WEIGHT,
                   'UME_LOSS_WEIGHT': UME_LOSS_WEIGHT,
                   'LR': LR,
                   'WEIGHT_DECAY': WEIGHT_DECAY,
                   'model_type': model.__class__.__name__}
    with open(os.path.join(out_path, 'run_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=6)
    return config_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, choices=['kitti', 'nuscenes'], default="kitti")
    args = parser.parse_args()
    config_dict = {'kitti': 'configs/train/train_kitti_config.yaml',
                   'nuscenes': 'configs/train/train_nuscenes_config.yaml'}
    args.config_path = config_dict[args.config]
    args = update_namespace_from_yaml(args, args.config_path)
    print(f"Train {args.dataset} config file: {args.config_path}")

    DATA_PATH = args.data_path
    DATASET = args.dataset
    CACHE_DATA_PATH = args.cache_data_path
    SEED = args.random_seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Dataset
    NUM_SAMPLES = args.num_pw_samples
    BATCH_SIZE = args.batch_size
    EVAL_BATCH_SIZE = BATCH_SIZE if args.eval_batch_size == -1 else args.eval_batch_size
    MAX_PC_SIZE = args.max_pc_size
    USE_AUG = args.use_aug
    CALC_INLEAR_RATIO_EVAL = args.calc_inlear_ratio_eval
    EVAL_NUM_KPTS = args.eval_num_kpts
    EVAL_INLEAR_THR = args.eval_inlear_thr
    SKIP_INVALID_ENTRIES = args.skip_invalid_entries

    collate_fn = partial(batch_collate_fn_dset, num_matches=NUM_SAMPLES, max_pc_size=MAX_PC_SIZE)

    # Misc
    DEVICE = args.device
    NUM_WORKERS = args.num_workers
    timestamp = time.strftime('%d%m%y_%H%M%S')
    run_name = f"{args.run_name}_{DATASET}_{timestamp}"
    out_path = os.path.join(args.output_path, run_name)
    os.makedirs(out_path)
    TB_PATH = out_path
    CHECKPOINT_PATH = out_path
    NUM_EPOCHS = args.num_epochs
    TRAIN_SIZE = args.train_size
    VAL_SIZE = args.val_size

    # Model
    OUT_CH = args.out_channels

    # Loss
    TAU = args.tau
    TAU_UME = args.tau_ume
    TAU_UME_NEG = args.tau_ume_neg
    UME_N_SAMPLES = args.ume_n_samples
    UME_MAX_NN = args.ume_max_nn
    UME_MIN_NN = args.ume_min_nn
    UME_R_NN = args.ume_r_nn
    USE_UME_LOSS = args.use_ume_loss
    PW_LOSS_WEIGHT = args.pw_loss_weight
    UME_LOSS_WEIGHT = args.ume_loss_weight
    NEG_EUCLID_DIST = 5
    REG_LOSS = args.use_reg_loss
    REG_LOSS_WEIGHT = args.reg_loss_weight
    REG_ROT_THR_DEG = 5.0
    REG_TRANS_THR_M = 0.6
    REG_LOSS_INTERSECTION_THR = args.reg_loss_intersection_thr
    REG_LOSS_CUBE_R = args.reg_loss_cube_r
    RESUME_TRAIN_PATH = args.resume_train_path
    START_EPOCH = 0

    # Optimizer
    LR = args.lr
    WEIGHT_DECAY = 0.0

    if DATASET == 'kitti':
        dset_train = SemanticKITTIDataset(data_path=DATA_PATH,
                                          split='train',
                                          cache_data_path=CACHE_DATA_PATH,
                                          dataset_size=TRAIN_SIZE,
                                          use_augmentations=USE_AUG,
                                          skip_invalid_entries=SKIP_INVALID_ENTRIES)

        dset_valid = SemanticKITTIDataset(data_path=DATA_PATH,
                                          split='val',
                                          cache_data_path=CACHE_DATA_PATH,
                                          dataset_size=VAL_SIZE)
    else:  # nuscenes
        dset_train = NuscenesDataset(data_path=DATA_PATH,
                                     split='train',
                                     cache_data_path=CACHE_DATA_PATH,
                                     dataset_size=TRAIN_SIZE,
                                     use_augmentations=USE_AUG,
                                     skip_invalid_entries=SKIP_INVALID_ENTRIES)

        dset_valid = NuscenesDataset(data_path=DATA_PATH,
                                     split='val',
                                     cache_data_path=CACHE_DATA_PATH,
                                     dataset_size=VAL_SIZE,
                                     use_augmentations=False,
                                     skip_invalid_entries=SKIP_INVALID_ENTRIES)

    dloader_train = torch.utils.data.DataLoader(dset_train,
                                                shuffle=True,
                                                num_workers=NUM_WORKERS,
                                                batch_size=BATCH_SIZE,
                                                collate_fn=collate_fn,
                                                pin_memory=True)

    dloader_valid = torch.utils.data.DataLoader(dset_valid,
                                                shuffle=False,
                                                num_workers=NUM_WORKERS,
                                                batch_size=BATCH_SIZE,
                                                collate_fn=collate_fn,
                                                pin_memory=True)

    # Model
    model = ResUNetSmall2(in_channels=1, out_channels=OUT_CH).to(DEVICE)

    # If Resume Train
    if RESUME_TRAIN_PATH != '':
        print(f'Resume Model: {RESUME_TRAIN_PATH}')
        if "_checkpoint" in RESUME_TRAIN_PATH:
            model.load_state_dict(torch.load(RESUME_TRAIN_PATH)['model_state_dict'])
            START_EPOCH = torch.load(RESUME_TRAIN_PATH)['epoch']
            print(f'Continue from Epoch: {START_EPOCH}')
        else:
            model.load_state_dict(torch.load(RESUME_TRAIN_PATH))

    # Losses
    point_wise_loss_fn = MyInfoNCELossNoSeg(num_samples=NUM_SAMPLES, tau=TAU, neg_euclid_dist=NEG_EUCLID_DIST)
    ume_loss_fn = UMEContrastiveLoss(num_samples=UME_N_SAMPLES,
                                     max_nn=UME_MAX_NN,
                                     min_nn=UME_MIN_NN,
                                     nn_r=UME_R_NN,
                                     tau=TAU_UME,
                                     tau_neg=TAU_UME_NEG)

    registration_loss_fn = CubeRegistrationLoss(rtume_max_nn=UME_MAX_NN, rtume_r_nn=UME_R_NN,
                                                nn_inter_ratio_thr=REG_LOSS_INTERSECTION_THR,
                                                cube_scale=REG_LOSS_CUBE_R)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # If Resume Train
    if RESUME_TRAIN_PATH != '':
        print(f'Resume Optimizer: {RESUME_TRAIN_PATH}')
        if "_checkpoint" in RESUME_TRAIN_PATH:
            optimizer.load_state_dict(torch.load(RESUME_TRAIN_PATH)['optimizer_state_dict'])

    # Logs
    config_dict = create_params_dict()
    summary_writer = SummaryWriter(log_dir=TB_PATH)
    best_total_loss = np.inf
    best_pw_loss = np.inf
    best_ume_loss = np.inf
    best_reg_loss = np.inf
    best_inlear_ratio = 0.0
    best_mchr = 0.0

    for epoch in range(START_EPOCH, NUM_EPOCHS):
        model.train()
        train_one_epoch(epoch, dloader_train, model, point_wise_loss_fn, optimizer, summary_writer)
        model.eval()
        total_loss_valid, pw_loss_valid, ume_loss_valid, valid_reg_loss, inlear_ratio_valid, mchr_valid = eval_one_epoch(
            epoch, dloader_valid, model, point_wise_loss_fn, summary_writer)

        # Save Best Model
        if total_loss_valid < best_total_loss:
            best_total_loss = total_loss_valid
            save_checkpoint(epoch, total_loss_valid, model, optimizer, CHECKPOINT_PATH, "best_total_loss.pth")
        if pw_loss_valid < best_pw_loss:
            best_pw_loss = pw_loss_valid
            save_checkpoint(epoch, total_loss_valid, model, optimizer, CHECKPOINT_PATH, "best_pointwise_loss.pth")
        if ume_loss_valid < best_ume_loss:
            best_ume_loss = ume_loss_valid
            save_checkpoint(epoch, total_loss_valid, model, optimizer, CHECKPOINT_PATH, "best_ume_loss.pth")
        if valid_reg_loss < best_reg_loss:
            best_reg_loss = valid_reg_loss
            save_checkpoint(epoch, total_loss_valid, model, optimizer, CHECKPOINT_PATH, "best_reg_loss.pth")
        if inlear_ratio_valid > best_inlear_ratio:
            best_inlear_ratio = inlear_ratio_valid
            save_checkpoint(epoch, total_loss_valid, model, optimizer, CHECKPOINT_PATH, "best_inlear_ratio.pth")
        if mchr_valid > best_mchr:
            best_mchr = mchr_valid
            save_checkpoint(epoch, total_loss_valid, model, optimizer, CHECKPOINT_PATH, "best_mCHR.pth")

        # Save Last Model
        save_checkpoint(epoch, total_loss_valid, model, optimizer, CHECKPOINT_PATH, "last_epoch.pth")

    quit()
