import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import knn_points
from pytorch3d.structures import padded_to_list




def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
  if norm_type == 'BN':
    return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
  elif norm_type == 'IN':
    return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')


class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = 'BN'

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               D=3):
    super(BasicBlockBase, self).__init__()

    self.conv1 = ME.MinkowskiConvolution(
        inplanes, planes, kernel_size=3, stride=stride, dimension=D)
    self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    self.conv2 = ME.MinkowskiConvolution(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        bias=False,
        dimension=D)
    self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = MEF.relu(out)

    out = self.conv2(out)
    out = self.norm2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = MEF.relu(out)

    return out



class BasicBlockBase2(nn.Module):
  expansion = 1
  NORM_TYPE = 'BN'

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               D=3):
    super(BasicBlockBase2, self).__init__()

    self.conv1 = ME.MinkowskiConvolution(
        inplanes, planes, kernel_size=3, stride=stride, dimension=D)
    self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out += residual
    out = MEF.relu(out)

    return out

class BasicBlockBN(BasicBlockBase):
  NORM_TYPE = 'BN'

class BasicBlockBN2(BasicBlockBase2):
  NORM_TYPE = 'BN'

class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = 'IN'


def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3):
  if norm_type == 'BN':
    return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  elif norm_type == 'BN2':
    return BasicBlockBN2(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  elif norm_type == 'IN':
    return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')


class ResUNet(ME.MinkowskiNetwork):
    # For voxel_size=0.03

  BLOCK_NORM_TYPE = 'BN'
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256, 512, 1024]
  TR_CHANNELS = [None, 128, 128, 256, 256, 512, 512]
  KERNEL_SIZES = [7, 5, 5, 5, 5, 5]
  STRIDS = [1, 4, 2, 2, 2, 3]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=True,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    KERNEL_SIZES = self.KERNEL_SIZES
    STRIDS = self.STRIDS
    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=KERNEL_SIZES[0],
        stride=STRIDS[0],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=KERNEL_SIZES[1],
        stride=STRIDS[1],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=KERNEL_SIZES[2],
        stride=STRIDS[2],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=KERNEL_SIZES[3],
        stride=STRIDS[3],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv5 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[4],
        out_channels=CHANNELS[5],
        kernel_size=KERNEL_SIZES[4],
        stride=STRIDS[4],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm5 = get_norm(NORM_TYPE, CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.block5 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[5], CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.conv6 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[5],
        out_channels=CHANNELS[6],
        kernel_size=KERNEL_SIZES[5],
        stride=STRIDS[5],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm6 = get_norm(NORM_TYPE, CHANNELS[6], bn_momentum=bn_momentum, D=D)

    self.block6 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[6], CHANNELS[6], bn_momentum=bn_momentum, D=D)

    self.conv5_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[6],
        out_channels=TR_CHANNELS[6],
        kernel_size=KERNEL_SIZES[-1],
        stride=STRIDS[-1],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm5_tr = get_norm(NORM_TYPE, TR_CHANNELS[6], bn_momentum=bn_momentum, D=D)

    self.block5_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[6], TR_CHANNELS[6], bn_momentum=bn_momentum, D=D)

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[5] + TR_CHANNELS[6],
        out_channels=TR_CHANNELS[5],
        kernel_size=KERNEL_SIZES[-2],
        stride=STRIDS[-2],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[5], TR_CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4] + TR_CHANNELS[5],
        out_channels=TR_CHANNELS[4],
        kernel_size=KERNEL_SIZES[-3],
        stride=STRIDS[-3],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=KERNEL_SIZES[-4],
        stride=STRIDS[-4],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=KERNEL_SIZES[-5],
        stride=STRIDS[-5],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm1_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block1_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.mlp1 = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[2] + CHANNELS[1],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s3 = self.conv3(out)
    out_s3 = self.norm3(out_s3)
    out_s3 = self.block3(out_s3)
    out = MEF.relu(out_s3)

    out_s4 = self.conv4(out)
    out_s4 = self.norm4(out_s4)
    out_s4 = self.block4(out_s4)
    out = MEF.relu(out_s4)

    out_s5 = self.conv5(out)
    out_s5 = self.norm5(out_s5)
    out_s5 = self.block5(out_s5)
    out = MEF.relu(out_s5)

    out_s6 = self.conv6(out)
    out_s6 = self.norm6(out_s6)
    out_s6 = self.block6(out_s6)
    out = MEF.relu(out_s6)

    out = self.conv5_tr(out)
    out = self.norm5_tr(out)
    out = self.block5_tr(out)
    out_s5_tr = MEF.relu(out)
    out = ME.cat(out_s5_tr, out_s5)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)
    out = ME.cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s3_tr = MEF.relu(out)

    out = ME.cat(out_s3_tr, out_s3)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat(out_s2_tr, out_s2)

    out = self.conv1_tr(out)
    out = self.norm1_tr(out)
    out = self.block1_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat(out_s1_tr, out_s1)
    out = self.mlp1(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coordinate_map_key=out.coordinate_map_key,
          coordinate_manager=out.coordinate_manager)
    else:
      return out



class ResUNetSmall(ME.MinkowskiNetwork):
    # For voxel_size=0.03

  BLOCK_NORM_TYPE = 'BN'
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 128, 128, 256, 256, 512]
  KERNEL_SIZES = [3, 3, 3, 3, 3]
  STRIDS = [1, 2, 2, 2, 3]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=True,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    KERNEL_SIZES = self.KERNEL_SIZES
    STRIDS = self.STRIDS
    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=KERNEL_SIZES[0],
        stride=STRIDS[0],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=KERNEL_SIZES[1],
        stride=STRIDS[1],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=KERNEL_SIZES[2],
        stride=STRIDS[2],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=KERNEL_SIZES[3],
        stride=STRIDS[3],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv5 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[4],
        out_channels=CHANNELS[5],
        kernel_size=KERNEL_SIZES[4],
        stride=STRIDS[4],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm5 = get_norm(NORM_TYPE, CHANNELS[-1], bn_momentum=bn_momentum, D=D)

    self.block5 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[-1], CHANNELS[-1], bn_momentum=bn_momentum, D=D)


    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[5],
        out_channels=TR_CHANNELS[5],
        kernel_size=KERNEL_SIZES[4],
        stride=STRIDS[4],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[5], TR_CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4] + TR_CHANNELS[5],
        out_channels=TR_CHANNELS[4],
        kernel_size=KERNEL_SIZES[3],
        stride=STRIDS[3],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=KERNEL_SIZES[2],
        stride=STRIDS[2],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=KERNEL_SIZES[1],
        stride=STRIDS[1],
        dilation=1,
        bias=False,
        dimension=D)
    self.norm1_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block1_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.mlp1 = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[2] + CHANNELS[1],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s3 = self.conv3(out)
    out_s3 = self.norm3(out_s3)
    out_s3 = self.block3(out_s3)
    out = MEF.relu(out_s3)

    out_s4 = self.conv4(out)
    out_s4 = self.norm4(out_s4)
    out_s4 = self.block4(out_s4)
    out = MEF.relu(out_s4)

    out_s5 = self.conv5(out)
    out_s5 = self.norm5(out_s5)
    out_s5 = self.block5(out_s5)
    out = MEF.relu(out_s5)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)
    out = ME.cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s3_tr = MEF.relu(out)

    out = ME.cat(out_s3_tr, out_s3)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat(out_s2_tr, out_s2)

    out = self.conv1_tr(out)
    out = self.norm1_tr(out)
    out = self.block1_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat(out_s1_tr, out_s1)
    out = self.mlp1(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coordinate_map_key=out.coordinate_map_key,
          coordinate_manager=out.coordinate_manager)
    else:
      return out


class ResUNet2(ResUNet):
    # For voxel_size=0.06
    BLOCK_NORM_TYPE = 'BN'
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256, 512, 1024]
    TR_CHANNELS = [None, 128, 128, 256, 256, 512, 512]
    KERNEL_SIZES = [5, 5, 5, 5, 5, 5]
    STRIDS = [1, 2, 2, 2, 2, 3]


class ResUNet3(ResUNet):
    # For voxel_size=0.06
    BLOCK_NORM_TYPE = 'BN'
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 64, 128, 256, 512]
    TR_CHANNELS = [None, 64, 64, 128, 128, 256, 256]
    KERNEL_SIZES = [5, 5, 5, 5, 5, 5]
    STRIDS = [1, 2, 2, 2, 2, 3]

class ResUNet4(ResUNet):
    # For voxel_size=0.06
    BLOCK_NORM_TYPE = 'BN'
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 64, 128, 256, 512]
    TR_CHANNELS = [None, 64, 64, 64, 128, 256, 256]
    KERNEL_SIZES = [3, 3, 3, 5, 5, 5]
    STRIDS = [1, 2, 2, 2, 2, 3]

class ResUNet5(ResUNet):
    # For voxel_size=0.06
    BLOCK_NORM_TYPE = 'BN2'
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 64, 128, 256, 512]
    TR_CHANNELS = [None, 64, 64, 64, 128, 128, 256]
    KERNEL_SIZES = [3, 3, 3, 5, 5, 5]
    STRIDS = [1, 2, 2, 2, 2, 3]







def convert_sparse_batch_to_padded_pc(coords, feat, input_pts):

    device = coords[0].device
    bs = len(coords)

    # Create Padded point cloud (since pcs are with different sizes)
    pc = Pointclouds(points=coords, features=feat)
    pc_sizes = pc.num_points_per_cloud()
    pc_pts_padd = pc.points_padded()  # (bs,max_pc_size, 3)
    pc_feat_padd = pc.features_padded()  # (bs,max_pc_size, feat_dim)
    pc_max_size = max(pc_sizes)
    valid_mask = (torch.arange(pc_max_size, device=device)[None].expand(bs, -1) < pc_sizes[:, None])  # (bs, max_pc_size)

    # Convert coords to pts
    voxel_size = 0.3
    # alpha, beta = convert_coords_to_grid_pts_batch(input_pts, input_coords, voxel_size)
    # pc_pts_padd = pc_pts_padd * alpha[:, None] + beta[:, None]
    pc_pts_padd = pc_pts_padd * voxel_size
    _, _, nn = knn_points(pc_pts_padd, input_pts, K=1, lengths1=pc_sizes, return_nn=True)
    pc_pts_padd = nn.squeeze(2)

    return pc_pts_padd, pc_feat_padd, valid_mask





class ResUNetSmall2(ResUNetSmall):
    # For voxel_size=0.25
    BLOCK_NORM_TYPE = 'BN2'
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128, 128]
    KERNEL_SIZES = [3, 3, 3, 3, 3]
    STRIDS = [1, 2, 2, 2, 3]

if __name__ == '__main__':
    device = 'cuda:0'
    pc_path = "/users_data/haitman/Datasets/My_Kitti360_Odometry/210523_1236/2013_05_28_drive_0000_sync/0000000009_labeled_velo.npy"
    pc = np.load(pc_path)
    pc_label = np.array([LABEL_ID2CAT_ID[l] for l in pc[:, 4]])
    mask = pc_label > 0
    new_pc_pts = torch.from_numpy(pc[mask, :3]).float()
    new_pc_sem = torch.from_numpy(pc_label[mask]).float()
    feats = torch.ones_like(new_pc_sem).unsqueeze(1)

    ds = 0.03
    coords, inds = ME.utils.sparse_quantize(new_pc_pts, return_index=True, quantization_size=ds)
    # Convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    return_coords = new_pc_pts[inds]

    feats = feats[inds]
    # feats = torch.tensor(feats, dtype=torch.float32)
    # coords = torch.tensor(coords, dtype=torch.int32)

    stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

    model = ResUNet(1, 16, normalize_feature=True, D=3).to(device)

    model = ResUNetSmall(1, 32, normalize_feature=True, D=3).to(device)

    out = model(stensor).F


    quit()



