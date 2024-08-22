# Nuscenes Install Guideline

## Introduction
To simplify coding, we convert the original Nuscenes dataset format into a simmialr format as the KITTI dataset.
Dataset is split into train\val\test folders, each folder contains sequence's folders with the recording files.
The final format keeps a similar structure to this of the KITTI dataset with lidar point cloud, sematic-segmentation information and pose information for each frame.

## Instructions

1. Download nuScenes data from  [nuscenes official page](https://www.nuscenes.org/nuscenes#download). Not all the data is relevant, therefore please download just:
   * Full dataset (v1.0)
     * Trainval:
       * Metadata
       * Lidar blobs only for part 1-10
     * Test
       * Metadata
       * Lidar blobs only
   * nuScenes-lidarseg
     * All
       * Metadata and sensor file blobs
2. Organize the downloaded files in one folder named `NUSCENES` as follows:
    <pre>
        NUSCENES/
        ├── lidarseg/
        │   └── v1.0-trainval/
        │       ├── file1_lidarseg.bin
        │       ├── file2_lidarseg.bin
        │       └── ...
        ├── samples/
        │   └── LIDAR_TOP/
        │       ├── file1.pcd.bin
        │       ├── file2.pcd.bin
        │       └── ...
        ├── sweeps/
        │   └── LIDAR_TOP/
        │       ├── file1.pcd.bin
        │       ├── file2.pcd.bin
        │       └── ...
        ├── v1.0-test/
        │   ├── metadata files...
        │   └── ...
        └── v1.0-trainval/
            ├── metadata files...
            └── ...
    </pre>

3. Download and install in a new conda-env the [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit):
    <pre>
    git clone https://github.com/nutonomy/nuscenes-devkit.git
    conda create -n nuscenes-devkit python=3.8
    conda activate nuscenes-devkit
    pip install nuscenes-devkit
   </pre>

4. Copy our minimal script to generate the nuScenes KITTI like version:
    <pre>cp .datasets/nuscenes/export_kitti_minimal.py ./nuscenes-devkit/python-sdk/nuscenes/scripts/export_kitti_minimal.py</pre>
5. Run 'export_kitti_minimal.py' to generate the nuScenes KITTI like version:
    <pre>
    cd ./nuscenes-devkit/python-sdk
    conda activate nuscenes-devkit
    python nuscenes/scripts/export_kitti_minimal.py --split train --input_path 'path_to_NUSCENS_raw_folder' --output_path 'path_to_output_folder'
    python nuscenes/scripts/export_kitti_minimal.py --split val --input_path 'path_to_NUSCENS_raw_folder' --output_path 'path_to_output_folder'
    python nuscenes/scripts/export_kitti_minimal.py --split test --input_path 'path_to_NUSCENS_raw_folder' --output_path 'path_to_output_folder'
   </pre>
    This may take a while.

## Acknowledgments
We thank [GCL](https://github.com/liuQuan98/GCL) and [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for the code of the dataset conversion.