# KITTI Install Guideline

## Introduction
We follow previous registration KITTI benchmarks by utilizing the standard train, val, test split (sequences: 0-5,6-7,8-10, respectively).
A short installation guideline based on [SemanticKITTI](https://www.semantic-kitti.org/dataset.html#download) is provided.

## Instructions

1. Download KITTI Odometry Benchmark data from  [SemanticKITTI official page](https://www.semantic-kitti.org/dataset.html#download). Not all the data is relevant, therefore please download just:
   * Odometry Benchmark Velodyne point clouds (80 GB)
   * SemanticKITTI label data (179 MB)
   
2. Organize the downloaded files in one folder named `KITTI` as follows:
    <pre>
        dataset/
        ├── sequences/
        │   ├── 00/
        │   │   ├── velodyne
        │   │   │   ├── 000000.bin
        │   │   │   ├── 000001.bin
        │   │   │   ├── ...
        │   │   ├── labels
        │   │   │   ├── 000000.label
        │   │   │   ├── 000001.label
        │   │   │   ├── ...
        │   │   └── poses.txt
        │   ├── ...
    </pre>

## Acknowledgments
We thank [SemanticKITTI](https://www.semantic-kitti.org/index.html).