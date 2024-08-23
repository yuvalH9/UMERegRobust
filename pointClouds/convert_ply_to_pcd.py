import open3d as o3d
import os


def convert_ply_to_pcd(directory):
    # Walk through the directory tree
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".ply"):
                # Construct full file path
                ply_path = os.path.join(root, filename)

                # Load the PLY file
                ply_point_cloud = o3d.io.read_point_cloud(ply_path)

                # Create the output PCD file path
                pcd_filename = os.path.splitext(filename)[0] + ".pcd"
                pcd_path = os.path.join(root, pcd_filename)

                # Save the point cloud as a PCD file
                o3d.io.write_point_cloud(pcd_path, ply_point_cloud)

                print(f"Converted {ply_path} to {pcd_path}")

if __name__ == "__main__":
    # Specify the directory containing the .ply files
    directory = "/home/haitman/Git/PaSCo/pointClouds"

    # Convert all .ply files in the specified directory
    convert_ply_to_pcd(directory)
