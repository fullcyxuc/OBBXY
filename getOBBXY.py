import os
import numpy as np
import pandas as pd
import open3d as o3d

def pca(points):
    points_ = points[:, :].transpose(1, 0)
    cov_m = np.cov(points_)
    eig_values, eig_vectors = np.linalg.eig(cov_m)
    return eig_values, eig_vectors

def getOBB(points):
    """
    get the obb that only rotate along z-axis
    :param points:
    :return:
        oriented bounding box (obb) in open3d format
        pointcloud of corners of obb in open3d format
    """

    # only consider on xy-plane
    xy = points[:, :2]
    z = points[:, 2]

    # get the transform matrix of xy
    _, tm_xy = pca(xy)
    # print(tm_xy)

    # transform xy to the new coordinate and get the new corner
    new_xy = np.matmul(xy, tm_xy)
    xmax = np.max(new_xy[:, 0])
    ymax = np.max(new_xy[:, 1])
    xmin = np.min(new_xy[:, 0])
    ymin = np.min(new_xy[:, 1])
    zmax = np.max(z)
    zmin = np.min(z)

    corner_list = np.array([
        [xmax, ymax, zmax],
        [xmax, ymax, zmin],
        [xmax, ymin, zmax],
        [xmax, ymin, zmin],
        [xmin, ymax, zmax],
        [xmin, ymax, zmax],
        [xmin, ymin, zmax],
        [xmin, ymin, zmin],
    ])

    # get the extent (length, width, height) of new x, new y, and z
    ext_x = xmax - xmin
    ext_y = ymax - ymin
    ext_z = zmax - zmin
    extent = np.array([ext_x, ext_y, ext_z])

    # get the new center
    center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])

    # get the transform matrix of xyz
    tm_xyz = np.array([
        [tm_xy[0][0], tm_xy[0][1], 0.0],
        [tm_xy[1][0], tm_xy[1][1], 0.0],
        [0.0, 0.0, 1.0]
    ])

    # use the transform matrix to transform the new center, new corner back to the old coordinate
    corner_list_old = np.matmul(corner_list, tm_xyz.T)
    center_old = np.matmul(center, tm_xyz.T)
    # print(center, center_old)

    to_visualize = []
    obb_o3d = o3d.geometry.OrientedBoundingBox(center_old, tm_xyz, extent)
    obb_o3d.color = (1, 0, 0)
    # original_pcd = o3d.geometry.PointCloud()
    # original_pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    # original_pcd.colors = o3d.utility.Vector3dVector(data[:, 3:] / 255)
    corner_pcd = o3d.geometry.PointCloud()
    corner_pcd.points = o3d.utility.Vector3dVector(corner_list_old)
    corner_pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for _ in range(len(corner_list_old))]))

    # to_visualize.append(obb_o3d)
    # to_visualize.append(original_pcd)
    # to_visualize.append(corner_pcd)
    # o3d.visualization.draw_geometries(to_visualize)

    return obb_o3d, corner_pcd



if __name__ == "__main__":
    data = pd.read_csv(r"test.txt", header=None, delimiter=" ").to_numpy().astype(np.float32)
    getOBB(data)

