import open3d as o3d
import numpy as np
import transforms3d as tf3d


def pointcloud_to_o3d(pointcloud):
    pointcloud_xyz = pointcloud[:, :3]
    if pointcloud.shape[1] == 4:
        color = np.zeros([pointcloud.shape[0], 3], dtype=float)
        color[:, 0] = pointcloud[:, 3]
        color[:, 1] = pointcloud[:, 3]
        color[:, 2] = pointcloud[:, 3]
    elif pointcloud.shape[1] > 4:
        color = pointcloud[:, 3:]
    else:
        color = None

    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(pointcloud_xyz)
    if color is not None:
        points.colors = o3d.utility.Vector3dVector(color)
    return points


def transform_pointcloud(pointcloud, T):
    pointcloud_pcd: o3d.geometry.PointCloud = pointcloud_to_o3d(pointcloud)
    pointcloud_pcd.transform(T)
    if pointcloud.shape[1] > 3:
        return np.concatenate([np.asarray(pointcloud_pcd.points), pointcloud[:, 3:]], axis=1)
    else:
        return np.asarray(pointcloud_pcd.points)


def pose_to_matrix(pose, axes="sxyz"):
    matrix = np.eye(4, dtype=pose.dtype)
    matrix[:3, 3] = pose[:3]

    if len(pose) == 6:
        matrix[:3, :3] = tf3d.euler.euler2mat(pose[3], pose[4], pose[5], axes=axes)
    else:
        matrix[:3, :3] = tf3d.quaternions.quat2mat(pose[3:])

    return matrix
