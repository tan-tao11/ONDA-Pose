import numpy as np
from scipy.linalg import svd
# from core.self6dpp.losses.ComputeT import get_transform
# from core.self6dpp.losses.GenerateCorrPC import getPointClouds
from ComputeT import get_transform
from GenerateCorrPC import getPointClouds
import cv2
import matplotlib.pyplot as plt


def check_coplanarity(points, tolerance=10):
    """
    Checks if the given 3D points lie on the same plane using numpy and SVD.
    :param points: a list of tuples representing the 3D points
    :param tolerance: a tolerance for checking coplanarity
    :return: True if the points are coplanar, False otherwise
    """
    # Converting the list of points to a numpy array
    A = np.array(points)

    # Subtracting the first point from all the other points
    B = A - A[0]

    # Computing the left singular vectors of B
    _, _, V = np.linalg.svd(B)

    # Checking the rank of the matrix V
    rank = np.linalg.matrix_rank(V)

    # Check the coplanarity using a tolerance
    normal = V[2]
    distances = np.dot(B, normal)
    max_dist = np.max(np.abs(distances))
    result = max_dist < tolerance
    # print("max_dist: ", max_dist)
    return result


def pairwise_distances(P, Q):
    dis = P - Q
    dis = np.linalg.norm(dis, axis=1)
    return dis


def compute_transform_ransac(P, Q, max_iterations=2000, threshold=0.004, sample_size=3, min_inliers=20):
    """
    使用RANSAC算法计算点云Q到点云P的刚体变换矩阵T
    P: 参考点云
    Q: 待配准点云
    max_iterations: 最大迭代次数
    threshold: 迭代中误差阈值
    sample_size: 每次采样的点对数量
    """
    best_T = None
    best_inliers = []
    best_error = np.inf

    for i in range(max_iterations):
        # 1. 随机采样点对
        idx = np.random.choice(P.shape[0], size=sample_size, replace=False)
        P_sample = P[idx]
        Q_sample = Q[idx]

        # 2. 计算变换矩阵
        T = get_transform(P_sample, Q_sample)

        # 3. 计算误差，并标记内点
        Q_transformed = (T[:3, :3] @ Q.T).T + T[:3, 3]
        errors = pairwise_distances(P, Q_transformed)
        inliers = np.where(errors < threshold)[0]

        # 4. 计算内点数，并更新最优解
        if len(inliers) > len(best_inliers):
            # if check_coplanarity(P[inliers]):
            #     # print('1')
            #     continue
            best_inliers = inliers
            best_T = get_transform(P[best_inliers], Q[best_inliers])
            best_error = np.mean(errors[best_inliers])
            if len(inliers) >= min_inliers:
                return best_T, best_inliers, best_error

    return best_T, best_inliers, best_error


if __name__ == '__main__':
    img1_path = '000380_c.png'
    img2_path = '000383_c.png'
    depth1_path = '000380.png'
    depth2_path = '000383.png'
    K = np.array([[572.4114, 0.0, 325.2611],
                  [0.0, 573.57043, 242.04899],
                  [0.0, 0.0, 1.0]])
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    depth1 = cv2.imread(depth1_path, -1)
    depth2 = cv2.imread(depth2_path, -1)
    #print(img1.shape)
    P, Q, kp1, kp2, matches = getPointClouds(img1, depth1, img2, depth2, 100, 1500, K)
    T, inliers, error = compute_transform_ransac(Q, P,  max_iterations=2000, threshold=5, sample_size=3, min_inliers=20)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    new_matches = []
    for i in inliers:
        new_matches.append(matches[i])
    result = cv2.drawMatches(img1, kp1, img2, kp2, new_matches, None, 1)
    plt.imshow(result)
    plt.show()
    print('T:', T)
    print('inliers:', len(inliers))
    print('error: ', error)
