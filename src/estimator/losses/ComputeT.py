import numpy as np

# 中心化点云矩阵
def centerize(points):
    center = np.mean(points, axis=0)
    points_centered = points - center
    return center, points_centered

# 求解变换矩阵 R@Q + t = P
def get_transform(P, Q):
    # 中心化点云矩阵
    center_P, P_centered = centerize(P)
    center_Q, Q_centered = centerize(Q)

    # 计算协方差矩阵
    H = np.dot(Q_centered.T, P_centered)

    # 使用SVD分解求解旋转矩阵
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 合并旋转矩阵和平移向量得到变换矩阵
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = center_P - np.dot(R, center_Q)
    return T


if __name__ == "__main__":
    # 生成两组点云P和Q
    P = np.random.rand(10, 3)
    Q = P

    # 假设P和Q的对应关系已知，可以进行变换
    T = get_transform(P, Q)
    Q_transformed = np.dot(T[:3, :3], Q.T).T + T[:3, 3]

    # 输出变换前后的点云
    print("P:", P)
    print("Q:", Q)
    print("Q_transformed:", Q_transformed)
