import cv2
import matplotlib.pyplot as plt
import numpy as np


def getPointClouds(img1, depth1, img2, depth2, min_depth, max_depth, K):
    pixels1 = []
    pixels2 = []
    depth_pixels1 = []
    depth_pixels2 = []

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(desc1, desc2, k=2)
    new_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            pixels1.append([round(pt1[0]), round(pt1[1])])
            depth_pixels1.append(depth1[round(pt1[1]), round(pt1[0])])
            pixels2.append([round(pt2[0]), round(pt2[1])])
            depth_pixels2.append(depth2[round(pt2[1]), round(pt2[0])])
            new_matches.append(m)

    # result = cv2.drawMatches(img1, kp1, img2, kp2, new_matches, None, 1)
    # plt.imshow(result)
    # plt.show()

    pixels1 = np.array(pixels1, np.float32)
    pixels2 = np.array(pixels2, np.float32)
    depth_pixels1 = np.array(depth_pixels1, np.float32)
    depth_pixels2 = np.array(depth_pixels2, np.float32)
    cam1 = pixel2cam(depth_pixels1, pixels1, K)
    cam2 = pixel2cam(depth_pixels2, pixels2, K)
    return cam1, cam2, kp1, kp2, new_matches


def pixel2cam(depth, pixel_coords, K, is_homogeneous=False):
    length = depth.shape[0]
    pixel_coords = np.hstack([pixel_coords, np.ones((length, 1))])
    pixel_coords = np.transpose(pixel_coords, (1, 0))
    cam_coords = (np.linalg.inv(K)@pixel_coords)*depth
    if is_homogeneous:
        cam_coords = np.concatenate([cam_coords, np.ones((1, length))], axis=0)
    cam_coords = np.transpose(cam_coords, (1, 0))
    return cam_coords


if __name__ == "__main__":
    img1 = '000004_c.png'
    img2 = '000009_c.png'
    depth1 = '000004.png'
    depth2 = '000009.png'
    K = np.array([[572.4114, 0.0, 325.2611],
                  [0.0, 573.57043, 242.04899],
                  [0.0, 0.0, 1.0]])
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    depth1 = cv2.imread(depth1, -1)
    depth2 = cv2.imread(depth2, -1)
    getPointClouds(img1, depth1, img2, depth2, 100, 1500, K)