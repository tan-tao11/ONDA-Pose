import torch
import cv2
import numpy as np
from torch import nn


def ReconstructNewView(depth_path, color_path, K, R, t):
    # read the depth image
    depth_img = cv2.imread(depth_path, -1)
    color_img = cv2.imread(color_path)
    height, width = depth_img.shape
    channel = 3
    new_view = torch.zeros((height, width, channel))
    points_3d = []
    for x in range(width):
        for y in range(height):
            z = depth_img[x, y]
            points_h = torch.tensor([x, y, 1], dtype=torch.float32)
            # the 3D coordinate in the first camera coordinate
            point_3d_i = torch.inverse(K)@points_h*z
            # the 3D coordinate in the second camera coordinate
            point_3d_j = torch.matmul(R, point_3d_i) + t


def meshgrid(height, width, is_homogeneous=True):
    """

    :param height: height of the grid
    :param width: width of the grid
    :return:x,y grid coordinates [height, width, 2]
    """
    x_t = torch.ones((height, 1)) @ torch.reshape(torch.tensor([i for i in range(width)], dtype=torch.float32), (1, width))
    y_t = torch.reshape(torch.tensor([i for i in range(height)], dtype=torch.float32), (height, 1)) @ torch.ones((1, width))
    if is_homogeneous:
        coords = torch.stack([x_t, y_t, torch.ones((height, width), dtype=torch.float32)], dim=0)
    else:
        coords = torch.stack([x_t, y_t], dim=0)
    return coords


def pixel2cam(depth, pixel_coords, K, is_homogeneous=True):
    height, width = depth.shape
    depth = torch.reshape(depth, (1, -1))
    pixel_coords = torch.reshape(pixel_coords, (3, -1))
    cam_coords = (torch.inverse(K)@pixel_coords)*depth
    if is_homogeneous:
        cam_coords = torch.cat([cam_coords, torch.ones((1, width*height), dtype=torch.float32).to(torch.device('cuda:0'))], dim=0)
    cam_coords = torch.reshape(cam_coords, (-1, height, width))
    return cam_coords


def cam2pixel(cam_coords, proj):
    d, height, width = cam_coords.shape
    cam_coords = torch.reshape(cam_coords, (d, -1))
    # Project the camera coordinates into image plane
    cam_coords = cam_coords.float()
    unnormalized_pixel_coords = proj@cam_coords
    unnormalized_pixel_coords = torch.reshape(unnormalized_pixel_coords, (-1, height, width))
    x_u = unnormalized_pixel_coords[0, :, :]
    y_u = unnormalized_pixel_coords[1, :, :]
    z_u = unnormalized_pixel_coords[2, :, :]
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = torch.stack([x_n, y_n], dim=0)
    pixel_coords = torch.transpose(pixel_coords, 0, 2)
    pixel_coords = torch.transpose(pixel_coords, 0, 1)
    return pixel_coords


def bilinear_sampler(img, coords):
    """Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    Args:
        img: source image to be sampled from [batch, channels, height_s, width_s]
        coords: coordinates of source pixels to sample from [batch, 2, height_t,
            width_t]. height_t/width_t correspond to the dimensions of the output
            image (don't need to be the same as height_s/width_s). The two channels
            correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, channels, height_t, width_t]
    """
    def _repeat(x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 0, 1)
        rep = rep.long()
        x = x.view(-1, 1) @ rep
        return x.view(-1)

    height_t, width_t, _ = coords.size()
    height_s, width_s, c = img.size()
    outsize = list(coords.size())
    outsize[2] = img.size()[2]

    # Clip coordinates to image bounds
    x, y = torch.split(coords, 1, dim=2)

    # Compute indices of source points
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # Clip indices to image bounds
    x0_safe = torch.clamp(x0, 0, width_s - 1)
    y0_safe = torch.clamp(y0, 0, height_s - 1)
    x1_safe = torch.clamp(x1, 0, width_s - 1)
    y1_safe = torch.clamp(y1, 0, height_s - 1)

    # Compute weights for sampling points
    wt_x0 = x1_safe - x
    wt_x1 = x - x0_safe
    wt_y0 = y1_safe- y
    wt_y1 = y - y0_safe

    # Compute flat indices into image arrays
    dim2 = width_s
    dim1 = width_s * height_s
    base = torch.reshape(_repeat(torch.arange(1).long() * dim1, height_t * width_t), (outsize[0], outsize[1], 1)).to(torch.device('cuda:0'))
    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = x0_safe + base_y0
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    # Gather pixels from image tensors
    # imgs_flat = img.permute(0, 2, 3, 1).contiguous().view(-1, img.size(1))
    imgs_flat = torch.reshape(img, (-1, c))
    im00 = torch.reshape(imgs_flat[idx00, :], outsize)
    im01 = torch.reshape(imgs_flat[idx01, :], outsize)
    im10 = torch.reshape(imgs_flat[idx10, :], outsize)
    im11 = torch.reshape(imgs_flat[idx11, :], outsize)

    # Interpolate values
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    #output = torch.sum(torch.stack([w00 * im00, w01 * im01, w10 * im10, w11 * im11]))
    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11

    return output


def computevsLoss(syn_img, t_img):
    diff_img = syn_img - t_img
    #print(diff_img)
    loss = torch.norm(diff_img)
    #loss = np.array(loss)
    return loss
    print(loss)


class loss_depth(nn.Module):
    def __init__(self):
      super(loss_depth, self).__init__()

    def forward(self, rot, trans, depth_imgs, rgb_imgs, K, num):
        loss_dict = {}
        a = 0
        for i in range(int(num / 2)):
            # put the camera inner params into GPU
            K = K.to(torch.device('cuda:0')).to(torch.float32)
            # source image
            s_img = rgb_imgs[2 * i + 1].transpose(0, 1).transpose(1,2)
            # ground-truth target image
            t_img = rgb_imgs[2 * i].transpose(0, 1).transpose(1,2)
            # ground-truth target depth
            t_depth = depth_imgs[2 * i]
            # Relative rotation and translation
            R_re = rot[2 * i + 1] @ torch.inverse(rot[2 * i])
            # print(rot[2 * i + 1])
            # print(rot[2 * i])
            trans_i_t = torch.unsqueeze(trans[2 * i], dim=0).transpose(0,1)
            trans_j_t = torch.unsqueeze(trans[2 * i + 1], dim=0).transpose(0,1)
            # print(trans_i_t)
            # print(trans_j_t)
            t_re = trans_j_t - torch.matmul(R_re, trans_i_t)
            pose = torch.cat((R_re, t_re), dim=1)
            proj = torch.matmul(K, pose)
            pixel = meshgrid(480, 640).to(torch.device('cuda:0'))
            cam_coords = pixel2cam(t_depth, pixel, K)
            pixels = cam2pixel(cam_coords, proj)
            # synthetic target image
            syn_t_img = bilinear_sampler(s_img, pixels)
            mask = (0.1 < t_depth).int()
            mask = torch.reshape(mask, (t_depth.shape[0], t_depth.shape[1], 1))
            mask1 = (t_depth < 1.4).int()
            mask1 = torch.reshape(mask1, (t_depth.shape[0], t_depth.shape[1], 1))
            # masked the pixels outside image
            mask_x_low = (pixels[:, :, 0] > 0).int()
            mask_x_up = (pixels[:, :, 0] < syn_t_img.size()[1]).int()
            mask_y_low = (pixels[:, :, 1] > 0).int()
            mask_y_up = (pixels[:, :, 1] < syn_t_img.size()[0]).int()
            mask_edge = mask_x_up * mask_x_low * mask_y_low * mask_y_up
            mask_edge = torch.reshape(mask_edge, (syn_t_img.shape[0], syn_t_img.shape[1], 1))
            syn_t_img = syn_t_img * mask_edge * mask1 * mask
            # 统计mask中非零元素个数
            num_not_zero = torch.sum(mask_edge * mask1 * mask)/3
            t_img = t_img * mask_edge * mask1 * mask
            # img = syn_t_img.detach().cpu().numpy()
            # img1 = np.zeros((480, 640, 3))
            #print(batch['roi_img'][0])
            # for i in range(480):
            #     for j in range(640):
            #         img1[i][j][0] = img[0][i][j]
            #         img1[i][j][1] = img[1][i][j]
            #         img1[i][j][2] = img[2][i][j]
            # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            # cv2.imwrite("img00.png", img)
            loss = computevsLoss(syn_t_img, t_img)/num_not_zero
            #print(loss)
            loss_dict[str(a)] = loss
            a += 1
        #exit()
        return sum(loss_dict.values())


if __name__ == "__main__":
    import torch
    t_color_path = r'datasets/BOP_DATASETS/lm/test/000001/rgb/000004.png'
    s_color_path = r'datasets/BOP_DATASETS/lm/test/000001/rgb/000009.png'
    depth_path = r'datasets/BOP_DATASETS/lm/test/000001/depth/000004.png'
    K = torch.tensor([[572.4114, 0.0, 325.2611],
                      [0.0, 573.57043, 242.04899],
                      [0.0, 0.0, 1.0]])
    Roi = torch.tensor([[0.108976, 0.99175203, 0.0674766],
                        [0.66973603, -0.0230918, -0.74224001],
                        [-0.73456001, 0.12607799, -0.66672802]])
    Roj = torch.tensor([[0.16422901, 0.98613298, -0.0238958],
                        [0.78948301, -0.145926, -0.59617198],
                        [-0.59139198, 0.0790434, -0.80250102]])
    toi = torch.tensor([[-103.59141046], [-49.80357101], [1025.07614997]])
    toj = torch.tensor([[-71.54985682], [-36.76336895], [1015.5891147]])
    R = Roj @ torch.inverse(Roi)
    t = toj - torch.matmul(R, toi)
    pose = torch.cat((R, t), dim=1)
    proj = torch.matmul(K, pose)
    pixel = meshgrid(480, 640)
    depth = cv2.imread(depth_path, -1).astype(np.float_)
    depth = torch.from_numpy(depth)
    cam_coords = pixel2cam(depth, pixel, K)
    pixels = cam2pixel(cam_coords, proj)
    s_img = cv2.imread(s_color_path)
    s_img = torch.from_numpy(s_img)
    syn_img = bilinear_sampler(s_img, pixels)
    mask = (100 < depth).int()
    mask = torch.reshape(mask, (depth.shape[0], depth.shape[1], 1))
    mask1 = (depth < 1400).int()
    mask1 = torch.reshape(mask1, (depth.shape[0], depth.shape[1], 1))
    # masked the zero pixels in the syn_img
    syn_img_norm = torch.norm(syn_img, dim = 2)
    mask_zero = (syn_img_norm > 0).int()
    mask_zero = torch.reshape(mask_zero, (syn_img_norm.shape[0], syn_img_norm.shape[1], 1))
    # masked the pixels outside image
    mask_x_low = (pixels[:, :, 0] > 0).int()
    mask_x_up = (pixels[:, :, 0] < syn_img.size()[1]).int()
    mask_y_low = (pixels[:, :, 1] > 0).int()
    mask_y_up = (pixels[:, :, 1] < syn_img.size()[0]).int()
    mask_edge = mask_x_up * mask_x_low * mask_y_low * mask_y_up
    mask_edge = torch.reshape(mask_edge, (syn_img_norm.shape[0], syn_img_norm.shape[1], 1))
    syn_img = syn_img * mask_edge * mask1
    t_img = torch.from_numpy(cv2.imread(t_color_path))
    t_img = t_img * mask_edge * mask1
    computevsLoss(syn_img, t_img)
    mask = np.array(mask)
    mask_edge = np.array(mask_edge)
    mask_x_low = np.array(mask_x_low)
    syn_img = np.array(syn_img)
    t_img = np.array(t_img)
    cv2.imshow('masked', mask_edge.astype(np.float_))
    cv2.imshow('syn_img', syn_img.astype('uint8'))
    cv2.imshow('t_img', t_img.astype('uint8'))
    cv2.waitKey(0)