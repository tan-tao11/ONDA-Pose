import os
import cv2
import numpy as np
import argparse

def process_images(mask_path, rgb_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    mask_files = os.listdir(mask_path)
    for mask_file in mask_files:
        if not mask_file.endswith(('.png', '.jpg', '.bmp', '.tiff')):
            continue

        mask_full_path = os.path.join(mask_path, mask_file)
        rgb_filename = mask_file.split('_')[0] + '.png'
        rgb_full_path = os.path.join(rgb_path, rgb_filename)
        save_full_path = os.path.join(save_path, rgb_filename)

        mask = cv2.imread(mask_full_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[Warning] Failed to read mask: {mask_full_path}")
            continue
        mask = mask.astype(np.float32) / 255.0

        rgb = cv2.imread(rgb_full_path)
        if rgb is None:
            print(f"[Warning] Failed to read RGB: {rgb_full_path}")
            continue

        rgb_mask = (rgb.transpose(2, 0, 1) * mask).transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(save_full_path, rgb_mask)

    print(f"[Done] Processed {len(mask_files)} mask files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply binary masks to RGB images and save results.")
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the mask images')
    parser.add_argument('--rgb_path', type=str, required=True, help='Path to the RGB images')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save masked RGB images')

    args = parser.parse_args()
    process_images(args.mask_path, args.rgb_path, args.save_path)