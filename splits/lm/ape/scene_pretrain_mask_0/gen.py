import os
rgb_path = r'/data3/tantao/my_projects/TexPose/TexPose-main/dataset/lm/lm_real_mask/000001/rgb'
rgb_files = os.listdir(rgb_path)
rgb_files.sort(key=lambda x : int(x.split('.')[0]))

with open('train.txt', 'w') as f:
    for i in range(185):
        f.write('ape_mask lm_real_mask/000001 '+'{:06d}\n'.format(int(rgb_files[i].split('.')[0])))