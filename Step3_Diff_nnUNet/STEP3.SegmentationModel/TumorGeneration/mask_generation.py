import os
import random
import nibabel as nib
import numpy as np
from collections import Counter

from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import zoom
from scipy.ndimage import zoom, rotate

def generate_mask(x, y, z, args):
    x=4*x
    y = x
    z = x
    data_path = args.data_root

    # 遍历data_path找到所有的liver_label文件夹
    liver_label_dirs = []
    for root, dirs, files in os.walk(data_path):
        for dir_name in dirs:
            if dir_name == 'liver_label':
                liver_label_dirs.append(os.path.join(root, dir_name))

    # 找到所有的nifti文件
    nifti_files = []
    for liver_label_dir in liver_label_dirs:
        for file_name in os.listdir(liver_label_dir):
            if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                nifti_files.append(os.path.join(liver_label_dir, file_name))

    if not nifti_files:
        print("没有找到任何nifti文件")
        return

    # 随机选择一个nifti文件
    selected_file = random.choice(nifti_files)

    # 读取nifti文件并找到肿瘤mask（值为2）
    nii = nib.load(selected_file)
    data = nii.get_fdata()

    # 获取mask值为2的部分
    tumor_mask = (data > 1.5)

    # 找到肿瘤的bounding box
    coords = np.where(tumor_mask)
    min_coords = np.min(coords, axis=1)
    max_coords = np.max(coords, axis=1)

    # 计算bounding box的尺寸
    bbox_dims = max_coords - min_coords

    # 找到最长的一边
    max_dim = np.max(bbox_dims)

    # 计算bounding box的中心点
    center_coords = (min_coords + max_coords) // 2

    # 计算crop区域的起始和结束坐标
    crop_start = np.maximum(center_coords - max_dim // 2, 0)
    crop_end = np.minimum(crop_start + max_dim, data.shape)

    # Crop得到立方体的肿瘤掩码区域
    cropped_tumor_mask = tumor_mask[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]

    # 计算体素个数
    voxel_count = np.sum(cropped_tumor_mask)

    print("选中的nifti文件是：", selected_file)
    print("肿瘤mask形状：", cropped_tumor_mask.shape)
    print("肿瘤mask的体素个数：", voxel_count)

    # 缩放到x边长大小
    current_shape = cropped_tumor_mask.shape
    zoom_factors = (x / current_shape[0], x / current_shape[1], x / current_shape[2])
    resized_tumor_mask = zoom(cropped_tumor_mask, zoom_factors, order=1)  # 使用线性插值进行缩放

    # Flatten the data to 1D array for counting
    data_flat = resized_tumor_mask.flatten()

    # Count the occurrences of each value
    value_counts = Counter(data_flat)

    # Convert to a dictionary for easy viewing
    value_counts_dict = dict(value_counts)

    # 缩放到x边长大小
    current_shape = cropped_tumor_mask.shape
    zoom_factors = (x / current_shape[0], x / current_shape[1], x / current_shape[2])
    resized_tumor_mask = zoom(cropped_tumor_mask.astype(float), zoom_factors, order=0)  # 使用最近邻插值进行缩放

    # 随机角度旋转
    angle_x = random.uniform(0, 360)
    angle_y = random.uniform(0, 360)
    angle_z = random.uniform(0, 360)

    rotated_tumor_mask = rotate(resized_tumor_mask, angle_x, axes=(1, 2), reshape=False, order=0, mode='constant', cval=0)
    rotated_tumor_mask = rotate(rotated_tumor_mask, angle_y, axes=(0, 2), reshape=False, order=0, mode='constant', cval=0)
    rotated_tumor_mask = rotate(rotated_tumor_mask, angle_z, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0)

    return rotated_tumor_mask.astype(int)

class Args:
    data_root = "D:/projects/DiffTumor/processed_liver_CT/"

if __name__ == '__main__':
    args = Args()
    mask = generate_mask(8, 8, 8, args)