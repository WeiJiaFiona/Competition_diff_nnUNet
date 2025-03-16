import os
import nibabel as nib
import numpy as np

# 定义文件路径
input_dir = r'D:\projects\DiffTumor\fifty_test\labelsTr'
output_dir = r'D:\projects\DiffTumor\STEP3.SegmentationModel\organ_pseudo_swin_new\liver'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取所有文件
files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]

for file in files:
    # 读取 NIfTI 文件
    file_path = os.path.join(input_dir, file)
    nifti_img = nib.load(file_path)
    nifti_data = nifti_img.get_fdata()

    # 提取强度大于等于1的掩码
    liver_mask = (nifti_data >= 1).astype(np.uint8)

    # 获取文件名中的数字部分
    num_part = ''.join(filter(str.isdigit, file))

    output_filename = f'{num_part}.nii.gz'
    output_path = os.path.join(output_dir, output_filename)

    # 创建新的 NIfTI 图像并保存
    liver_nifti_img = nib.Nifti1Image(liver_mask, nifti_img.affine)
    nib.save(liver_nifti_img, output_path)

    print(f'Saved: {output_path}')
