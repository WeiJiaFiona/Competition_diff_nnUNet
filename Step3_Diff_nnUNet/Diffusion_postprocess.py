#!/usr/bin/env python
# coding: utf-8

# # Diff-nnUNet预测结果后处理
# ## 原理：
# 引入SimpleITK库中的ConnectedComponentImageFilter函数和LabelShapeStatisticsImageFilter函数，保留最大连通域，并求出输入mask的连通域个数。
# ## 处理步骤：
# Diff-nnUNet的预测结果（文件夹“predict”）划分了肝脏（数值1）和肿瘤（数值2），对结果进行如下处理：
# - 将预测结果中的数值1置零，数值2置一。即删去肝脏标签，只保留肿瘤标签。（文件夹“predict_processed”）
# - 保留肿瘤标签的最大连通域，即为所需最大肿瘤。（文件夹“predict_processed_connected_domain”）

# 导入库

# In[1]:


import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
import glob
import nibabel as nib


# 保留最大连通域的函数

# In[2]:


def connected_domain(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:最大连通域  res_itk
    :return:连通域个数  num_connected_label
    :return:每个连通域的体积  each_area
    """
    
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(itk_mask)
 
    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)
 
    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数
 
    area_max_label = 0  # 最大的连通域的label
    area_max = 0
    
    each_area = [lss_filter.GetNumberOfPixels(i) for i in range(1, num_connected_label + 1)]
    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)

          # 根据label获取连通域面积
        if area > area_max:
            area_max_label = i
            area_max = area
    
    np_output_mask = sitk.GetArrayFromImage(output_mask)
  
    res_mask = np.zeros_like(np_output_mask)
    res_mask[np_output_mask == area_max_label] = 1

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())

    return  num_connected_label, res_itk


# 批量保留最大连通域的函数

# In[3]:


def save_connected_domain(pre_path, save_path):

    name_list = os.listdir(pre_path)
    none_single_connected={}
    
    for name in name_list:
        print(name)

        prediction_path =   os.path.join(pre_path , name)
        num_connected_label, save_biggest_img= connected_domain(sitk.Cast(sitk.ReadImage(prediction_path), sitk.sitkInt32))
        print("     connected_domain get")

        if num_connected_label != 1:
            none_single_connected[name[0:3]] = num_connected_label
    
        save_img_path = os.path.join(save_path, name)
        sitk.WriteImage(save_biggest_img, save_img_path)
        print("     connected_domain saved")
        print()
    
    print("none_single_connected:", none_single_connected)


# 创建文件夹

# In[4]:


def ensure_folder_exists(folder_path):
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path):
        # 如果不存在，则创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹{folder_path}已创建。")
    else:
        print(f"文件夹{folder_path}已存在。")


# 删去肝脏标签

# In[5]:


def process_and_save_nii_files(source_folder, target_folder):
    nii_files = glob.glob(os.path.join(source_folder, "*.nii.gz"))
    
    for file_path in nii_files:
        nii_data = nib.load(file_path)
        data = nii_data.get_fdata()
        
        # 将数值1置零，将数值2置一
        data[data == 1] = 0
        data[data == 2] = 1
        
        # 创建新的NIfTI图像对象
        new_nii = nib.Nifti1Image(data, affine=nii_data.affine, header=nii_data.header)
        
        # 构建目标文件路径
        target_file_path = os.path.join(target_folder, os.path.basename(file_path))
        
        # 保存处理后的文件
        nib.save(new_nii, target_file_path)
        # print(f"Processed file saved to {target_file_path}")

ori_pre_path = "/root/wjz_diff/50test/predict"
pre_path = "/root/wjz_diff/50test/predict_processed"
ensure_folder_exists(pre_path)
process_and_save_nii_files(ori_pre_path, pre_path)


# 保留最大连通域

# In[11]:


pre_path = "/root/wjz_diff/50test/predict_processed"
save_path = "/root/wjz_diff/50test/predict_processed_connected_domain"
ensure_folder_exists(save_path)
save_connected_domain(pre_path, save_path)

