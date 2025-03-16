### Step1 使用半监督训练的nnUNet补全pseudo label
* nnUNet的使用方式请见官方文档how_to_use_nnUNet.md，此处仅说明半监督补全肿瘤标签的流程
* 数据准备：将赛方提供的仅有最大肿瘤标注的CT影像准备成nnUNet要求的数据集格式
* 大小肿瘤数据集划分：
```
cd Step1_nnUNet_pseudo_label\Split_dataset
```
运行calculate_volumn.ipynb文件，根据肿瘤体积中位数划分大小肿瘤数据集

* nnUNet代码更改：`\nnUNet\nnunetv2\training\nnUNetTrainer\nnUNetTrainer.py` 文件第695-696行注释掉镜像增广，防止在非肝脏区域预测肿瘤
```
#if mirror_axes is not None and len(mirror_axes) > 0:
   #tr_transforms.append(MirrorTransform(mirror_axes))
```
* 使用小肿瘤数据集DS训练nnUNet，使之具有预测较小肿瘤的能力：
基本指令
```
nnUNetv2_plan_and_preprocess -d 202 --verify_dataset_integrity
nnUNetv2_train 202 3d_fullres 0 all 
```
* 半监督预训练权重下载地址：https://pan.baidu.com/s/1gXjOJtfCVSf2qZ8frir0Nw?pwd=cuc3 提取码: cuc3
下载以后放至`Step1_nnUNet_pseudo_label\ckpt\Dataset202_SmallTumor`文件夹下
* 使用预训练权重预测出原先未标注小肿瘤的pseudo label
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 202 -c 3d_fullres
```
* 最后与赛方提供的最大肿瘤标注取并集，获得全标签mask，见`Full_label_mask`文件夹下结果

