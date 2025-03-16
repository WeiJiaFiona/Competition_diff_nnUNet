#!/bin/bash

vqgan_ckpt="D:/projects/DiffTumor/STEP2.DiffusionModel/pretrained_models/AutoencoderModel.ckpt"
fold=0
datapath="D:/projects/DiffTumor/Task03_Liver/"
tumorlabel="D:/projects/DiffTumor/preprocessed_labels/"

# 输出当前工作目录，确认脚本在正确的路径下执行
pwd

# 执行 Python 脚本并传递参数
python D:/projects/DiffTumor/STEP2.DiffusionModel/train.py \
dataset.name="liver_tumor_train" \
dataset.fold="$fold" \
dataset.data_root_path="$datapath" \
dataset.label_root_path="$tumorlabel" \
dataset.dataset_list="['liver_tumor_data_early_fold']" \
dataset.uniform_sample="False" \
model.results_folder_postfix="liver_early_tumor_fold${fold}" \
model.vqgan_ckpt="$vqgan_ckpt"


