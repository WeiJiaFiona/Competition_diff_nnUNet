import os
import subprocess

# Set environment variables
# datapath = "D:/projects/DiffTumor/Task03_Liver/"

datapath = "D:/projects/DiffTumor/test/"
organ = "liver"
fold = 1
datafold_dir = f"cross_eval/{organ}_aug_data_fold/"

# U-Net configuration
model = "nnunet"
logdir = f"D:/projects/DiffTumor/STEP3.SegmentationModel/{organ}/{organ}.fold{fold}.nnunet"
save_dir = f"D:/projects/DiffTumor/STEP3.SegmentationModel/out/{organ}/{organ}.fold{fold}.nnunet"

# Run the validation.py command with specified arguments
subprocess.run([
    "python", "-W", "ignore", "validation.py",
    "--model", model,
    "--data_root", datapath,
    "--datafold_dir", datafold_dir,
    "--tumor_type", "tumor",
    "--organ_type", organ,
    "--fold", str(fold),
    "--log_dir", logdir,
    "--save_dir", save_dir
])


