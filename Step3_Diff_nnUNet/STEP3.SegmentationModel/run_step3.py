import os
import subprocess
import random
import platform
import torch.multiprocessing as mp

# Set environment variables
healthy_datapath = "D:/projects/DiffTumor/processed_liver_CT/"
datapath = "D:/projects/DiffTumor/processed_liver_CT/"

cache_rate = 1.0
batch_size = 2
val_every = 1000
workers = 0
organ = "liver"
fold = 0

# U-Net configuration
backbone = "nnunet"
logdir = f"runs/{organ}.fold{fold}.{backbone}"
datafold_dir = f"cross_eval/{organ}_aug_data_fold/"
dist = random.randint(10000, 109999)


# Run the main.py command with specified arguments
subprocess.run([
    "python", "-W", "ignore", "main.py",
    "--model_name", backbone,
    "--cache_rate", str(cache_rate),
    "--dist-url", f"tcp://127.0.0.1:{dist}",
    "--workers", str(workers),
    "--max_epochs", "2000",
    "--val_every", str(val_every),
    "--batch_size", str(batch_size),
    "--save_checkpoint",
    "--noamp",
    "--organ_type", organ,
    "--organ_model", organ,
    "--tumor_type", "tumor",
    "--fold", str(fold),
    "--ddim_ts", "50",
    "--logdir", logdir,
    "--healthy_data_root", healthy_datapath,
    "--data_root", datapath,
    "--datafold_dir", datafold_dir
])
