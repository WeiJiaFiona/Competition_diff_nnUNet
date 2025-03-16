## Diffusion-enhanced-nnUNet

### 0. Installation

Unzip our code package into the project folder. We recommend that you extract it to ```D:/projects/DiffTumor/```  for easy operation.

The code is tested on `python 3.8, Pytorch 1.12`.

**Set up environment:**

```
conda create -n difftumor python=3.8
source activate difftumor # or conda activate difftumor
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## 1. Train Autoencoder Model

You can train Autoencoder Model on AbdomenAtlas 1.0 dataset by your own. The release of AbdomenAtlas 1.0 can be found at https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas_1.0_Mini.

```
cd STEP1.AutoencoderModel
datapath=<your-datapath> (e.g., /data/bdomenAtlasMini1.0/)
gpu_num=1
cache_rate=0.05
batch_size=4
dataset_list="AbdomenAtlas1.0Mini"
python train.py dataset.data_root_path=$datapath dataset.dataset_list=$dataset_list dataset.cache_rate=$cache_rate dataset.batch_size=$batch_size model.gpus=$gpu_num
```

We offer the pre-trained checkpoint of Autoencoder Model, which was trained on AbdomenAtlas 1.1 dataset (see details in [SuPreM](https://github.com/MrGiovanni/SuPreM)). This checkpoint can be directly used for STEP2 if you do not want to re-train the Autoencoder Model. Simply download it to `STEP2.DiffusionModel/pretrained_models/AutoencoderModel.ckpt`

## 2. Train Diffusion Model

To train the diffusion model for liver tumor synthesis, follow the instructions below.

### Dataset Preparation

We have stored the numbers of the segmented small tumor dataset in the `/diffusion_data/data_number.txt` file. Please place the CT image files in the `img` folder and the tumor annotation files in the `GT` folder according to the numbers.

### Command

Use the following command to start the training process:

```bash
python D:/projects/DiffTumor/STEP2.DiffusionModel/train.py \
    dataset.name="liver_tumor_train" \
    dataset.fold=0 \
    dataset.data_root_path="D:/projects/DiffTumor/diffusion_data/img/" \
    dataset.label_root_path="D:/projects/DiffTumor/diffusion_data/GT/" \
    dataset.dataset_list="['liver_tumor_data_early_fold']" \
    dataset.uniform_sample="False" \
    model.results_folder_postfix="liver_early_tumor_fold0" \
    model.vqgan_ckpt="D:/projects/DiffTumor/STEP2.DiffusionModel/pretrained_models/AutoencoderModel.ckpt"
```

### Environment Variables

Ensure the following environment variables are set:

```bash
export PYTHONUNBUFFERED=1
export vqgan_ckpt="D:/projects/DiffTumor/STEP2.DiffusionModel/pretrained_models/AutoencoderModel.ckpt"
export fold=0
export tumorlabel="D:/projects/DiffTumor/preprocessed_labels/"
```

### Training Script

The training script is located at `D:/projects/DiffTumor/STEP2.DiffusionModel/train.py`. Ensure all the paths and parameters are correctly set as shown in the command above.

### Result

``` /STEP2.DiffusionModel/checkpoints/ddpm/liver_tumor_train/liver_early_tumor_fold0/our_best_model.pt```

## 3. Train Segmentation Model

We provide our full-label dataset and the segmented small tumor dataset in the liver region. This part of the dataset can be found in the [Baidu Netdisk](https://pan.baidu.com/s/18RxT0ncf44yG6xRHsLU4LA?pwd=3bfq) link. After downloading, extract it to the project root directory. The file name is ```test.zip``` and ```full_label_training.zip```. 

We also provide the model weights needed to generate tumors. This part of the weight is derived from the results obtained in the previous part. Unzip it to ```/STEP3.SegmentationModel/TumorGeneration/```. The file name is ```model_weight.zip```.

Navigate to the `STEP3.SegmentationModel` directory and run `run_step3.py` directly.

The best model weights of the nnUNet segmentation model we trained are stored in: `STEP3.SegmentationModel/liver/liver.fold2.nnunet/model_epoch395.pt`.

## 4. Evaluation

Navigate to the `STEP3.SegmentationModel` directory and run `run_step4.py` directly.
## 5. Postprocess
run `python Diffusion_postprocess.py` to preserve the largest connected component

## Acknowledgement
This code is credited to **Zongwei Zhou** [Generalizable Tumor Synthesis](https://github.com/MrGiovanni/DiffTumor). We have made some modifications to better achieve our goals.

