**This work focuses on classifying breast cancer whole slide images (WSIs) into two primary categories: HER2-positive and HER2-negative. In addition, it further addresses the challenging cases with equivocal HER2 2+ scores, by distinguishing them as either FISH-positive or FISH-negative.**

## 📚 Acknowledgements
This pipeline is inspired by the excellent work in the [AIRMEC/im4MEC repository](https://github.com/AIRMEC/im4MEC/tree/main).

For MoCoV3-based self-supervised learning, we use the implementation from [Facebook Research's MoCoV3](https://github.com/facebookresearch/moco-v3).


## 🔄 Pipeline Steps for HER2 Classification from WSIs

## ✅ 1. Environment Setup
To begin, create and activate the virtual environment using the provided requirements.yaml:

conda env create --prefix ./.conda -f requirements.yml

conda activate ./.conda

## 2. 🧩 Patch Extraction from Whole Slide Images (WSIs)
Extract patches of size 360×360 from each WSI listed in her2wsifilespath.txt. This generates 2048 patches per slide, resized to 224×224:

cat her2wsifilespath.txt | xargs -I WSIFILE echo python sample_tiles.py --input_slide=WSIFILE --output_dir=wsi_patches --tile_size=360 --n=2048 --out_size=224 | bash

## 3. 🧠 Self-Supervised Pretraining (MoCoV3 with ResNet-50)
Train a ResNet-50 encoder using MoCoV3-based contrastive learning on the extracted patches of the wsi dataset:

python moco-self/main_moco.py wsi_patches --moco-m-cos --crop-min=.2 --dist-url --dist-url 'tcp://127.0.0.1:10004' --multiprocessing-distributed --world-size 1 --rank 0

## 4. 🔍 Patch Feature Extraction Using Pretrained ResNet-50
Extract features for all tiles listed in svsfiles.txt using the pretrained encoder checkpoint:

cat her2wsifilespath.txt | xargs -I WSIFILE echo python preprocess.py --input_slide=WSIFILE --output_dir patch_features --tile_size 360 --out_size 224 --batch_size 4096 --checkpoint /path_to_resnet50_checkpoint/checkpoint_0299.pth.tar --backbone resnet50 --workers 4 | bash

## 5. 🧪 Train Attention-Based MIL Classifier
Train a self-attention-based multiple instance learning (MIL) model using the extracted features:

python train.py --manifest her2labels.csv --data_dir patch_features --input_feature_size 2048 --fold 0

## 6. 🌡️ Visualize Attention Heatmaps
Generate attention heatmaps for a given WSI using the trained attention model:

python attention.py --input_slide /path_to_slde/slide.svs --output_dir heatmaps --manifest her2labels.csv --encoder_checkpoint /path_to_resnet50_checkpoint/checkpoint_0299.pth.tar --encoder_backbone resnet50 --attn_checkpoint /path_to_attention_checkpoint/30_checkpoint.pt --attn_model_size small --input_feature_size 2048 --tile_size 360 --out_size 224 --overlap_factor 1 --display_level 2
