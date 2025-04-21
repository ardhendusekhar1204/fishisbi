for extraction of tiles from whole slide images :
cat her2wsifilespath.txt | xargs -I WSIFILE echo python sample_tiles.py --input_slide=WSIFILE --output_dir=wsi_patches --tile_size=360 --n=2048 --out_size=224 | bash


initial self-supervision resnet50
python moco-self/main_moco.py wsi_patches --moco-m-cos --crop-min=.2 --dist-url --dist-url 'tcp://127.0.0.1:10004' --multiprocessing-distributed --world-size 1 --rank 0



cat svsfiles.txt | xargs -I WSIFILE echo python preprocess.py --input_slide=WSIFILE --output_dir patch_features --tile_size 360 --out_size 224 --batch_size 4096 --checkpoint /path_to_resnet50_checkpoint/checkpoint_0299.pth.tar --backbone resnet50 --workers 4 | bash



to train the self attention model
python train.py --manifest her2labels.csv --data_dir patch_features --input_feature_size 2048 --fold 0


to plot the attention heatmaps
python attention.py --input_slide /path_to_slde/slide.svs --output_dir heatmaps --manifest her2labels.csv --encoder_checkpoint /path_to_resnet50_checkpoint/checkpoint_0299.pth.tar --encoder_backbone resnet50 --attn_checkpoint /path_to_attention_checkpoint/30_checkpoint.pt --attn_model_size small --input_feature_size 2048 --tile_size 360 --out_size 224 --overlap_factor 1 --display_level 2
