poetry run python test_me.py \
    --inet dsrnet_s \
    --model dsrnet_model_sirs \
    --dataset sirs_dataset  \
    --name dsrnet_s_test \
    --hyper \
    --if_align \
    --resume \
    --weight_path "./weights/dsrnet_s_epoch14.pt" \
    --base_dir "/home/yamagami/works/ishida/okamoto_anormally_detection/datasets_okamoto/label_studio/VID_20240124_112513/bad/"


poetry run python -m src.classification.classification_patches_with_refrem \
    --inet dsrnet_s \
    --model dsrnet_model_sirs \
    --dataset sirs_dataset  \
    --name dsrnet_s_test \
    --hyper \
    --if_align \
    --resume \
    --weight_path "./DSRNet/weights/dsrnet_s_epoch14.pt" 

