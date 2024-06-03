python test_me.py \
    --inet dsrnet_s \
    --model dsrnet_model_sirs \
    --dataset sirs_dataset  \
    --name dsrnet_s_test \
    --hyper \
    --if_align \
    --resume \
    --weight_path "./weights/dsrnet_s_epoch14.pt" \
    --base_dir "comp/"

