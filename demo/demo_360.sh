# scripts for nerf_360 data

# train the model
python train.py --log_dir record/garden --dataset_name 360 --scene garden
# visualize the model
python video.py --log_dir record/garden --dataset_name llff --scene garden --model_weight_path record/garden/model_10000.pt
# test the model on the evaluation view directions
python test.py --log_dir record/garden --dataset_name llff --scene garden --model_weight_path record/garden/model_10000.pt