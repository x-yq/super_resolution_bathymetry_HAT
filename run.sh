python3 train.py --img_output_folder "Bathymetry_predict"\
 --best_model_path "weights/bathymetry_ft_combined.pth"\
  --pretrained_weights_path "weights/best_psnr_hat_retrained_combined.pth"\
  --target_w 30 \
  --target_h 30 \
  --upscale 2 \
  --batch_size 8 \
  --interpolate_mode 'nearest' \
  --task 'ft_combined_imagery_trained_model_norm' \
  --epochs 5 \
  --train_combine_dataset 1 \
  --fine_tune 0

# python3 train.py --img_output_folder "denormalized_x4"\
#  --best_model_path "weights/best_loss_hat.pth"\
#   --pretrained_weights_path "weights/pretrained_HAT/HAT-L_SRx4_ImageNet-pretrain.pth"\
#   --target_w 72 \
#   --target_h 72 \
#   --upscale 4 \
#   --interpolate_mode 'nearest'