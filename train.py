from sys import stderr
import time
import torch.optim as optim
import torch.nn as nn
import tifffile
from models.hat_arch import HAT
from torchvision.transforms import transforms as T
from datasets.dataset_loader import dataset_loader
import torch
import numpy as np
import os
from PIL import Image
import torch
from PSNR_SSIM import PSNR_SSIM
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
from setup import parser
import torch.nn.functional as F
import datetime

def log(log_file, msg):
    """
    Logs a message to the log file and stderr.

    Args:
        log_file: File object for the log file.
        msg (str): Message to log.
    """
    print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
    log_file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)
    log_file.flush()
    os.fsync(log_file)


class ResizeWithCV2(object):
    def __init__(self, target_size, interpolate_mode):
        self.target_size = target_size
        self.mode = interpolate_mode

    def __call__(self, img):
        #resized_img = cv2.resize(np.array(img), self.target_size)
        resized_img = resize(np.array(img), self.target_size,order=1,mode='reflect',anti_aliasing=False)

        return resized_img

def unfreeze_selected_layers(log_file, model, num_layers_to_unfreeze=1):

    # Initially, freeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze the last few RHAG layers
    num_total_layers = len(model.layers)
    for i in range(num_total_layers - num_layers_to_unfreeze, num_total_layers):
        for name, param in model.layers[i].named_parameters():
            log(log_file, 'unfreeze: '+ name)
            param.requires_grad = True

    # Unfreeze the final normalization layer
    for name, param in model.norm.named_parameters():
        log(log_file, 'unfreeze norm layer: '+name)
        param.requires_grad = True

    # Unfreeze convolution layers for image reconstruction
    if hasattr(model, 'conv_before_upsample'):
        for name, param in model.conv_before_upsample.named_parameters():
            log(log_file, 'unfreeze conv layer: '+name)
            param.requires_grad = True

    if hasattr(model, 'upsample'):
        for name, param in model.upsample.named_parameters():
            log(log_file, 'unfreeze upscale layer: '+name)
            param.requires_grad = True

    if hasattr(model, 'conv_last'):
        for name, param in model.conv_last.named_parameters():
            log(log_file, 'unfreeze conv last layer: '+name)
            param.requires_grad = True

    return model

def train(log_file, args, model, dataloader, criterion, optimizer, device, num_epochs=15, validation_loader=None, save_path=None):
    best_psnr = 0.0
    best_epoch = 0
    metric = PSNR_SSIM()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_ssim = 0.0
        running_psnr = 0.0
        # Training loop
        for train_inputs_paths,_, inputs, targets in dataloader:
            inputs = inputs.to(device,dtype=torch.float32)
            targets = targets.to(device,dtype=torch.float32)
            # inputs = inputs.to(dtype=torch.float32)
            # targets = targets.to(dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_psnr += metric.calculate_psnr_pt(targets, outputs, crop_border=0, test_y_channel=False, img_size=(30,30))[0]
            running_ssim += metric.calculate_ssim_pt(targets, outputs, crop_border=0, test_y_channel=False, img_size=(30,30))[0]

        # Calculate average training loss for the epoch
        train_loss = running_loss / len(dataloader)
        avg_psnr = running_psnr / len(dataloader)
        avg_ssim = running_ssim / len(dataloader)
        
        # Print training loss for the epoch
        if args.train_combine_dataset:
            log(log_file, f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Average PSNR: {avg_psnr:.4f} dB, Average SSIM: {avg_ssim:.4f}')
        else:
            if 'agia_napa' in train_inputs_paths[0]:
                location = 'agia_napa'
            else:
                location = 'puck_lagoon'
            log(log_file, f'{location}: Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Average PSNR: {avg_psnr:.4f} dB, Average SSIM: {avg_ssim:.4f}')

        # Validation loop
        if validation_loader and (epoch+1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            with torch.no_grad():
                for val_inputs_paths,_, val_inputs, val_targets in validation_loader:
                    val_inputs = val_inputs.to(device,dtype=torch.float32)
                    val_targets = val_targets.to(device,dtype=torch.float32)
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_targets).item()
                    val_psnr += metric.calculate_psnr_pt(val_targets, val_outputs, crop_border=0, test_y_channel=False, img_size=(30,30))[0]
                    val_ssim += metric.calculate_ssim_pt(val_targets, val_outputs, crop_border=0, test_y_channel=False, img_size=(30,30))[0]

            val_loss /= len(validation_loader)
            val_psnr /= len(validation_loader)
            val_ssim /= len(validation_loader)
        
            # Print training loss for the epoch
            if args.train_combine_dataset:
                log(log_file, f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}, Validation PSNR: {val_psnr:.4f} dB, Validation SSIM: {val_ssim:.4f}')
            else:
                if 'agia_napa' in val_inputs_paths[0]:
                    location = 'agia_napa'
                else:
                    location = 'puck_lagoon'
                log(log_file, f'{location}: Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}, Validation PSNR: {val_psnr:.4f} dB, Validation SSIM: {val_ssim:.4f}')

            # Save the model weights if validation loss is the lowest so far
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                if save_path:
                    log(log_file, 'saving the best model in' + save_path)
                    torch.save(model.state_dict(), save_path)
    
    log(log_file, f'Best Validation PSNR: {best_psnr} at Epoch {best_epoch+1}')

# def denormalize(img, target_size, interpolate_mode, img_path):
#     image = img.cpu().detach()
#     # image = F.interpolate(image, target_size,mode = interpolate_mode)
#     image = image.numpy().clip(0,1)
#     # if image.ndim == 3:
#     #     image = np.transpose(image, (1, 2, 0))
#     if image.ndim == 3:
#         image = np.transpose(image[[2, 1, 0], :, :], (1, 2, 0))

#      ## denormalize
#     norm_params = {
#         "s2_an": np.load("configs/norm_params/norm_param_s2_an.npy"),
#         "s2_pl": np.load("configs/norm_params/norm_param_s2_pl.npy"),
#         "spot6_an": np.load("configs/norm_params/norm_param_spot6_an.npy"),
#         "spot6_pl": np.load("configs/norm_params/norm_param_spot6_pl.npy")
#     }

#     # "De-Normalization"
#     if '410' in img_path:
#         norm_param_s2 = norm_params["s2_an"]
#         norm_param_spot6 = norm_params["spot6_an"]
#     else:
#         norm_param_s2 = norm_params["s2_pl"]
#         norm_param_spot6 = norm_params["spot6_pl"]

#     image = image * (norm_param_s2[1] - norm_param_s2[0]) + norm_param_s2[0]
#     return image


def save_image(dataset, interpolate_mode, img_path, image, label, base_name, output_root, task, bathy=True, denorm=False, stich=True, target_size=(36,36)):
    if not denorm:
        image = image.cpu().detach().permute(1,2,0).numpy().clip(0,1)
        label = label.cpu().detach().permute(1,2,0).numpy().clip(0,1)
        image = resize(image, target_size,order=1,mode='reflect',anti_aliasing=False)
        #label = resize(label, target_size,order=1,mode='reflect',anti_aliasing=False)
        image = (image * 255.0).round().astype(np.uint8)
        label = (label * 255.0).round().astype(np.uint8)
    # else:
    #     #image = denormalize(image,target_size,interpolate_mode,img_path)

    if 'agia_napa' in img_path:
        img_file = 'agia_napa'
    else:
        img_file = 'puck_lagoon'

    if bathy:
        image = np.mean(image.cpu().detach().numpy(), axis=0)
        print(image.shape)
    
    #if not stich:
    path = f'images/{output_root}/{task}/{img_file}'
    os.makedirs(path,exist_ok = True)
    #tifffile.imwrite(os.path.join(path, base_name), image)
    dataset.save_as_tiff(image, img_path, os.path.join(path, base_name))

def predict_and_save(dataset, log_file, args, model, dataloader, output_root, device, interpolate_mode, target_size):
    model.eval()
    total_psnr_an = 0.0
    total_ssim_an = 0.0
    total_psnr_pl = 0.0
    total_ssim_pl = 0.0
    num_an = 0
    num_pl = 0
    metric = PSNR_SSIM()

    with torch.no_grad():
        for img_paths, label_paths, inputs, targets in dataloader:
            inputs = inputs.to(device,dtype=torch.float32)
            targets = targets.to(device,dtype=torch.float32)
            # inputs = inputs.to(torch.float32)
            # targets = targets.to(torch.float32)
            outputs = model(inputs)

            psnr = metric.calculate_psnr_pt(targets, outputs, crop_border=0, test_y_channel=False, img_size=target_size)[0]
            ssim = metric.calculate_ssim_pt(targets, outputs, crop_border=0, test_y_channel=False, img_size=target_size)[0]

            if 'agia_napa' in img_paths[0]:
                total_psnr_an += psnr
                total_ssim_an += ssim
                num_an += 1
            else:
                total_psnr_pl += psnr
                total_ssim_pl += ssim
                num_pl += 1

            for output, target, img_path, label_path in zip(outputs, targets, img_paths, label_paths):
                base_name = os.path.basename(img_path)
                save_image(dataset, interpolate_mode,img_path, output,target,base_name,output_root, task=args.task, denorm=True, stich=False, target_size=target_size)
    
    log(log_file, 'If PSNR and SSIM are zero, the model trained on distinct dataset')

    if num_an:
        avg_psnr_an = total_psnr_an / num_an
        avg_ssim_an = total_ssim_an / num_an
        log(log_file, f'Average PSNR in agia_napa: {avg_psnr_an:.4f} dB \tAverage SSIM: {avg_ssim_an:.4f}')


    if num_pl:
        avg_psnr_pl = total_psnr_pl / num_pl
        avg_ssim_pl = total_ssim_pl / num_pl
        log(log_file, f'Average PSNR in puck_lagoon: {avg_psnr_pl:.4f} dB \tAverage SSIM: {avg_ssim_pl:.4f}')

def main(args):

    ## make output directory
    date_ = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = f"results/{date_}_{args.img_output_folder}"
    os.makedirs(output_path)

    ## logging in text_file
    log_file = open(os.path.join(output_path, "log.txt"), "a")

    log(log_file, "Used parameters...")
    for arg in vars(args):
        log(log_file, "\t" + str(arg) + " : " + str(getattr(args, arg)))

    # Transforms for input images and target images. Resize the target image to 36x36.
    transform = T.Compose([
        ResizeWithCV2((64, 64), 'bilinear'),
        T.ToTensor()
    ])

    target_transform = T.Compose([
        ResizeWithCV2((128, 128), 'bilinear'),
        T.ToTensor(),
        #ResizeWithCV2((args.target_w, args.target_h),args.interpolate_mode)
    ])

    #device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Define model and load pretrained model if exists
    model = HAT(upscale=args.upscale,
    in_chans=3,
    img_size=64,
    window_size=16,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    img_range=1.,
    depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    #depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    #num_heads=[6, 6, 6, 6, 6, 6],
    num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv')

    # Loss function and optimizer
    criterion = nn.L1Loss() #ssim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare dataloaders
    configs=[]
    if args.train_combine_dataset:
        configs = [{
            "dataset_name": "MagicBathyNet",
            "root_dir": "./datasets/data/",
            #"root_dir": "/faststorage/cv4rs_2024_superpixel/super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification/datasets/data",
            "transform": transform,
            "target_transform":target_transform,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "test_size": 0.2,
            "val_size": 0.1,
            "bathymetry": True
        }]
    else:
        configs.append ({
            "dataset_name": "MagicBathyNet",
            "root_dir": "./datasets/data/",
            #"root_dir": "/faststorage/cv4rs_2024_superpixel/super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification/datasets/data",
            "transform": transform,
            "target_transform":target_transform,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "test_size": 0.2,
            "val_size": 0.1,
            "locations": ["agia_napa"],
            "bathymetry": True
        })
        configs.append ({
            "dataset_name": "MagicBathyNet",
            "root_dir": "./datasets/data/",
            #"root_dir": "/faststorage/cv4rs_2024_superpixel/super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification/datasets/data",
            "transform": transform,
            "target_transform":target_transform,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "test_size": 0.2,
            "val_size": 0.1,
            "locations": ["puck_lagoon"],
            "bathymetry": True
        })
    
    for config in configs:
        # if args.pretrained_weights_path:
        #     state_dict = torch.load(args.pretrained_weights_path)
        #     if 'params_ema' in state_dict.keys():
        #         model.load_state_dict(state_dict['params_ema'],strict=True)
        #     else:
        #         model.load_state_dict(state_dict,strict=True)

        if args.fine_tune:
            model = unfreeze_selected_layers(log_file,model, 1)
        
        model.to(device)

        data_loader_manager = dataset_loader(**config)
        train_loader = data_loader_manager.get_dataloader("train")
        val_loader = data_loader_manager.get_dataloader('val')
        dataset = data_loader_manager.datasets['train']

        # Train model and save the best model in save_path
        #train(log_file, args, model,train_loader,criterion,optimizer,device,args.epochs,validation_loader=val_loader,save_path=args.best_model_path)

        # Load the best trained model
        model.load_state_dict(torch.load(args.best_model_path, map_location=torch.device('cpu')),strict=True)
        model.to(device)

        data_loader_manager.batch_size = 1
        if data_loader_manager.batch_size != 1:
            raise ValueError("batch size must be 1")
        test_loader = data_loader_manager.get_dataloader('test')

        # Predict on test set and sotre the images and labels in output path
        predict_and_save(dataset, log_file, args, model, test_loader, args.img_output_folder, device, args.interpolate_mode, target_size=(args.target_w,args.target_h))

    log_file.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)