import tifffile
from osgeo import gdal
import numpy as np
from skimage.transform import resize
import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
from PIL import Image
import scipy


#file = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/Bathymetry_predict/ft_combined_imagery_trained_model/puck_lagoon/depth_2176.tif'
#file = '/faststorage/cv4rs_2024_superpixel/super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification/datasets/data/puck_lagoon/depth/spot6/depth_2176.tif'
#file = '/Users/x-yq/Desktop/CV4RS/super-resolution/images/Bathymetry_predict/ft_combined_imagery_trained_model/puck_lagoon/depth_61.tif'


def resize_image(input_file, target_size, output_file, order=1):
    # Open the image file
    if input_file.lower().endswith('.jpg'):
        img = Image.open(input_file)
        img = np.array(img)
    else:
        img = tifffile.imread(input_file)

    # Resize the image
    # ratio0 = target_size[0] / float(img.shape[0])
    # ratio1 = target_size[1] / float(img.shape[1])
    # img_resized = scipy.ndimage.zoom(img, (ratio0, ratio1), order=order)
    img_resized = resize(img, target_size, order=order, mode='reflect', anti_aliasing=False)
    
    # Ensure the data type matches the input image
    img_resized = img_resized.astype(img.dtype)

    dir = os.path.dirname(output_file)
    os.makedirs(dir, exist_ok=True)

    # Save the resized image
    if input_file.lower().endswith('.jpg'):
        img_resized = Image.fromarray(img_resized)
        img_resized.save(output_file, 'JPEG')
    else:
        tifffile.imwrite(output_file, img_resized)

def save_img_as_jpg(input_file, output_file, colormap='viridis', bathy=True,  vmin=None, vmax=None, denorm=False):
    
    img = tifffile.imread(input_file)
    
    img_min, img_max = img.min(), img.max()
    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    img = img[..., [2,1,0]]

    # Save the image as a JPG using matplotlib
    dir = os.path.dirname(output_file)
    os.makedirs(dir, exist_ok=True)
    #output_file = os.path.join(dir, str(os.path.basename(output_file)).replace('.tif', '.jpg'))
    output_file = os.path.join(dir, str(os.path.basename(output_file)).replace('.tif', '.tif'))

    #tifffile.imwrite(output_file, img)
    
    if bathy:
        plt.imsave(output_file, img, cmap=colormap, vmin = vmin, vmax=vmax)
    else:
        plt.imsave(output_file, img, cmap=None)

###########################################################################################################

def resize_folder(folder, target_size, output_folder, order=1, format='*.tif', bathy=False):
    
    image_paths = glob.glob(os.path.join(folder, '**', format), recursive=True)

    for file in image_paths:
        # Open the image file
        if file.lower().endswith('.jpg'):
            img = Image.open(file)
            img = np.array(img)
        else:
            img = tifffile.imread(file)

        # Resize the image
        if bathy:
            ratio0 = target_size[0] / float(img.shape[0])
            ratio1 = target_size[1] / float(img.shape[1])
            img_resized = scipy.ndimage.zoom(img, (ratio0, ratio1), order=0)
            #img_resized = resize(img, target_size, order=order, mode='reflect', anti_aliasing=False)
        else:
            ratio0 = target_size[0] / float(img.shape[0])
            ratio1 = target_size[1] / float(img.shape[1])
            img_resized = scipy.ndimage.zoom(img, (ratio0, ratio1, 1), order=0)
            #img_resized = resize(img, target_size, order=order, mode='reflect', anti_aliasing=False)
        
        # Ensure the data type matches the input image
        #img_resized = img_resized.astype(img.dtype)

        # # Create output directory
        # dir = os.path.join(os.path.dirname(os.path.dirname(file)), f'vis_{norm_status}')
        
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, os.path.basename(file))

        # Save the resized image
        if file.lower().endswith('.jpg'):
            img_resized = Image.fromarray(img_resized)
            img_resized.save(output_file, 'JPEG')
        else:
            tifffile.imwrite(output_file, img_resized)

def save_folder_as_jpg(folder, output_folder, colormap='viridis', source_format='*.tif', bathy=True, vmin=None, vmax=None):

    vmin = np.inf
    vmax = -np.inf
    images_data = []

    image_paths = glob.glob(os.path.join(folder, '**', source_format), recursive=True)

    # First iteration: Find vmin and vmax, store image data
    for file in image_paths:
        img = np.array(tifffile.imread(file))
        img_remaped = (img * 255.0).astype(np.uint8)
        images_data.append((img_remaped, file))
        vmin = min(vmin, np.min(img))
        vmax = max(vmax, np.max(img))

    print('vmin: ', vmin)
    print('vmax: ', vmax)

    os.makedirs(output_folder, exist_ok=True)

    # Second step: Save images with vmin and vmax
    for img_remaped, file in images_data:

        output_path = os.path.join(output_folder, str(os.path.basename(file)).replace('.tif', '.jpg'))

        if bathy:
            plt.imsave(output_path, img, cmap=colormap, vmin = vmin, vmax = vmax)
        else:
            plt.imsave(output_path, img, cmap=None)

    
def check_range(file, gt_file, channel_num=1):

    img = tifffile.imread(file)
    gt_img = tifffile.imread(gt_file)

    print(img.min(), img.max())
    print(gt_img.min(),gt_img.max())
    print(img)

    # for i in range(channel_num):
    #     print('channel ', i)
    #     if channel_num == 3:
    #         print('denormalized img:', img[:,:,i].min(), img[:,:,i].max(), img.shape)
    #         print('ground truth spot6 img:', gt_img[:,:,i].min(), gt_img[:,:,i].max(), gt_img.shape)
    #     else:
    #         print('denormalized img:', img.min(), img.max(), img.shape)
    #         print('ground truth spot6 img:', gt_img.min(), gt_img.max(), gt_img.shape)

    #print(img)


# input_file = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/Bathymetry_ft/ft_combined_norm_and_denorm/puck_lagoon/denorm/depth_1231.tif'
# gt_file = '/faststorage/cv4rs_2024_superpixel/super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification/datasets/data/puck_lagoon/depth/spot6/depth_1231.tif'
# resize_image(input_file, (30,30), input_file)
# check_range(input_file, gt_file)
# output_file = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/Bathymetry_ft/ft_combined_norm_and_denorm/puck_lagoon/vis_jpg/depth_1231.jpg'
# save_img_as_jpg(input_file, output_file, bathy=True)

# #input_folder = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/Bathymetry_ft/ft_combined_norm_and_denorm/puck_lagoon/denorm'
# input_folder = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/Bathymetry_ft/ft_combined_norm_and_denorm/agia_napa/denorm/'
# #output_folder = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/Bathymetry_ft/ft_combined_norm_and_denorm/puck_lagoon/denorm_vis'
# output_folder = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/Bathymetry_ft/ft_combined_norm_and_denorm/agia_napa/denorm_vis'
# resize_folder(input_folder,(700,700),output_folder)
# save_folder_as_jpg(output_folder,output_folder,bathy=True)

# f = 'denorm_resized'
# taks_class = 'SR_IMG'
# task = 'ft_pl_best_loss'

# location = 'puck_lagoon'
# input_folder = f'/faststorage/cv4rs_2024_superpixel/Yeqiao/images/{taks_class}/{task}/{location}/denorm'
# output_folder = f'/faststorage/cv4rs_2024_superpixel/Yeqiao/images/{taks_class}/{task}/{location}/denorm_resized'
# resize_folder(input_folder,(30,30),output_folder,order=0, bathy=False)

# location = 'agia_napa'
# input_folder = f'/faststorage/cv4rs_2024_superpixel/Yeqiao/images/{taks_class}/{task}/{location}/denorm'
# output_folder = f'/faststorage/cv4rs_2024_superpixel/Yeqiao/images/{taks_class}/{task}/{location}/denorm_resized'
# resize_folder(input_folder,(30,30),output_folder,order=0, bathy=False)

#input_file = f'/faststorage/cv4rs_2024_superpixel/super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification/datasets/data/puck_lagoon/depth/spot6/depth_2987.tif'
# input_file2 = f'/Users/x-yq/Desktop/CV4RS/HAT_pre_bathy_on_sr_img/pretrained_410.tif'
# check_range(input_file2,input_file2, 3)


# input_file = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/SR_IMG/ft_pl_best_loss/puck_lagoon/norm/img_2987.tif'
# output_file = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/SR_IMG/ft_pl_best_loss/puck_lagoon/norm_vis/ft_pl_2987.tif'
# resize_image(input_file, (700,700), output_file,order=0)
# save_img_as_jpg(output_file,output_file,bathy=False)


input_file = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/SR_IMG/pretrained_bilinear/puck_lagoon/denorm/img_2987.tif'
output_file = '/faststorage/cv4rs_2024_superpixel/Yeqiao/images/SR_IMG/pretrained_bilinear/puck_lagoon/denorm_vis/pretrained_2987.jpg'
save_img_as_jpg(input_file,output_file,bathy=False,denorm=True)
resize_image(output_file, (700,700), output_file,order=0)

