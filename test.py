from skimage.transform import resize
import numpy as np
import tifffile
import matplotlib.pyplot as plt

path = '/Users/x-yq/Desktop/CV4RS/super-resolution/images/HAT_x4_ne/img_410.tif'
img = tifffile.imread(path)
img = resize(img, (700,700),order=0,mode='reflect',anti_aliasing=False)
output_path = '/Users/x-yq/Desktop/CV4RS/super-resolution/images/HAT_x4_ne/img_410.png'
plt.imsave(output_path,img)