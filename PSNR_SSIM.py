import numpy as np
import torch.nn.functional as F
import torch
import cv2
from datasets.dataset_loader import dataset_loader
import torchvision.transforms as T
import tifffile

class PSNR_SSIM:

    def calculate_psnr_pt(self, img, img2, crop_border, img_size=None, **kwargs):
        """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

        Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.

        Returns:
            float: PSNR result.
        """

        if img_size:
            img = F.interpolate(img, img_size ,mode = 'bilinear')
            img2 = F.interpolate(img2, img_size ,mode = 'bilinear')

        if img2.shape[1] != img.shape[1]:
            img = img.repeat(1, img2.shape[1], 1, 1)

        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

        if crop_border != 0:
            img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

        img = img.to(torch.float32)
        img2 = img2.to(torch.float32)

        mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
        return 10. * torch.log10(1. / (mse + 1e-8))

    def calculate_ssim_pt(self, img, img2, crop_border, img_size=None, **kwargs):
        """Calculate SSIM (structural similarity) (PyTorch version).

        ``Paper: Image quality assessment: From error visibility to structural similarity``

        The results are the same as that of the official released MATLAB code in
        https://ece.uwaterloo.ca/~z70wang/research/ssim/.

        For three-channel images, SSIM is calculated for each channel and then
        averaged.

        Args:
            img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.

        Returns:
            float: SSIM result.
        """

        if img_size:
            img = F.interpolate(img, img_size ,mode = 'bilinear')
            img2 = F.interpolate(img2, img_size ,mode = 'bilinear')
            
        if img2.shape[1] != img.shape[1]:
            img = img.repeat(1, img2.shape[1], 1, 1)

        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

        if crop_border != 0:
            img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

        img = img.to(torch.float32)
        img2 = img2.to(torch.float32)

        ssim = self._ssim_pth(img * 255., img2 * 255.)
        return ssim

    def calculate_psnr(self, img, img2, crop_border=0, input_order='HWC', **kwargs):
        """Calculate PSNR (Peak Signal-to-Noise Ratio).

        Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            img (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

        Returns:
            float: PSNR result.
        """

        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        if input_order not in ['HWC', 'CHW']:
            raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
        img = self.reorder_image(img, input_order=input_order)
        img2 = self.reorder_image(img2, input_order=input_order)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        img = img.astype(np.float32)
        img2 = img2.astype(np.float32)

        mse = np.mean((img - img2)**2)
        if mse == 0:
            return float('inf')
        return 10. * np.log10(255. * 255. / mse)

    def calculate_ssim(self, img, img2, crop_border, input_order='HWC', **kwargs):
        """Calculate SSIM (structural similarity).

        ``Paper: Image quality assessment: From error visibility to structural similarity``

        The results are the same as that of the official released MATLAB code in
        https://ece.uwaterloo.ca/~z70wang/research/ssim/.

        For three-channel images, SSIM is calculated for each channel and then
        averaged.

        Args:
            img (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                Default: 'HWC'.

        Returns:
            float: SSIM result.
        """

        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

        if input_order not in ['HWC', 'CHW']:
            raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
        img = self.reorder_image(img, input_order=input_order)
        img2 = self.reorder_image(img2, input_order=input_order)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        img = img.astype(np.float32)
        img2 = img2.astype(np.float32)

        ssims = []
        for i in range(img.shape[2]):
            ssims.append(self._ssim(img[..., i], img2[..., i]))
        return np.array(ssims).mean()
    
    def reorder_image(self, img, input_order='HWC'):
        """Reorder images to 'HWC' order.

        If the input_order is (h, w), return (h, w, 1);
        If the input_order is (c, h, w), return (h, w, c);
        If the input_order is (h, w, c), return as it is.

        Args:
            img (ndarray): Input image.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                If the input image shape is (h, w), input_order will not have
                effects. Default: 'HWC'.

        Returns:
            ndarray: reordered image.
        """

        if input_order not in ['HWC', 'CHW']:
            raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
        if len(img.shape) == 2:
            img = img[..., None]
        if input_order == 'CHW':
            img = img.transpose(1, 2, 0)
        return img

    def _ssim(self, img, img2):
        """Calculate SSIM (structural similarity) for one channel images.

        It is called by func:`calculate_ssim`.

        Args:
            img (ndarray): Images with range [0, 255] with order 'HWC'.
            img2 (ndarray): Images with range [0, 255] with order 'HWC'.

        Returns:
            float: SSIM result.
        """

        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()

    def _ssim_pth(self, img, img2):
        """Calculate SSIM (structural similarity) (PyTorch version).

        It is called by func:`calculate_ssim_pt`.

        Args:
            img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

        Returns:
            float: SSIM result.
        """
        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

        mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
        mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

        cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
        return ssim_map.mean([1, 2, 3])


# ## Usage1 compute psnr and ssim based on tensor output and label in batch

# # prepare your model and transform here
# model = ...
# transform = ...
# device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
# # devicde = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configs = {
# "dataset_name": "SpecificImages",
# "root_dir": "./datasets/data/",
# "transform": transform,
# "batch_size": 1,
# "num_workers": 0,
# "test_size": 0.2,
# "val_size": 0.1
# }

# data_loader_manager = dataset_loader(**configs)
# dataloader = data_loader_manager.get_dataloader('test')

# model.eval()
# total_psnr = 0.0
# total_ssim = 0.0
# metric = PSNR_SSIM()

# with torch.no_grad():
#     for inputs, targets in dataloader:
#         inputs = inputs.to(device,dtype=torch.float32)
#         targets = targets.to(device,dtype=torch.float32)

#         outputs = model(inputs)
        
#         ## param img_size used to resize both targets and outputs to the same size using F.interpolate
#         ## (30,30) here for example the x2 upscale
#         psnr = metric.calculate_psnr_pt(targets, outputs, crop_border=0, img_size=(30,30))[0]
#         ssim = metric.calculate_ssim_pt(targets, outputs, crop_border=0, img_size=(30,30))[0]
#         print(f'PSNR: {psnr:.4f} dB \tAverage SSIM: {ssim:.4f}')

#         total_psnr += psnr
#         total_ssim += ssim

# avg_psnr = total_psnr / len(dataloader.dataset)
# avg_ssim = total_ssim / len(dataloader.dataset)
# print(f'Average PSNR: {avg_psnr:.4f} dB \tAverage SSIM: {avg_ssim:.4f}')


# ## Usage2 compute psnr and ssim based on postprocessed image (rgb image with shape W,H,C)

# transform = T.Compose([
#         T.ToTensor()
# ])
# configs = {
#     "dataset_name": "SpecificImages",
#     "root_dir": "./datasets/data/",
#     "transform": transform,
#     "batch_size": 1,
#     "num_workers": 0,
#     "test_size": 0.2,
#     "val_size": 0.1
# }
# data_loader_manager = dataset_loader(**configs)
# dataloader = data_loader_manager.get_dataloader("test")

# metric = PSNR_SSIM()
# total_psnr=0.0
# total_ssim=0.0

# ## Please change these paths to your saved images. The first one should be img_410.tif
# super_resolved_img = ['.../img_410.tif',
#                         '.../img_2987.tif']

# for i, (img, label) in enumerate(dataloader):
#     ## turn label to range of [0,255]
#     label = label[0].permute(1, 2, 0).numpy()
#     label = (label * 255.0).round().astype(np.uint8)

#     output = tifffile.imread(super_resolved_img[i])

#     ## label and the output shape must be the same. Please resize them beforehand if necessary.
#     psnr = metric.calculate_psnr(label, output, crop_border=0)
#     ssim = metric.calculate_ssim(label, output, crop_border=0)

#     total_psnr+=psnr
#     total_ssim+=ssim

#     print(f'PSNR: {psnr:.4f} dB \tAverage SSIM: {ssim:.4f}')

# avg_psnr = total_psnr / len(dataloader.dataset)
# avg_ssim = total_ssim / len(dataloader.dataset)
# print(f'Average PSNR: {avg_psnr:.4f} dB \tAverage SSIM: {avg_ssim:.4f}')