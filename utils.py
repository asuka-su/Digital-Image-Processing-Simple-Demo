import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from math import exp

__all__ = [
    'RGB2HSL', 'HSL2RGB', 
    'resize_nearest', 'resize_bilinear', 'resize_bicubic', 'resize_lanczos', 'rotate', 'shearing', 
    'seed_given_generator', 
    'joint_bilateral_filter', 'guided_filter', 
    'histogram_equalization', 'cla_he' 
]

PI = np.pi

#---------------------------------#
#          hw1 functions          #
#---------------------------------#
def RGB2HSL(rgb_image):
    rgb_image = rgb_image / 255.0
    r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
    
    _max = np.maximum.reduce([r, g, b])
    _min = np.minimum.reduce([r, g, b])

    l = (r + g + b) / 3
    s = np.where(l != 0, 1 - _min / l, 0)
    h = np.where(
        _max != _min, 
        np.select(
            [b < g, b == g, b > g], 
            [
                np.arccos((2 * r - g - b) / (2 * np.sqrt(r * r + g * g + b * b - r * g - r * b - b * g))), 
                np.where(r > g, 0, PI),     # something will go wrong if b==g...
                2 * PI - np.arccos((2 * r - g - b) / (2 * np.sqrt(r * r + g * g + b * b - r * g - r * b - b * g))), 
            ]
        ), 
        0
    )

    return np.dstack((h, s, l))


def HSL2RGB(hsl_image):
    h, s, l = hsl_image[:,:,0], hsl_image[:,:,1], hsl_image[:,:,2]

    b_1 = l * (1 - s)
    r_1 = l * (1 + s * np.cos(h) / np.cos(PI / 3 - h))
    g_1 = 3 * l - r_1 - b_1 
    rgb_1 = np.dstack((r_1, g_1, b_1))

    h_prime_2 = h - 2 * PI / 3
    r_2 = l * (1 - s)
    g_2 = l * (1 + s * np.cos(h_prime_2) / np.cos(PI / 3 - h_prime_2))
    b_2 = 3 * l - r_2 - g_2
    rgb_2 = np.dstack((r_2, g_2, b_2))

    h_prime_3 = h - 4 * PI / 3
    g_3 = l * (1 - s)
    b_3 = l * (1 + s * np.cos(h_prime_3) / np.cos(PI / 3 - h_prime_3))
    r_3 = 3 * l - g_3 - b_3
    rgb_3 = np.dstack((r_3, g_3, b_3))

    condition_1 = (h >= 0) & (h < 2 * PI / 3)
    condition_2 = (h >= 2 * PI / 3) & (h < 4 * PI / 3)
    condition_3 = (h >= 4 * PI / 3) & (h < 2 * PI)
    rgb_image = np.clip(np.select(
        [condition_1[:,:,np.newaxis], condition_2[:,:,np.newaxis], condition_3[:,:,np.newaxis]], 
        [rgb_1, rgb_2, rgb_3]
    ), 0, 1)

    return (255 * rgb_image).astype(np.uint8)


#---------------------------------#
#          hw2 functions          #
#---------------------------------#
def resize_nearest(input_image, scaling, use_cv2):
    scaling = float(scaling.replace('x', ''))
    
    if use_cv2:
        return cv2.resize(input_image, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)

    w, h, _ = input_image.shape
    new_w = int(w * scaling)
    new_h = int(h * scaling)

    w_ind = np.minimum(np.around(np.arange(new_w) / scaling), w - 1).astype(int)
    h_ind = np.minimum(np.around(np.arange(new_h) / scaling), h - 1).astype(int)

    output_image = input_image[w_ind[:,np.newaxis], h_ind].astype(np.uint8)

    return output_image


def resize_bilinear(input_image, scaling, use_cv2):
    scaling = float(scaling.replace('x', ''))
    
    if use_cv2:
        return cv2.resize(input_image, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_LINEAR)

    w, h, _ = input_image.shape
    new_w = int(w * scaling)
    new_h = int(h * scaling)

    w_ind = np.clip(np.arange(new_w) / scaling, 0, w - 1)
    h_ind = np.clip(np.arange(new_h) / scaling, 0, h - 1)

    w_down = np.floor(w_ind).astype(int)
    w_up = np.ceil(w_ind).astype(int)
    h_down = np.floor(h_ind).astype(int)
    h_up = np.ceil(h_ind).astype(int)

    w_offset = (w_ind - np.floor(w_ind))[:,np.newaxis]
    h_offset = (h_ind - np.floor(h_ind))[np.newaxis,:]  # bug found!

    output_image = (((1 - w_offset) * (1 - h_offset))[:,:,np.newaxis] * input_image[w_down[:,np.newaxis], h_down] + \
                    ((1 - w_offset) * h_offset)[:,:,np.newaxis] * input_image[w_down[:,np.newaxis], h_up] + \
                    (w_offset * (1 - h_offset))[:,:,np.newaxis] * input_image[w_up[:,np.newaxis], h_down] + \
                    (w_offset * h_offset)[:,:,np.newaxis] * input_image[w_up[:,np.newaxis], h_up]).astype(np.uint8)
    
    return output_image


def resize_bicubic(input_image, scaling, use_cv2):
    scaling = float(scaling.replace('x', ''))
    
    if use_cv2:
        return cv2.resize(input_image, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_CUBIC)

    w, h, _ = input_image.shape
    new_w = int(w * scaling)
    new_h = int(h * scaling)

    w_ind = np.arange(new_w) / scaling
    h_ind = np.arange(new_h) / scaling

    w_1 = np.clip(np.floor(w_ind) - 1, 0, w - 1).astype(int)
    w_2 = np.clip(np.floor(w_ind), 0, w - 1).astype(int)
    w_3 = np.clip(np.ceil(w_ind), 0, w - 1).astype(int)
    w_4 = np.clip(np.ceil(w_ind) + 1, 0, w - 1).astype(int)
    h_1 = np.clip(np.floor(h_ind) - 1, 0, h - 1).astype(int)
    h_2 = np.clip(np.floor(h_ind), 0, h - 1).astype(int)
    h_3 = np.clip(np.ceil(h_ind), 0, h - 1).astype(int)
    h_4 = np.clip(np.ceil(h_ind) + 1, 0, h - 1).astype(int)
    w_offset = (w_ind - np.floor(w_ind))[:,np.newaxis]
    h_offset = (h_ind - np.floor(h_ind))[np.newaxis:,]

    def S(x):
        if np.abs(x) <= 1:
            return 1 - 2 * (x ** 2) + np.abs(x) ** 3
        elif np.abs(x) <= 2:
            return 4 - 8 * np.abs(x) + 5 * (x ** 2) - np.abs(x) ** 3
        else:
            return 0
        
    s = np.vectorize(S)
    A_11 = s(-1 - w_offset) * s(-1 - h_offset)
    A_12 = s(-1 - w_offset) * s(-h_offset)
    A_13 = s(-1 - w_offset) * s(1 - h_offset)
    A_14 = s(-1 - w_offset) * s(2 - h_offset)
    A_21 = s(-w_offset) * s(-1 - h_offset)
    A_22 = s(-w_offset) * s(-h_offset)
    A_23 = s(-w_offset) * s(1 - h_offset)
    A_24 = s(-w_offset) * s(2 - h_offset)
    A_31 = s(1 - w_offset) * s(-1 - h_offset)
    A_32 = s(1 - w_offset) * s(-h_offset)
    A_33 = s(1 - w_offset) * s(1 - h_offset)
    A_34 = s(1 - w_offset) * s(2 - h_offset)
    A_41 = s(2 - w_offset) * s(-1 - h_offset)
    A_42 = s(2 - w_offset) * s(-h_offset)
    A_43 = s(2 - w_offset) * s(1 - h_offset)
    A_44 = s(2 - w_offset) * s(2 - h_offset)

    output_image = np.clip(A_11[:,:,np.newaxis] * input_image[w_1[:,np.newaxis], h_1] + \
                    A_12[:,:,np.newaxis] * input_image[w_1[:,np.newaxis], h_2] + \
                    A_13[:,:,np.newaxis] * input_image[w_1[:,np.newaxis], h_3] + \
                    A_14[:,:,np.newaxis] * input_image[w_1[:,np.newaxis], h_4] + \
                    A_21[:,:,np.newaxis] * input_image[w_2[:,np.newaxis], h_1] + \
                    A_22[:,:,np.newaxis] * input_image[w_2[:,np.newaxis], h_2] + \
                    A_23[:,:,np.newaxis] * input_image[w_2[:,np.newaxis], h_3] + \
                    A_24[:,:,np.newaxis] * input_image[w_2[:,np.newaxis], h_4] + \
                    A_31[:,:,np.newaxis] * input_image[w_3[:,np.newaxis], h_1] + \
                    A_32[:,:,np.newaxis] * input_image[w_3[:,np.newaxis], h_2] + \
                    A_33[:,:,np.newaxis] * input_image[w_3[:,np.newaxis], h_3] + \
                    A_34[:,:,np.newaxis] * input_image[w_3[:,np.newaxis], h_4] + \
                    A_41[:,:,np.newaxis] * input_image[w_4[:,np.newaxis], h_1] + \
                    A_42[:,:,np.newaxis] * input_image[w_4[:,np.newaxis], h_2] + \
                    A_43[:,:,np.newaxis] * input_image[w_4[:,np.newaxis], h_3] + \
                    A_44[:,:,np.newaxis] * input_image[w_4[:,np.newaxis], h_4], 0, 255).astype(np.uint8)

    return output_image


def resize_lanczos(input_image, scaling, use_cv2):
    scaling = float(scaling.replace('x', ''))

    if use_cv2:
        return cv2.resize(input_image, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_LANCZOS4)

    return input_image


def rotate(input_image, center_x, center_y, deg, default_color):
    center_x = int(center_x)
    center_y = int(center_y)
    w, h, _ = input_image.shape

    rad = deg / 180 * PI

    temph, tempw = np.meshgrid(np.arange(w), np.arange(h))

    w_ind = (tempw - center_x) * np.cos(rad) + (temph - center_y) * np.sin(rad) + center_x
    h_ind = (tempw - center_x) * (-np.sin(rad)) + (temph - center_y) * np.cos(rad) + center_y
    w_bound = (w_ind < -1) | (w_ind > w)
    h_bound = (h_ind < -1) | (h_ind > h)
    in_bound = w_bound | h_bound

    w_ind = np.clip(w_ind, 0, w - 1)
    h_ind = np.clip(h_ind, 0, h - 1)
    w_down = np.floor(w_ind).astype(int)
    w_up = np.ceil(w_ind).astype(int)
    h_down = np.floor(h_ind).astype(int)
    h_up = np.ceil(h_ind).astype(int)

    w_offset = (w_ind - np.floor(w_ind))
    h_offset = (h_ind - np.floor(h_ind))

    output_image = (((1 - w_offset) * (1 - h_offset))[:,:,np.newaxis] * input_image[w_down, h_down] + \
                ((1 - w_offset) * h_offset)[:,:,np.newaxis] * input_image[w_down, h_up] + \
                (w_offset * (1 - h_offset))[:,:,np.newaxis] * input_image[w_up, h_down] + \
                (w_offset * h_offset)[:,:,np.newaxis] * input_image[w_up, h_up]).astype(np.uint8)
    output_image = np.where(in_bound[:,:,np.newaxis], np.array(default_color)[np.newaxis, np.newaxis,:], output_image)
    
    return output_image


def shearing(input_image, side, size, default_color):
    w, h, _ = input_image.shape
    size = float(size)
    temph, tempw = np.meshgrid(np.arange(w), np.arange(h))

    if side == 'Right':
        w_ind = tempw - size * temph
        h_ind = temph
    elif side == 'Down':
        w_ind = tempw
        h_ind = temph - size * tempw
    elif side == 'Left':
        w_ind = tempw - size * (h - temph)
        h_ind = temph
    elif side == 'Up':
        w_ind = tempw
        h_ind = temph - size * (w - tempw)
    else:
        return input_image

    w_bound = (w_ind < -1) | (w_ind > w)
    h_bound = (h_ind < -1) | (h_ind > h)
    in_bound = w_bound | h_bound

    w_ind = np.clip(w_ind, 0, w - 1)
    h_ind = np.clip(h_ind, 0, h - 1)
    w_down = np.floor(w_ind).astype(int)
    w_up = np.ceil(w_ind).astype(int)
    h_down = np.floor(h_ind).astype(int)
    h_up = np.ceil(h_ind).astype(int)

    w_offset = (w_ind - np.floor(w_ind))
    h_offset = (h_ind - np.floor(h_ind))

    output_image = (((1 - w_offset) * (1 - h_offset))[:,:,np.newaxis] * input_image[w_down, h_down] + \
                ((1 - w_offset) * h_offset)[:,:,np.newaxis] * input_image[w_down, h_up] + \
                (w_offset * (1 - h_offset))[:,:,np.newaxis] * input_image[w_up, h_down] + \
                (w_offset * h_offset)[:,:,np.newaxis] * input_image[w_up, h_up]).astype(np.uint8)
    output_image = np.where(in_bound[:,:,np.newaxis], np.array(default_color)[np.newaxis, np.newaxis,:], output_image)
    
    return output_image


#---------------------------------#
#          hw3 functions          #
#---------------------------------#
image_size = 64
nc = 3
nz = 100
ngf = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(ngf * 8), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ngf * 4), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ngf * 2), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ngf), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), 
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


netG = None
def load_model():
    global netG
    if netG is None:
        netG = Generator().to(device)
        netG.load_state_dict(torch.load('./assets/checkpoint/dcgan_checkpoint.pth'))
    return netG

def seed_given_generator(seed):
    netG = load_model()
    torch.manual_seed(seed)
    test_batch_size = 64
    noise = torch.randn(test_batch_size, nz, 1, 1, device=device)
    with torch.no_grad():
        fake = netG(noise).detach().cpu()
    vis = vutils.make_grid(fake, padding=2, normalize=True)
    vis = (vis.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    
    return vis


#---------------------------------#
#          hw4 functions          #
#---------------------------------#
def joint_bilateral_filter(input_image, guidance_image, filter_size, sigma_s, sigma_r, use_cv2):
    if use_cv2:
        return cv2.ximgproc.jointBilateralFilter(guidance_image, input_image, d = 2 * filter_size + 1, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    
    # how to improve using numpy?
    filter_shift = [(i, j) for i in range(-filter_size, filter_size + 1) for j in range(-filter_size, filter_size + 1)]
    w, h, _ = input_image.shape
    output_image = np.zeros_like(input_image)
    guidance_image = cv2.cvtColor(guidance_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    input_image = input_image.astype(np.float32)

    for x in range(w):
        for y in range(h):
            pixel_Ii = guidance_image[x, y]
            filter_weight = []

            for shift in filter_shift:
                x_cur, y_cur = x + shift[0], y + shift[1]
                if (x_cur < 0 or y_cur < 0 or x_cur >= w or y_cur >= h):
                    continue
                pixel_Ij = guidance_image[x_cur, y_cur]
                pos = exp(-(shift[0] ** 2 + shift[1] ** 2) / (sigma_s ** 2))
                color = exp(-((pixel_Ii - pixel_Ij) ** 2) / (sigma_r ** 2))
                filter_weight.append(pos * color)
            filter_weight = [t / sum(filter_weight) for t in filter_weight]

            pixel_outi = np.zeros_like(input_image[x, y, :]).astype(np.float32)
            id = 0
            for shift in filter_shift:
                x_cur, y_cur = x + shift[0], y + shift[1]
                if (x_cur < 0 or y_cur < 0 or x_cur >= w or y_cur >= h):
                    continue
                pixel_outi += input_image[x_cur, y_cur, :] * filter_weight[id]
                id += 1
            output_image[x, y, :] = pixel_outi
    
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return output_image

    
def guided_filter(input_image, guidance_image, filter_size, eps, use_cv2):
    if use_cv2:
        return cv2.ximgproc.guidedFilter(guidance_image, input_image, 2 * filter_size + 1, eps)
    
    w, h, _ = input_image.shape
    filter_weight_a = np.zeros((w, h))
    filter_weight_b = np.zeros((w, h))
    input_image = input_image.astype(np.float32)
    guidance_image = guidance_image.astype(np.float32)
    output_image = np.zeros_like(input_image)

    for x in range(w):
        for y in range(h):
            x_start = max(x - filter_size, 0)
            x_end = min(x + filter_size, w - 1) + 1
            y_start = max(y - filter_size, 0)
            y_end = min(y + filter_size, h - 1) + 1
            I_k = guidance_image[x_start:x_end, y_start:y_end, :]
            p_k = input_image[x_start:x_end, y_start:y_end, :]
            filter_weight_a[x, y] = ((I_k * p_k).mean() - I_k.mean() * p_k.mean()) / ((I_k * I_k).mean() - (I_k.mean()) ** 2 + eps)
            filter_weight_b[x, y] = p_k.mean() - filter_weight_a[x, y] * I_k.mean()
    
    for x in range(w):
        for y in range(h):
            x_start = max(x - filter_size, 0)
            x_end = min(x + filter_size, w - 1) + 1
            y_start = max(y - filter_size, 0)
            y_end = min(y + filter_size, h - 1) + 1
            filter_weight_a_k = filter_weight_a[x_start:x_end, y_start:y_end].mean()
            filter_weight_b_k = filter_weight_b[x_start:x_end, y_start:y_end].mean()
            output_image[x, y, :] = filter_weight_a_k * guidance_image[x, y, :] + filter_weight_b_k
    
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return output_image

#---------------------------------#
#          hw5 functions          #
#---------------------------------#
def histogram_equalization(input_image):
    w, h, _ = input_image.shape
    hsl_image = RGB2HSL(input_image)
    l_space = hsl_image[:,:,2]

    cdf = np.zeros((256))
    for x in range(w):
        for y in range(h):
            cdf[round(l_space[x, y] * 255)] += 1
    
    cdf = [t / (w * h) for t in cdf]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + cdf[i]
    for x in range(w):
        for y in range(h):
            hsl_image[x, y, 2] = cdf[round(l_space[x, y] * 255)]
    
    eps = 0.02
    hsl_image[:, :, 1] = np.where(
        hsl_image[:, :, 1] + eps > 1, 
        1, hsl_image[:, :, 1] + eps
    )
    
    return HSL2RGB(hsl_image)


def cla_he(input_image, limit, grid_size):
    w, h, _ = input_image.shape
    hsl_image = RGB2HSL(input_image)
    l_space = hsl_image[:,:,2]
    mapping = np.zeros((grid_size[0], grid_size[1], 256))

    x_start = [t * (w // grid_size[0]) for t in range(grid_size[0])] + [w]
    x_center = [(x_start[i] + x_start[i + 1]) / 2 for i in range(grid_size[0])]
    y_start = [t * (h // grid_size[1]) for t in range(grid_size[1])] + [h]
    y_center = [(y_start[j] + y_start[j + 1]) / 2 for j in range(grid_size[1])]

    if limit <= 0 or limit > 255:
        limit = 255

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            l_operate = l_space[x_start[i]:x_start[i + 1], y_start[j]:y_start[j + 1]]
            x_len = x_start[i + 1] - x_start[i]
            y_len = y_start[j + 1] - y_start[j]
            
            cdf = np.zeros((256))
            for m in range(x_len):
                for n in range(y_len):
                    cdf[round(l_operate[m, n] * 255)] += 1
            
            clip_limit = int(limit * x_len * y_len / 255)
            clipped = 0
            for k in range(256):
                if cdf[k] > clip_limit:
                    clipped += cdf[k] - clip_limit
                    cdf[k] = clip_limit
            clip_distribute = clipped // 256
            clip_residual = clipped % 256
            for k in range(256):
                cdf[k] += clip_distribute
                if k < clip_residual:
                    cdf[k] += 1
            
            cdf = [t / (x_len * y_len) for t in cdf]
            for k in range(1, 256):
                cdf[k] = cdf[k] + cdf[k - 1]
            mapping[i, j, :] = cdf
    
    blockx, blocky = 0, 0
    def update_block(x, block, center_list):
        if x == 0:
            return 0
        if block == len(center_list):
            return block
        if x >= center_list[block]:
            return block + 1
        return block

    for x in range(w):
        blockx = update_block(x, blockx, x_center)
        for y in range(h):
            blocky = update_block(y, blocky, y_center)
            index_l = round(l_space[x, y] * 255)
            if (
                (blockx == 0 and blocky == 0) or
                (blockx == grid_size[0] and blocky == 0) or 
                (blockx == 0 and blocky == grid_size[1]) or
                (blockx == grid_size[0] and blocky == grid_size[1])
            ):
                hsl_image[x, y, 2] = mapping[max(0, blockx - 1), max(0, blocky - 1), index_l]
            elif (blockx == 0 or blockx == grid_size[0]):
                eps = (y - y_center[blocky - 1]) / (y_center[blocky] - y_center[blocky - 1])
                hsl_image[x, y, 2] = (1 - eps) * mapping[max(0, blockx - 1), blocky - 1, index_l] + eps * mapping[max(0, blockx - 1), blocky, index_l]
            elif (blocky == 0 or blocky == grid_size[1]):
                eps = (x - x_center[blockx - 1]) / (x_center[blockx] - x_center[blockx - 1])
                hsl_image[x, y, 2] = (1 - eps) * mapping[blockx - 1, max(0, blocky - 1), index_l] + eps * mapping[blockx, max(0, blocky - 1), index_l]
            else:
                eps_x = (x - x_center[blockx - 1]) / (x_center[blockx] - x_center[blockx - 1])
                eps_y = (y - y_center[blocky - 1]) / (y_center[blocky] - y_center[blocky - 1])
                hsl_image[x, y, 2] = (1 - eps_x) * (1 - eps_y) * mapping[blockx - 1, blocky - 1, index_l] + \
                    (1 - eps_x) * eps_y * mapping[blockx - 1, blocky, index_l] + \
                    eps_x * (1 - eps_y) * mapping[blockx, blocky - 1, index_l] + \
                    eps_x * eps_y * mapping[blockx, blocky, index_l]
    
    eps = 0.02
    hsl_image[:, :, 1] = np.where(
        hsl_image[:, :, 1] + eps > 1, 
        1, hsl_image[:, :, 1] + eps
    )
    
    return HSL2RGB(hsl_image)
