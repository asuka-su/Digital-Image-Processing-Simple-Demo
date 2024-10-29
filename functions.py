import numpy as np
import gradio as gr

from utils import *

def function_hw1(input_image, h_slider, s_slider, l_slider, r_visible, g_visible, b_visible):
    if input_image is None:
        raise gr.Error('WHERE is your input image?', duration=5)

    PI = np.pi
    hsl_image = RGB2HSL(input_image)

    h = hsl_image[:,:,0] + h_slider * 2 * PI
    h = np.select(
        [h < 0, h >= 2 * PI], 
        [h + 2 * PI, h - 2 * PI], 
        default=h
    )

    s = hsl_image[:,:,1]
    s = np.clip(s * (1 + s_slider), 0, 1)

    l = hsl_image[:,:,2]
    l = np.clip(l * (1 + l_slider), 0, 1)

    temp = HSL2RGB(np.dstack((h, s, l)))
    output_image = HSL2RGB(np.dstack((h, s, l)))
    if not r_visible:
        output_image[:,:,0] = 0
    if not g_visible:
        output_image[:,:,1] = 0
    if not b_visible:
        output_image[:,:,2] = 0

    return output_image, temp[:,:,0], temp[:,:,1], temp[:,:,2]

def function_hw2(input_image, method, scaling, use_cv2):
    if input_image is None:
        raise gr.Error('WHERE is your input image?', duration=5)    

    if method == 'Nearest':
        return resize_nearest(input_image, scaling, use_cv2)
    elif method == 'Bilinear':
        return resize_bilinear(input_image, scaling, use_cv2)
    elif method == 'Bicubic':
        return resize_bicubic(input_image, scaling, use_cv2)
    elif method == 'Lanczos':
        if not use_cv2:
            raise gr.Error('Lanczos is not implemented yet...try with cv2', duration=5)
        return resize_lanczos(input_image, scaling, use_cv2)
    else:
        raise gr.Error(f'Unknown resize method choice {method}', duration=5)

def function_hw2_ex(input_image, method, center_x, center_y, deg, R_value, G_value, B_value, side, size):
    if input_image is None:
        raise gr.Error('WHERE is your input image?', duration=5)
    
    if method == 'Rotation':
        w, h, _ = input_image.shape
        if w <= int(center_x) or h <= int(center_y):
            gr.Warning(f'Rotation center out of bound, size: ({w}, {h}), center: ({center_x}, {center_y})', duration=5)
        return rotate(input_image, center_x, center_y, deg, [int(R_value), int(G_value), int(B_value)])
    elif method == 'Shearing':
        return shearing(input_image, side, size, [int(R_value), int(G_value), int(B_value)])
    
    return input_image

def function_hw3(input_image):
    if input_image is None:
        raise gr.Error('WHERE is your input image?', duration=5)   
    output_image = input_image
    return output_image

def function_hw4(input_image):
    if input_image is None:
        raise gr.Error('WHERE is your input image?', duration=5)
    output_image = input_image
    return output_image

def function_hw5(input_image):
    if input_image is None:
        raise gr.Error('WHERE is your input image?', duration=5)
    output_image = input_image
    return output_image