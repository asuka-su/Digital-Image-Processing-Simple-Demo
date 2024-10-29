import cv2
import numpy as np

PI = np.pi

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
