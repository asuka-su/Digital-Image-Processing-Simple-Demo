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