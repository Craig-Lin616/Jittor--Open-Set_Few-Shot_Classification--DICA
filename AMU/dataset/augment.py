import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


# 翻转图像
def flip_image(image, mode='horizontal'):
    if mode == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("Mode should be 'horizontal' or 'vertical'")


# 旋转图像
def rotate_image(image, angle):
    return image.rotate(angle)


# 缩放图像
def scale_image(image, scale_factor):
    width, height = image.size
    return image.resize((int(width * scale_factor), int(height * scale_factor)))


# 裁剪图像
def crop_image(image, crop_box):
    return image.crop(crop_box)


# 调整亮度、对比度、饱和度、色调
def adjust_color(image, brightness=1, contrast=1, saturation=1, hue=1):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation)
    # hue adjustment not directly available in PIL, skipped
    return image


# 添加噪声
def add_noise(image, noise_type='gaussian', mean=0, std=1):
    # This function is a placeholder; PIL doesn't support direct noise addition
    return image


# 模糊图像
def blur_image(image, blur_type='gaussian', radius=2):
    if blur_type == 'gaussian':
        return image.filter(ImageFilter.GaussianBlur(radius))
    elif blur_type == 'motion':
        return image.filter(
            ImageFilter.MotionBlur(radius))  # Pillow doesn't have MotionBlur, custom implementation needed
    else:
        raise ValueError("Blur type should be 'gaussian' or 'motion'")


# 仿射变换
def affine_transform(image, matrix):
    return image.transform(image.size, Image.AFFINE, matrix)


def pad_img(img):
    img_array = np.array(img)
    orig_w, orig_h = img.width, img.height
    max_size = max(orig_h, orig_w)
    h_padding = (max_size - orig_h) / 2.  # horizontal padding
    v_padding = (max_size - orig_w) / 2.
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    l_pad, r_pad, t_pad, b_pad = int(l_pad), int(r_pad), int(t_pad), int(b_pad)
    if img.mode == 'L':  # gray image
        try:
            img_array_pad = np.pad(img_array, ((l_pad, r_pad), (t_pad, b_pad)), mode='constant', constant_values=0)
        except:
            print('xx')
    elif img.mode == 'RGB':
        try:
            img_array_pad = np.pad(img_array, ((l_pad, r_pad), (t_pad, b_pad), (0, 0)), mode='constant',
                                   constant_values=0)
        except:
            print('xx')
    elif img.mode == 'RGBA' or img.mode == 'CMYK':
        try:
            img_array_pad = np.pad(img_array, ((l_pad, r_pad), (t_pad, b_pad), (0, 0)), mode='constant',
                                   constant_values=0)
        except:
            print('xx')
    else:
        print('xx')
    assert img_array_pad.shape[0] == img_array_pad.shape[1]

    return img_array_pad
