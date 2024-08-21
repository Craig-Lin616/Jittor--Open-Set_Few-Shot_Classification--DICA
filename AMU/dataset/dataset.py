import math
import os
import random

import jclip as clip
from utils.utils import *
from jclip.amu import *
from jittor.transform import _setup_size
from jittor.dataset import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize, RandomResizedCrop, \
    RandomHorizontalFlip, RandomCrop
from PIL import Image
import PIL
from .augment import *

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class ImageToTensor(object):

    def __call__(self, input):
        input = np.asarray(input)
        if len(input.shape) < 3:
            input = np.expand_dims(input, -1)
        return to_tensor(input)

class Resize:

    def __init__(self, size, mode=Image.BILINEAR):
        if isinstance(size, int):
            self.size = size
        else:
            self.size = _setup_size(
                size,
                error_msg="If size is a sequence, it should have 2 values")
        self.mode = mode

    def __call__(self, img: Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        if isinstance(self.size, int):
            w, h = img.size

            short, long = (w, h) if w <= h else (h, w)
            if short == self.size:
                return img

            new_short, new_long = self.size, int(self.size * long / short)
            new_w, new_h = (new_short, new_long) if w <= h else (new_long,
                                                                 new_short)
            size = (new_h, new_w)
        return resize(img, size, self.mode)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    # assert -30 <= v <= 30
    # if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert v >= 0.0
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert 0 <= v
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l


def augment_list_no_color():
    l = [
        # (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        # (Color, 0.05, 0.95),
        # (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        # (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        # (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l

class RandAugment:
    def __init__(self, n, m, exclude_color_aug=False):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        if not exclude_color_aug:
            self.augment_list = augment_list()
        else:
            self.augment_list = augment_list_no_color()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)  # for fixmatch
        return img

def _transform(n_px):
    return Compose([
        Resize(n_px, mode=Image.BICUBIC),
        CenterCrop(n_px), _convert_image_to_rgb,
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])

aux_transform = Compose([
    Resize(224, mode=Image.BICUBIC),
    CenterCrop(224), _convert_image_to_rgb,
    ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ImageToTensor()
])

def transform_weak_clip(n_px):
    return Compose([
        Resize(n_px, mode=Image.BICUBIC),
        RandomCrop((n_px, n_px)), _convert_image_to_rgb,
        RandomHorizontalFlip(),
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])

def transform_weak_aux(n_px):
    return Compose([
        Resize(n_px, mode=Image.BICUBIC),
        RandomCrop((n_px, n_px)), _convert_image_to_rgb,
        RandomHorizontalFlip(),
        ImageNormalize((0.485, 0.456, 0.406),
                       (0.229, 0.224, 0.225)),
        ImageToTensor()
    ])

def transform_strong_clip(n_px):
    return Compose([
        Resize(n_px, mode=Image.BICUBIC),
        RandomCrop((n_px, n_px)), _convert_image_to_rgb,
        RandomHorizontalFlip(),
        RandAugment(3, 5),
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])

def transform_strong_aux(n_px):
    return Compose([
        Resize(n_px, mode=Image.BICUBIC),
        RandomCrop((n_px, n_px)), _convert_image_to_rgb,
        RandomHorizontalFlip(),
        RandAugment(3, 5),
        ImageNormalize((0.485, 0.456, 0.406),
                       (0.229, 0.224, 0.225)),
        ImageToTensor()
    ])

class LabeledImageDataset(Dataset):
    def __init__(self, images, labels, img_dir='/root/autodl-tmp/Dataset/', transform=_transform(224), transform_aux=aux_transform):
        super().__init__()
        self.img_dir = img_dir
        self.images = images
        self.transform = transform
        self.transform_aux = transform_aux
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.images[idx])
        with Image.open(img_path) as img:
            img_clip = self.transform(img)
            img_aux = self.transform_aux(img)
        label = self.labels[idx]

        return img_clip, img_aux, label

class UnLabeledImageDataset(Dataset):
    def __init__(self, images, img_dir='/root/autodl-tmp/Dataset/'):
        super().__init__()
        self.img_dir = img_dir
        self.images = images
        self.transform_w_clip = transform_weak_clip(224)
        self.transform_w_aux = transform_weak_aux(224)
        self.transform_s_clip = transform_strong_clip(224)
        self.transform_s_aux = transform_strong_aux(224)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.images[idx])
        with Image.open(img_path) as img:
            img_clip_s = self.transform_s_clip(img)
            img_aux_s = self.transform_s_aux(img)
            img_clip_w = self.transform_w_clip(img)
            img_aux_w = self.transform_w_aux(img)
        return img_clip_s, img_aux_s, img_clip_w, img_aux_w

class TestImageDataset(Dataset):
    def __init__(self, images, img_dir='/root/autodl-tmp/Dataset/TestSetA/', transform=_transform(224), transform_aux=aux_transform):
        super().__init__()
        self.img_dir = img_dir
        self.images = images
        self.transform = transform
        self.transform_aux = transform_aux
        # train_labels = jt.concat(labels)
        # targets = train_labels.unsqueeze(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.images[idx])
        with Image.open(img_path) as img:
            img_0 = img
            img_1 = flip_image(img, mode='horizontal')
            # img_2 = rotate_image(img, 45)
            # img_3 = scale_image(img, 1.5)
            # img_5 = blur_image(img)
            # img_6 = adjust_color(img, 1.2, 1.5, 1.3)
            # img_7 = affine_transform(img, (1, 0.2, 0, 0.2, 1, 0))
            img_8 = pad_img(img)
            img_clip = []
            img_aux = []
            img_clip.append(self.transform(img_0))
            img_clip.append(self.transform(img_1))
            # img_clip.append(self.transform(img_2))
            # img_clip.append(self.transform(img_3))
            img_clip.append(self.transform(img_8))
            # img_clip.append(self.transform(img_6))
            # img_clip.append(self.transform(img_7))
            # img_clip.append(self.transform(img_8))
            img_aux.append(self.transform_aux(img_0))
            img_aux.append(self.transform_aux(img_1))
            # img_aux.append(self.transform_aux(img_2))
            # img_aux.append(self.transform_aux(img_3))
            img_aux.append(self.transform_aux(img_8))
            # img_aux.append(self.transform_aux(img_6))
            # img_aux.append(self.transform_aux(img_7))
            # img_aux.append(self.transform_aux(img_8))
            img_clip = np.array(img_clip)
            img_aux = np.array(img_aux)
            # img_clip = self.transform(img)
            # img_aux = self.transform_aux(img)
        return img_clip, img_aux, img_path

class UnLabeledImageDataset_to_pick(Dataset):
    def __init__(self, images, img_dir = '/root/autodl-tmp/Dataset/', transform=_transform(224), transform_aux=aux_transform):
        super().__init__()
        self.img_dir = img_dir
        self.images = images
        self.transform = transform
        self.transform_aux = transform_aux

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.images[idx])
        with Image.open(img_path) as img:
            img_clip = self.transform(img)
            img_aux = self.transform_aux(img)
        return img_clip, img_aux, img_path
