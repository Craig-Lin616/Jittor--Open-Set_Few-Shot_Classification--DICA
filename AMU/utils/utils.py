import logging
import datetime
from PIL import Image
from tqdm import tqdm
import numpy as np
import jclip as clip
import jittor as jt
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize, RandomResizedCrop, \
    RandomHorizontalFlip
import os.path as osp
from skimage.color import gray2rgb
import json
jt.flags.use_cuda = 1

def to_tensor(data):
    return jt.Var(data)

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


class ImageToTensor(object):

    def __call__(self, input):
        input = np.asarray(input)
        if len(input.shape) < 3:
            input = np.expand_dims(input, -1)
        return to_tensor(input)

try:
    from jittor.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


tfm_train_base = Compose([
    RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=BICUBIC),
    RandomHorizontalFlip(p=0.5),
    ImageToTensor()
]
)

tfm_test_base = Compose([
    Resize(224),
    CenterCrop(224),
    _convert_image_to_rgb,
    ImageToTensor(),
])

tfm_aux=Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224), _convert_image_to_rgb,
        ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ImageToTensor()
])

tfm_clip=Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224), _convert_image_to_rgb,
        ImageNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ImageToTensor()
])

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.equal(target.view(1, -1).expand(pred.shape))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def gpt_clip_classifier(classnames, clip_model, template):
    with jt.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            json_files = ['./prompt/animal_prompts.json', './prompt/cal_prompts.json',
                          './prompt/cars_prompts.json', './prompt/dog_prompt.json',
                          './prompt/food_prompts.json']
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 检查是否存在给定的classname以及对应的list
                    if classname in data:
                        texts = texts + data[classname]
                    else:
                        texts = texts
            texts = clip.tokenize(texts)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = jt.stack(clip_weights, dim=1)
    return clip_weights


imgs_dir = '/root/autodl-tmp/Dataset/'

def load_aux_weight(model, train_imgs, train_labels, tfm_norm):
    aux_features = []
    aux_labels = []
    with jt.no_grad():
        aux_features_current = []
        batch_size = 64
        for i in range(0, len(train_imgs), batch_size):
            new_image = []
            batch_imgs = train_imgs[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]
            for img in tqdm(batch_imgs):
                img = osp.join(imgs_dir, img)
                with Image.open(img) as image:
                    # if image.mode == 'L':  # 判断是否为灰度图
                    #     image = gray2rgb(image)# 如果是，就转称彩色 （通道数变成三）
                    new_image.append(image.copy())
            images = []
            for img in new_image:
                images.append(tfm_aux(img))
            del new_image
            images = jt.stack(images)
            image_features = model(images)
            aux_features_current.append(image_features)
            target = batch_labels
            aux_labels.append(target)
        aux_features.append(jt.concat(aux_features_current, dim=0).unsqueeze(0))

    aux_features = jt.concat(aux_features, dim=0).mean(dim=0)
    aux_features /= aux_features.norm(dim=-1, keepdim=True)

    aux_labels = jt.contrib.concat([jt.array(sublist) for sublist in aux_labels], dim=0)
    # aux_labels = jt.concat(aux_labels[0])

    return aux_features, aux_labels

def load_clip_weight(model, train_imgs, train_labels, tfm_norm):
    clip_features = []
    clip_labels = []
    with jt.no_grad():
        clip_features_current = []
        batch_size = 64
        for i in range(0, len(train_imgs), batch_size):
            new_image = []
            batch_imgs = train_imgs[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]
            for img in tqdm(batch_imgs):
                img = osp.join(imgs_dir, img)
                with Image.open(img) as image:
                    # if image.mode == 'L':  # 判断是否为灰度图
                    #     image = gray2rgb(image)# 如果是，就转称彩色 （通道数变成三）
                    new_image.append(image.copy())
            images = []
            for img in new_image:
                images.append(tfm_clip(img))
            del new_image
            images = jt.stack(images)
            image_features = model.encode_image(images)
            clip_features_current.append(image_features)
            target = batch_labels
            clip_labels.append(target)
        clip_features.append(jt.concat(clip_features_current, dim=0).unsqueeze(0))

    clip_features = jt.concat(clip_features, dim=0).mean(dim=0)
    clip_features /= clip_features.norm(dim=-1, keepdim=True)

    clip_labels = jt.contrib.concat([jt.array(sublist) for sublist in clip_labels], dim=0)

    return clip_features, clip_labels


def load_test_features(model, loader, tfm_norm):
    features, labels = [], []
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images, target
            if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                image_features = model.encode_image(tfm_norm(images))  # for clip model
            else:
                image_features = model(tfm_norm(images))
            features.append(image_features)
            labels.append(target)

    features, labels = jt.concat(features), jt.concat(labels)
    features = features

    return features, labels

def config_logging(args):
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M')
    now = datetime.datetime.now().strftime("%m-%d-%H_%M")
    # FileHandler
    fh = logging.FileHandler(f'result/{args.exp_name}_{now}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger