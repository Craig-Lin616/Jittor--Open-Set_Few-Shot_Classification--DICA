import math
import os
import random

from jittor.optim import LambdaLR
from jittor import nn

import jclip as clip
from utils.utils import *
from jclip.moco import load_moco_resnext, load_moco_resnet
from jclip.amu import *
from jittor.transform import _setup_size
from jittor.dataset import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from dataset.dataset import *
from tqdm import tqdm
from dataset.dataset import _transform
from eval import output_result_to_server_testB
import jittor as jt
from jittor.models import resnet101
from jittor.models.resnet import *
import sys

def choose_sample_each_class(train_labels_txt='./data/train.txt'):

    imgs_labels = open(train_labels_txt).read().splitlines()
    train_imgs = [l.split(' ')[0] for l in imgs_labels]
    train_labels = [int(l.split(' ')[1]) for l in imgs_labels]
    train_labels_float = [jt.float32([int(l.split(' ')[1])]) for l in imgs_labels]

    # 每个类挑四张图，根据train_labels中的label来挑选
    cnt = {}

    # n way k-shot labeled sample
    new_train_imgs = []
    new_train_labels = []
    new_train_labels_float = []

    # unlabeled sample
    more_train_imgs = []
    more_train_labels = []

    for i in range(len(train_imgs)):
        label = int(train_labels[i])
        if label not in cnt:
            cnt[label] = 0
        if cnt[label] < 4:
            new_train_imgs.append(train_imgs[i])
            new_train_labels.append(train_labels[i])
            new_train_labels_float.append(train_labels_float[i])
            cnt[label] += 1
        # if cnt[label] >= 4 and label > 243:
        if cnt[label] >= 4:
            if cnt[label] > 4:
                more_train_imgs.append(train_imgs[i])
                more_train_labels.append(train_labels[i])
            cnt[label] += 1

    # 打开文本文件
    with open('./data/chosen_samples.txt', 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除行末的换行符
            line = line.strip()
            # 以空格分隔每行的图片路径和类别ID
            parts = line.split(' ')
            # 将图片路径和类别ID分别存入对应的列表中
            new_train_imgs.append(parts[0])
            new_train_labels.append(int(parts[1]))

    for class_id in range(374):
        repeat_pick_pool = [new_train_imgs[index] for index, value in enumerate(new_train_labels) if
                            value == class_id]
        if len(repeat_pick_pool) < 32:
            a = 32 - len(repeat_pick_pool)
            print("cls id:", class_id, ",a is:", a)
            for i in range(a):
                random_idx = random.randint(0, len(repeat_pick_pool) - 1)
                new_train_imgs.append(repeat_pick_pool[random_idx])
                new_train_labels.append(class_id)

    print("train imgs nums:", len(new_train_imgs),",train_imgs_labels:", len(new_train_labels))

    return train_imgs, new_train_imgs, new_train_labels, more_train_imgs, more_train_labels, new_train_labels_float

def init_model_for_AMU(new_train_imgs, new_train_labels, classes_txt='./data/classes.txt', clip_weight_path="./weight/ViT-B-32.pkl"):

    classes = open(classes_txt).read().splitlines()
    # remove the prefix Animal, Thu-dog, Caltech-101, Food-101
    new_classes = []
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
        if c.startswith('Thu-dog'):
            c = c[8:]
        if c.startswith('Caltech-101'):
            c = c[12:]
        if c.startswith('Food-101'):
            c = c[9:]
        # if c.startswith('Stanford-Cars'):
        #     c = c[14:]
        new_classes.append(c)

    # get CLIP model
    clip_model, preprocess = clip.load(clip_weight_path)
    clip_model.eval()

    # AUX MODEL
    aux_model, feat_dim = load_moco_resnet()  # Aux model path
    aux_model.eval()

    # Textual features
    template = ["a photo of {}."]
    clip_weights = gpt_clip_classifier(new_classes, clip_model, template)

    # Load visual features of few-shot training set
    aux_features, aux_labels = load_aux_weight(aux_model, new_train_imgs, new_train_labels, tfm_norm=tfm_aux)

    model = AMU_Model(
        clip_model=clip_model,
        aux_model=aux_model,
        sample_features_clip = None,
        sample_features=[aux_features, aux_labels],
        clip_weights=clip_weights,
        feat_dim=feat_dim,
        class_num=374,
        lambda_merge=0.35,
        alpha=0.5,
        uncent_type="none",
        uncent_power=0.4,
        adj_matrix_file=None
    )

    return model, preprocess, new_classes

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def train_one_iter(model, train_images, train_labels, optimizer, scheduler, imgs_aux):
    # Train
    model.aux_adapter.train()
    # model.aux_model._modules['layer4'].train()
    model.apply(freeze_bn)  # freeze BN-layer
    loss_list = []
    loss_aux_list = []
    loss_merge_list = []

    return_dict = model(train_images, labels=train_labels, images_aux=imgs_aux)
    loss = return_dict['loss']

    loss_list.append(return_dict['loss'].item())
    loss_aux_list.append(return_dict['loss_aux'].item())
    loss_merge_list.append(return_dict['loss_merge'].item())

    optimizer.zero_grad()
    optimizer.step(return_dict['loss'])
    scheduler.step()
    return loss

def train_and_eval(model, labeled_trainloader, optimizer, scheduler, stage_1_epoch):
    total_batch_len = len(labeled_trainloader) // 8
    for epoch in range(0, stage_1_epoch):  # 50
        labeled_iter = iter(labeled_trainloader)
        for i in range(total_batch_len):
            imgs, imgs_aux, labels = next(labeled_iter)
            loss = train_one_iter(model, imgs, labels, optimizer, scheduler, imgs_aux)
            print(f"Training AMU now: epoch:{epoch}, loss:{loss.item()}")



if __name__ == '__main__':

    jt.flags.use_cuda = 1
    random.seed(2)
    jt.misc.set_global_seed(1)

    train_dataset_path = sys.argv[1]

    # # choose 4 sample each class
    train_images, new_train_imgs, new_train_labels, more_train_imgs, more_train_labels, new_train_labels_float = choose_sample_each_class()

    # # build amu-model
    model, preprocess, new_classes = init_model_for_AMU(new_train_imgs, new_train_labels)

    # build optimizer
    optimizer = jt.optim.AdamW(
        model.aux_adapter.parameters(),
        weight_decay=0.01,
        lr=1e-3,
        eps=1e-4
    )

    # build sampler, dataset and dataloader
    train_sampler = RandomSampler
    labeled_dataset = LabeledImageDataset(new_train_imgs, new_train_labels, img_dir=train_dataset_path)
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=8,
        num_workers=4,
        drop_last=False)


    # build scheduler
    stage_1_epoch = 50
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, stage_1_epoch * len(train_images))

    train_and_eval(model, labeled_trainloader, optimizer, scheduler, stage_1_epoch)
    model.save('./weight/resnet.pkl')
