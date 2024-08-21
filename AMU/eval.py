import math
import os
import random

from jittor.optim import LambdaLR

import jclip as clip
from utils.utils import *
from jclip.amu import *
from jittor.transform import _setup_size
from jittor.dataset import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from dataset.dataset import *
from tqdm import tqdm
from dataset.dataset import _transform

def output_result_to_server_testB(model, test_img_dir='/root/autodl-tmp/TestSetB', save_file_txt='/root/autodl-tmp/result_search_to_server.txt'):
    test_sampler = SequentialSampler
    model.eval()
    with jt.no_grad():
        test_imgs = os.listdir(test_img_dir)
        testdataset = TestImageDataset(test_imgs, test_img_dir)
        test_dataloader = DataLoader(
            testdataset,
            sampler=test_sampler(testdataset),
            batch_size=1,
            num_workers=1,
            drop_last=False)

        print('Testing data on TestSetB is processing:')
        with open(save_file_txt, 'w') as save_file:
            for i in range(4305):
                print(i)
                imgs_clip, imgs_aux, image_path = next(iter(test_dataloader))
                imgs_aug_clip = []
                imgs_aug_aux = []

                for i in range(2):
                    sub_tensor_clip = imgs_clip[:, i, :, :, :]  # 提取第 i 个 (3, 224, 224) 层
                    sub_tensor_aux = imgs_aux[:, i, :, :, :]
                    imgs_aug_clip.append(sub_tensor_clip)
                    imgs_aug_aux.append(sub_tensor_aux)

                # 调用 pred 函数处理当前批次的图像
                with jt.no_grad():
                    predoctions = []
                    for img_clip, img_aux in zip(imgs_aug_clip, imgs_aug_aux):
                        batch_scores = model.pred(img_clip, img_aux)
                        predoctions.append(batch_scores)
                    batch_scores = jt.mean(jt.stack(predoctions), dim=0)
                scores = batch_scores.numpy()
                for prediction in scores.tolist():
                    prediction = np.asarray(prediction)
                    top5_idx = prediction.argsort()[-1:-6:-1]
                    img = image_path[0][26:]
                    save_file.write(img + ' ' +
                                    ' '.join(str(idx) for idx in top5_idx) + '\n')
