#!/bin/bash
# 设置训练集路径 (TrainSet文件夹应置于该路径下)
train_dataset_path="/root/autodl-tmp/Dataset/"
#  训练以resnext101_32x8d为AUX MODEL的模型
python train_resnext.py "$train_dataset_path"
#  训练以resnet101为AUX MODEL的模型
python train_resnet.py "$train_dataset_path"
#  从resnet的aux模型中抽取出aux_model_weight和aux_adapter_weight,便于测试时集成
python save_for_embed.py