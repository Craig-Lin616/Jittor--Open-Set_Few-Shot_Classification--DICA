# Jittor--Open-Set_Few-Shot_Classification--DICA

#### 配置环境 
python 3.8.10, cuda 11.3, cudnn8.2.0, jittor1.3.8.5, g++ 9.4.0

#### 所使用的预训练模型与参数大小

1.OPENAI官方预训练的ViT-B/32版本的CLIP模型：
https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

2.Jittor官方预训练的Resnet101模型：
jittorhub://resnet101.pkl

3.Jittor官方预训练的Resnext101_32x8d模型：
jittorhub://resnext101_32x8d.pkl

测试时所使用的所有模型参数量大小为286M

#### 开源代码链接

https://github.com/Craig-Lin616/Jittor--Open-Set_Few-Shot_Classification--DICA/tree/main

#### 方法思路
**步骤1 -- 以半监督方式扩增训练数据集**

**步骤1.1**：顺序选择每类的前4个样本，共1496个样本作为labeled训练集，剩余的样本作为unlabeled训练集。使用labeled数据集进行初始模型（详见步骤3）的训练，经过50轮训练后，使用训练完成的模型对于unlabeled训练集中所有图片进行预测，根据每张图片的top1预测作为预测类别，在各类中选择置信度最高的4个样本，然后进行阈值判断，若大于预定义的阈值0.8，则将该样本加入labeled训练集，同时从unlabeled的训练集中删去，否则从该类已被选择的样本中随机抽取一个加入训练集（维持各类样本数平衡）

**步骤1.2**：使用步骤1.1得到的新训练集重新训练模型，训练50轮，完成训练后重复步骤1.1的操作，直至各类样本数为32个，此步骤选择的样本存放于data/chosen_samples.txt中。

**步骤2 -- 使用新的类别描述**

对于数据集中所有的400个类别，我们使用大语言模型GPT3.5生成各类别的多条text description，提问内容包括但不限于what does xxx(类别名) look like?
然后我们将各类别的所有text description编码成text embedding，并使用每个类别的平均embedding作为该类的text embedding用于CLIP模型分类


**步骤3 -- 通过AMU-tuning的方式训练Aux-Adapter**

文章链接：https://arxiv.org/pdf/2404.08958.pdf

使用train.sh完成训练，其包括了两项训练任务，分别是以resnext101_32x8d为aux_model的aux_adapter的训练，以及以resnet101为aux_model的aux_adapter的训练，训练得到的两个aux_model以及对应的aux_adapter将在测试时进行集成。

**步骤4 -- 测试时，添加Training-free的CLIP-Adapter**

Training-free的CLIP_Adapter为一个（512，374）的线性分类器，其每个类别512维的权重向量，由该类32张图片视觉向量的均值决定。

测试时，对于任意一张图片，其logit将由freeze的CLIP模型、步骤2训练得到的两个AUX模型（包含AUX_Adapter），以及该步骤中创建的CLIP_Adapter共同按照设置的权重相加得到，其权重为：

final_logit = clip_logit + clip_adapter_logit + 0.7 \* (0.9 \* resnext_aux_model_logit + 0.1 \* resnet_aux_model_logit)

对于B榜中的新类别预测，如果某张图像被Freeze CLIP预测为某个novel class，则不进行上述的final_logit加权操作，直接使用clip_logit作为该图片的logit。

**步骤5 -- 测试时图片增强**

对于测试时的每一张图片采用随机水平翻转和随机填充的方式进行测试时图片增强
