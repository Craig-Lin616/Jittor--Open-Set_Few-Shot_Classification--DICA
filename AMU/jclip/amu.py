import jittor as jt
from jittor import nn, init
from jittor.transform import Compose, ImageNormalize
import os.path as osp
from skimage.color import gray2rgb
from PIL import Image
import numpy as np
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize, RandomResizedCrop, \
    RandomHorizontalFlip
from jittor.dataset import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, images, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.images[idx])
        with Image.open(img_path) as img:
            if self.transform:
                img = self.transform(img)
            return img

def to_tensor(data):
    # return jt.Var(data)
    return data

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
tfm_train_base = Compose([
    RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=BICUBIC),
    RandomHorizontalFlip(p=0.5),
    ImageToTensor()
]
)

def logit_normalize(logit):
    logits_std = jt.std(logit)
    logits_mean = jt.mean(logit)
    logit = (logit - logits_mean) / logits_std
    return logit

def uncertainty(logits, type, power):
    logits = nn.softmax(logits,dim=-1)

    if type == 'entropy':
        entropy = -jt.sum(logits * jt.log2(logits), dim=-1, keepdim=True) / jt.log2(
            jt.Var(logits.shape[-1]).float())
        entropy = (entropy * power).exp()
        return entropy
    elif type == 'energy':
        max_values = logits.max(dim=-1, keepdim=True).values
        logits = logits - max_values
        tau = 2
        energy = tau * (jt.log(jt.sum(jt.exp(logits / tau), dim=-1, keepdim=True)) + max_values)
        return 1.0 / (energy ** power)
    elif type == 'max':
        max_values = logits.max(dim=-1, keepdim=True).values
        return 1.0 / (max_values) ** power
    elif type == 'max-min':
        diff = logits.max(dim=-1, keepdim=True).values - logits.min(dim=-1, keepdim=True).values
        return 1.0 / diff ** power
    elif type == 'var':
        variance = jt.std(logits, dim=-1, keepdim=True)
        return variance
    elif type == 'top5':
        top2 = logits.topk(5, dim=-1).values
        confidence = (top2[:, 0] - top2[:, -1]).unsqueeze(-1)
        return 1.0 / (confidence) ** power

    elif type == 'moment':
        mu = jt.mean(logits, dim=-1, keepdim=True)
        sigma = jt.std(logits, dim=-1, keepdim=True)
        normalized_logits = (logits - mu) / sigma
        moment_4 = jt.mean(normalized_logits ** 4, dim=-1, keepdim=True)
        return 1 / ((moment_4 / 250) ** power)
        # return 1.5 - 0.12 * moment_4
        # return filp(moment_4)
        # return (- moment_4 * power).exp()
    elif type == 'none':
        return jt.Var(1.0)
    else:
        raise RuntimeError('Invalid uncertainty type.')

class Linear_Adapter(nn.Module):
    def __init__(self, feat_dim, class_num, sample_features=None):
        super().__init__()
        self.fc = nn.Linear(feat_dim, class_num, bias=False)
        # init
        if sample_features is not None:
            print('init adapter weight by training samples...')
            aux_features, aux_labels = sample_features[0], sample_features[1]
            aux_features = aux_features

            init_weight = jt.zeros(feat_dim, class_num)
            # for i in range(len(aux_labels)):
            #     init_weight[:, aux_labels[i]] += aux_features[i]

            # 将 jittor 张量转换为 numpy 数组
            init_weight_np = init_weight.numpy()
            aux_features_np = aux_features.numpy()
            # 使用 numpy 进行累加操作
            for i in range(len(aux_labels)):
                init_weight_np[:, aux_labels[i]] += aux_features_np[i]

            # 将结果转换回 jittor 张量
            init_weight = jt.array(init_weight_np)

            feat_per_class = len(aux_labels) / class_num
            init_weight = init_weight / feat_per_class
            self.fc.weight = jt.transpose(init_weight)
        else:
            print('init adapter weight by random...')

    def execute(self, feat):
        return self.fc(feat)

class Text_Adapter(nn.Module):
    def __init__(self, feat_dim, class_num):
        super().__init__()
        self.linear_1 = nn.Linear(feat_dim, feat_dim // 4, bias=False)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(feat_dim // 4, feat_dim, bias=False)
        self.relu2 = nn.ReLU()


    def execute(self, feat):
        feat = self.relu(self.linear_1(feat))
        feat = self.relu2(self.linear_2(feat))
        return feat

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

tfm_clip = Compose([ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

def _convert_image_to_rgb(image):
    return image.convert("RGB")

tfm_aux=Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224), _convert_image_to_rgb,
        ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ImageToTensor()
])

class AMU_Model(nn.Module):
    def __init__(self, clip_model, aux_model, sample_features_clip, sample_features, clip_weights, feat_dim, class_num, lambda_merge, alpha,
                 uncent_type, uncent_power, adj_matrix_file):
        super().__init__()

        self.clip_model = clip_model
        # self.clip_model_resnet = clip_model_resnet

        #  计算模型参数量
        total_params = 0
        for name, param in self.clip_model.named_parameters():
            total_params += param.numel()
        total_params_M = total_params / (1000 * 1000)
        print("clip模型参数量为：", str(total_params_M), "M")

        self.aux_model = aux_model
        self.clip_weights = clip_weights

        self.clip_adapter = Linear_Adapter(512, class_num, sample_features=sample_features_clip)
        self.clip_adapter.eval()
        #  计算模型参数量
        total_params = 0
        for name, param in self.clip_adapter.named_parameters():
            total_params += param.numel()
        total_params_M = total_params / (1000 * 1000)
        print("clip_adapter模型参数量为：", str(total_params_M), "M")


        self.aux_adapter = Linear_Adapter(feat_dim, class_num, sample_features=sample_features)
        #  计算模型参数量
        total_params = 0
        for name, param in self.aux_adapter.named_parameters():
            total_params += param.numel()
        total_params_M = total_params / (1000 * 1000)
        print("aux_adapter模型参数量为：", str(total_params_M), "M")

        self.aux_model_2 = None
        self.aux_adapter_2 = None

        self.lambda_merge = lambda_merge
        self.uncent_type = uncent_type
        self.uncent_power = uncent_power
        self.alpha = alpha

    def execute(self, images=None, clip_features=None, aux_features=None, labels=None, images_aux=None):
        if images is not None:
            clip_features, aux_features = self.forward_feature(images,images_aux)
        clip_features /= clip_features.norm(dim=-1, keepdim=True)
        aux_features /= aux_features.norm(dim=-1, keepdim=True)
        clip_logits, aux_logits = self.forward_adapter(clip_features, aux_features)

        # fusion
        factor = uncertainty(
            clip_logits.float(),
            power=self.uncent_power,
            type=self.uncent_type
        )
        logits = clip_logits + factor * aux_logits * self.alpha

        # loss
        if labels is not None:
            loss_merge = nn.cross_entropy_loss(logits, labels)
            loss_aux = nn.cross_entropy_loss(aux_logits, labels)
            loss = self.lambda_merge * loss_merge + (1 - self.lambda_merge) * loss_aux
        else:
            loss_merge = None
            loss_aux = None
            loss = None

        return_dict = {
            "logits": logits,
            "clip_logits": clip_logits,
            "aux_logits": aux_logits,
            "loss": loss,
            "loss_merge": loss_merge,
            "loss_aux": loss_aux,
        }

        return return_dict

    def forward_feature(self, images, images_aux):
        clip_features = self.clip_model.encode_image(images)
        aux_feature = self.aux_model(images_aux)

        return clip_features, aux_feature

    def forward_adapter(self, clip_features, aux_features):
        # logits
        clip_logits = 100. * clip_features @ self.clip_weights

        aux_logits = self.aux_adapter(aux_features)
        aux_logits = logit_normalize(aux_logits)

        return clip_logits, aux_logits

    def pred(self, images,images_aux):

        if self.aux_model_2 is not None:
            clip_features = self.clip_model.encode_image(images)
            aux_features = self.aux_model(images_aux)
            aux_features_2 = self.aux_model_2(images_aux)

            clip_features /= clip_features.norm(dim=-1, keepdim=True)
            aux_features /= aux_features.norm(dim=-1, keepdim=True)
            aux_features_2 /= aux_features_2.norm(dim=-1, keepdim=True)

            clip_logits = 100. * clip_features @ self.clip_weights

            clip_adapter_logits = self.clip_adapter(clip_features)
            clip_adapter_logits = logit_normalize(clip_adapter_logits)

            aux_logits = self.aux_adapter(aux_features)
            aux_logits = logit_normalize(aux_logits)

            aux_logits_2 = self.aux_adapter_2(aux_features_2)
            aux_logits_2 = logit_normalize(aux_logits_2)

            # for novel cls
            aux_logits = jt.nn.pad(aux_logits, (0, 29))
            scores = clip_logits.numpy()
            clip_result = 0
            for prediction in scores.tolist():
                prediction = np.asarray(prediction)
                top5_idx = prediction.argsort()[-1:-6:-1]
                clip_result = top5_idx[0]

            aux_logits_2 = jt.nn.pad(aux_logits_2, (0, 29))
            clip_adapter_logits = jt.nn.pad(clip_adapter_logits, (0, 29))

            # # fusion
            # factor = uncertainty(
            #     clip_logits.float(),
            #     power=self.uncent_power,
            #     type='entropy'
            # )

            if clip_result > 373:
                logits = clip_logits
            else:
                logits = clip_logits + clip_adapter_logits + 0.7 * (aux_logits * 0.9 + aux_logits_2 * 0.1)

            # logits = clip_logits + clip_adapter_logits + 0.7 * (aux_logits * 0.8 + aux_logits_2 * 0.1 + aux_logits_3 * 0.1)

            return logits

        else:
            clip_features, aux_features = self.forward_feature(images,images_aux)
            clip_features /= clip_features.norm(dim=-1, keepdim=True)
            aux_features /= aux_features.norm(dim=-1, keepdim=True)
            clip_logits, aux_logits = self.forward_adapter(clip_features, aux_features)

            # # for novel cls
            # aux_logits = jt.nn.pad(aux_logits, (0, 29))
            # scores = clip_logits.numpy()
            # clip_result = 0
            # for prediction in scores.tolist():
            #     prediction = np.asarray(prediction)
            #     top5_idx = prediction.argsort()[-1:-6:-1]
            #     clip_result = top5_idx[0]
            # # print(clip_result)

            # fusion
            factor = uncertainty(
                clip_logits.float(),
                power=self.uncent_power,
                type=self.uncent_type
            )

            # # for novel cls
            # if clip_result > 373:
            #     logits = clip_logits
            # else:
            #     logits = clip_logits + factor * aux_logits * self.alpha

            logits = clip_logits + factor * aux_logits * self.alpha

            return logits