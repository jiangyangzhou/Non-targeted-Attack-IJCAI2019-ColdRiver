
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
import numpy as np
import types
from numpy import random
from models.vgg import vgg16_bn
from models.inception import inception_v3
from models.resnet import resnet50
from models.googleNet import googlenet
from models.densenet import densenet121, densenet161    
from models.incept_resnet_v2 import InceptionResNetV2
from models.inception_v4 import InceptionV4

import imp
import glob
import os
import PIL
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as multiprocessing 
#multiprocessing.set_start_method('spawn')

def load_model(model,pth_file, device):
    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    print('loading weights from : ', pth_file)
    model.load_state_dict(torch.load(pth_file))
    return model

def get_model_dic(device):
    models = {}
    #densenet_121 = densenet121(num_classes=110)
    #load_model(densenet_121,"./pre_weights/ep_38_densenet121_val_acc_0.6527.pth",device)

    densenet_161 = densenet161(num_classes=110)
    load_model(densenet_161,"./pre_weights/ep_30_densenet161_val_acc_0.6990.pth",device)

    resnet_50 = resnet50(num_classes=110)
    load_model(resnet_50,"./pre_weights/ep_41_resnet50_val_acc_0.6900.pth",device)

    incept_v3 = inception_v3(num_classes=110)
    load_model(incept_v3,"./pre_weights/ep_36_inception_v3_val_acc_0.6668.pth",device)
    
    #incept_v1 = googlenet(num_classes=110)
    #load_model(incept_v1,"./pre_weights/ep_33_googlenet_val_acc_0.7091.pth",device)
    
    #vgg16 = vgg16_bn(num_classes=110)
    #load_model(vgg16, "./pre_weights/ep_30_vgg16_bn_val_acc_0.7282.pth",device)
    incept_resnet_v2_adv = InceptionResNetV2(num_classes=110)
    load_model(incept_resnet_v2_adv, "./pre_weights/ep_22_InceptionResNetV2_val_acc_0.8214.pth",device)

    incept_v4_adv = InceptionV4(num_classes=110) 
    load_model(incept_v4_adv,"./pre_weights/ep_37_InceptionV4_val_acc_0.7119.pth",device)

    MainModel = imp.load_source('MainModel', "./models_old/tf_to_pytorch_resnet_v1_50.py")
    resnet_model = torch.load('./models_old/tf_to_pytorch_resnet_v1_50.pth').to(device)

    MainModel = imp.load_source('MainModel', "./models_old/tf_to_pytorch_vgg16.py")
    vgg_model = torch.load('./models_old/tf_to_pytorch_vgg16.pth').to(device)

    MainModel = imp.load_source('MainModel', "./models_old/tf_to_pytorch_inception_v1.py")
    inception_model = torch.load('./models_old/tf_to_pytorch_inception_v1.pth').to(device)
    
    models={#"densenet121":densenet_121,
         "densenet161":densenet_161, 
        "resnet_50":resnet_50, 
  #      "incept_v1":incept_v1, 
        "incept_v3":incept_v3,
        "incept_resnet_v2_adv": incept_resnet_v2_adv,
        "incept_v4_adv": incept_v4_adv,
        #"vgg16":vgg16
        "old_incept":inception_model,
        "old_res":resnet_model,
        "old_vgg":vgg_model
        }
       
    return models

def input_diversity(image, prob, low, high): 
    if random.random()<prob:   
        return image
    rnd = random.randint(low, high)
    rescaled = F.upsample(image, size=[rnd, rnd], mode='bilinear')
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = random.randint( 0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left 
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)         
    return padded   

def preprocess(image,model_name, prob):
    if model_name=="incept_v3" or 'incept_v4'in model_name or 'incept_resnet_v2' in model_name:
        return input_diversity(image,prob,270,299)
    else:
        image = F.upsample(image, size=(224, 224), mode='bilinear')
    if model_name=="old_res" or model_name=="old_vgg":
        image = ((image/2.0)+0.5)*255.0
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        image[:, 0,:, :] = image[:, 0,:, :] - _R_MEAN
        image[:, 1,:, :] = image[:, 1,:, :] - _G_MEAN
        image[:, 2,:, :] = image[:, 2,:, :] - _B_MEAN
        return input_diversity(image,prob,200,224)
    else:
        return input_diversity(image,prob,200,224)

class EnsembleNet(nn.Module):
    def __init__(self,device,ablation='',prob=0.5):
        super(EnsembleNet, self).__init__()
        self.models = get_model_dic(device)
        self.preprocess = preprocess
        self.ablation = ablation
        self.prob=prob
        self.models_list = []
    def forward(self,x):
        i=0
        for model in self.models.keys():
            if model==self.ablation:
                continue
            if random.random()<self.prob:
                continue
            self.models_list.append(model)
            pre_x = self.preprocess(x,model, 0.3)
            if model=='incept_v3':
                out = 0.5*self.models[model](pre_x)[0]+0.5*self.models[model](pre_x)[1]
            elif model=='incept_v1':
                out = 0.4*self.models[model](pre_x)[0]+0.4*self.models[model](pre_x)[1] + \
                      0.4*self.models[model](pre_x)[2]
            else:
                out = self.models[model](pre_x)
            out_sum = out if i==0 else out_sum + out
            i=i+1
        if i==0:
            model = random.choice(list(self.models.keys()))
            pre_x = self.preprocess(x, model, 0.3)
            out_sum = self.models[model](pre_x)
            out_sum=sum(out_sum)/len(out_sum) if model=="incept_v1" or model=="incept_v3" else out_sum
        else:
            out_sum = out_sum/i
        return out_sum
        
def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

def get_kernel(kernel_size):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.Tensor(stack_kernel)
    return stack_kernel


class Attack(object):
    def __init__(self, gpu_ids, prob1=0.7,prob2=0.7, prob3=0.5, prob4=0.5):
        self.prob1=prob1
        self.prob3=prob3
        self.prob4=prob4
        print(gpu_ids)
        if len(gpu_ids)==1:
            self.device=torch.device('cuda:%d'%gpu_ids[0])
            self.ens_model = EnsembleNet(self.device)
        else:
            self.device=torch.device('cuda:%d'%gpu_ids[0]) 
            self.ens_model =  EnsembleNet(self.device)     
            self.ens_model = torch.nn.DataParallel(self.ens_model, device_ids=gpu_ids, output_device=gpu_ids[0])
        self.kernels = {9: get_kernel(9), 11: get_kernel(11), 13: get_kernel(13), 15: get_kernel(15), 17: get_kernel(17)}
        self.kernel_size=[9,11,13,15,17]

    def __call__(self,image, label):
        if random.random() > self.prob1:
            return image
        else:
            max_epsilon = random.randint(5,30)
            eps = 2.0 * max_epsilon / 255.0
            num_iter = 1 if random.random()<self.prob3 else random.randint(2,10)
            alpha = eps / num_iter
            momentum = 0.8+0.2*random.random()
            image.requires_grad = True
            image = image.to(self.device)
            label = label.to(self.device)
            for iter in range(num_iter):
                self.ens_model.zero_grad()   
                out = self.ens_model(image)
                loss = nn.CrossEntropyLoss()(out, label)
                loss.backward()
                data_grad = image.grad.data
                if random.random()<self.prob4:
                    kernel_size = self.kernel_size[random.randint(len(self.kernels))]
                    stack_kernel = self.kernels[kernel_size].to(self.device)
                    data_grad = F.conv2d(data_grad, stack_kernel, padding=(kernel_size-1)//2)
                for i in range(data_grad.shape[0]):
                    data_grad[i] = data_grad[i]/torch.mean(data_grad[i].abs())
                if iter==0:
                    noise = data_grad
                else:
                    noise = noise*momentum + data_grad
                if random.random()<0.5:
                    image_adv = image.data + noise*alpha/(iter+1)
                else:
                    image_adv = image.data + noise.sign()*alpha
                image_adv = torch.clamp(image_adv,-1.0,1.0)
                image.data = image_adv
                image.grad.zero_()
            return image.cpu()



class ImageAugmentation(object):
    def __init__(self, device, size=224):
        self.size = size
        self.ens_model = EnsembleNet(device)
        self.transformer_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size, (0.7, 1), interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
        ])

    def __call__(self, img):
        return self.transformer_train(img)

