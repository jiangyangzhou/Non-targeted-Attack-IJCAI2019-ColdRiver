import subprocess
from pandas.io.json import json_normalize
import pandas as pd
import os
import PIL
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from models.vgg import vgg16_bn
from models.inception import inception_v3
from models.resnet import resnet50,resnet152
from models.googleNet import googlenet
from densenet import densenet121, densenet161
from models.incept_resnet_v2 import InceptionResNetV2
from models.inception_v4 import InceptionV4


from models.unet import UNet
from data_util import *

from scipy.misc import imread
from scipy.misc import imresize
import random 

import imp
from collections import defaultdict, OrderedDict
import time
import io



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=1,
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=4,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    parser.add_argument('--input_dir', default = '/home/shh/Passport/jyz/data/IJCAI/dev_data',
                        help="data input dir", type=str)
    parser.add_argument('--output_dir', default="Out",
                        help='output dir', type=str)
    parser.add_argument('--log_dir',default="./logs/test_search", type=str)
    parser.add_argument('--results_file', default='results.csv',type=str)
    parser.add_argument('--mode', default="nontarget", type=str)
    parser.add_argument('--attack_file', default='attack_tijiao.py', type=str)
    parser.add_argument('--if_attack',default=1,type=int)
    parser.add_argument('--jpeg_quality',default=70,type=float)  
    return parser.parse_args()    

args = parse_args()
print(args)

def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

check_mkdir(args.log_dir)
log_file = "%s/eval.log"%args.log_dir
err_file = "%s/eval.err"%args.log_dir
log_all_file = "%s/all.log"%args.log_dir
err_all_file = "%s/all.err"%args.log_dir



def load_model(model,pth_file,device):
    model = model.to(device)
    model.load_state_dict(torch.load(pth_file))
    return model


def get_model_dics(device, model_list= None):
    if model_list is None:
        model_list = ['densenet121', 'densenet161', 'resnet50', 'resnet152',
                      'incept_v1', 'incept_v3', 'inception_v4', 'incept_resnet_v2',
                      'incept_v4_adv2', 'incept_resnet_v2_adv2',
                      'black_densenet161','black_resnet50','black_incept_v3',
                      'old_vgg','old_res','old_incept']
    models = {}
    for model in model_list:
        if model=='densenet121':
            models['densenet121'] = densenet121(num_classes=110)
            load_model(models['densenet121'],"../pre_weights/ep_38_densenet121_val_acc_0.6527.pth",device)
        if model=='densenet161':
            models['densenet161'] = densenet161(num_classes=110)
            load_model(models['densenet161'],"../pre_weights/ep_30_densenet161_val_acc_0.6990.pth",device)
        if model=='resnet50':
            models['resnet50'] = resnet50(num_classes=110)
            load_model(models['resnet50'],"../pre_weights/ep_41_resnet50_val_acc_0.6900.pth",device)
        if model=='incept_v3':
            models['incept_v3'] = inception_v3(num_classes=110)
            load_model(models['incept_v3'],"../pre_weights/ep_36_inception_v3_val_acc_0.6668.pth",device)
        if model=='incept_v1':
            models['incept_v1'] = googlenet(num_classes=110)
            load_model(models['incept_v1'],"../pre_weights/ep_33_googlenet_val_acc_0.7091.pth",device)
    #vgg16 = vgg16_bn(num_classes=110)
    #load_model(vgg16, "./pre_weights/ep_30_vgg16_bn_val_acc_0.7282.pth",device)
        if model=='incept_resnet_v2':
            models['incept_resnet_v2'] = InceptionResNetV2(num_classes=110)  
            load_model(models['incept_resnet_v2'], "../pre_weights/ep_17_InceptionResNetV2_ori_0.8320.pth",device)

        if model=='incept_v4':
            models['incept_v4'] = InceptionV4(num_classes=110)
            load_model(models['incept_v4'],"../pre_weights/ep_17_InceptionV4_ori_0.8171.pth",device)
        if model=='incept_resnet_v2_adv':
            models['incept_resnet_v2_adv'] = InceptionResNetV2(num_classes=110)  
            load_model(models['incept_resnet_v2_adv'], "../pre_weights/ep_22_InceptionResNetV2_val_acc_0.8214.pth",device)

        if model=='incept_v4_adv':
            models['incept_v4_adv'] = InceptionV4(num_classes=110)
            load_model(models['incept_v4_adv'],"../pre_weights/ep_24_InceptionV4_val_acc_0.6765.pth",device)
        if model=='incept_resnet_v2_adv2':
            models['incept_resnet_v2_adv2'] = InceptionResNetV2(num_classes=110)  
            #load_model(models['incept_resnet_v2_adv2'], "../test_weights/ep_29_InceptionResNetV2_adv2_0.8115.pth",device)
            load_model(models['incept_resnet_v2_adv2'], "../test_weights/ep_13_InceptionResNetV2_val_acc_0.8889.pth",device)

        if model=='incept_v4_adv2':
            models['incept_v4_adv2'] = InceptionV4(num_classes=110)
#            load_model(models['incept_v4_adv2'],"../test_weights/ep_32_InceptionV4_adv2_0.7579.pth",device)
            load_model(models['incept_v4_adv2'],"../test_weights/ep_50_InceptionV4_val_acc_0.8295.pth",device)

        if model=='resnet152':
            models['resnet152'] = resnet152(num_classes=110)
            load_model(models['resnet152'],"../pre_weights/ep_14_resnet152_ori_0.6956.pth",device)
        if model=='resnet152_adv':
            models['resnet152_adv'] = resnet152(num_classes=110)
            load_model(models['resnet152_adv'],"../pre_weights/ep_29_resnet152_adv_0.6939.pth",device)
        if model=='resnet152_adv2':
            models['resnet152_adv2'] = resnet152(num_classes=110)
            load_model(models['resnet152_adv2'],"../pre_weights/ep_31_resnet152_adv2_0.6931.pth",device)



        if model=='black_resnet50':
            models['black_resnet50'] = resnet50(num_classes=110)
            load_model(models['black_resnet50'],"../test_weights/ep_0_resnet50_val_acc_0.7063.pth",device)
        if model=='black_densenet161':
            models['black_densenet161'] = densenet161(num_classes=110)
            load_model(models['black_densenet161'],"../test_weights/ep_4_densenet161_val_acc_0.6892.pth",device)
        if model=='black_incept_v3':
            models['black_incept_v3']=inception_v3(num_classes=110)
            load_model(models['black_incept_v3'],"../test_weights/ep_28_inception_v3_val_acc_0.6680.pth",device)
        if model=='old_res':
            MainModel = imp.load_source('MainModel', "./models_old/tf_to_pytorch_resnet_v1_50.py")
            models['old_res'] = torch.load('./models_old/tf_to_pytorch_resnet_v1_50.pth').to(device)
        if model=='old_vgg':
            MainModel = imp.load_source('MainModel', "./models_old/tf_to_pytorch_vgg16.py")
            models[model] = torch.load('./models_old/tf_to_pytorch_vgg16.pth').to(device)
        if model=='old_incept':
            MainModel = imp.load_source('MainModel', "./models_old/tf_to_pytorch_inception_v1.py")
            models[model]  = torch.load('./models_old/tf_to_pytorch_inception_v1.pth').to(device)
       
    return models

def load_data_for_defense(input_dir, batch_size=16):   #Only forward
    all_img_paths = glob.glob(os.path.join(input_dir, '*.png'))
    all_labels = [-1 for i in range(len(all_img_paths))]
    dev_data = pd.DataFrame({'image_path':all_img_paths, 'label_idx':all_labels})

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders


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
 
def preprocess(image,model_name="vgg16",prob=1.0):
    if "incept_v3" in model_name or model_name[:16]=='incept_resnet_v2' or model_name[:9]=='incept_v4' or model_name=='resnet_152' or model_name=="black_incept_v3":
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
        return input_diversity(image,prob,220,224)
    
    else:
        return input_diversity(image,prob,220,224)

def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def MSE(old_dir, new_dir,filename):
    img1 = imread(os.path.join(old_dir,filename)).astype(np.float)
    img2 = imread(os.path.join(new_dir,filename)).astype(np.float)
    #print(np.sum((img1-img2)**2,axis=2).shape)
    mse = np.sqrt(np.sum((img1-img2)**2,axis=2)).mean()
    return mse

def images_tojpeg(images,images_new):
    buffer = io.BytesIO()
    #print('1',images.mean())
    for i in range(images.shape[0]):
       pil_img = transforms.ToPILImage()(images[i].detach().cpu())
       pil_img.save(buffer, format='jpeg', quality=args.jpeg_quality)
       images_new[i] = transforms.ToTensor()(Image.open(buffer).convert('RGB'))



def test_target_attack(device, models_dic, old_dir, new_dir, labels_dic, mode):
    loader2 = load_data_for_defense(new_dir, args.batch_size)
    scores , accuracys= {}, {}
    per_score =0
    err = 0
    old_score,adv_score,black_score = [],[],[]
    with torch.no_grad():
        for key in models_dic.keys():
            model = models_dic[key]
            j=0
            score = 0
            correct = 0
            for data in loader2['dev_data']:
                image = data["image"].to(device)
                filenames = data["filename"]
                #images_tojpeg(image,image)
                img = preprocess(image,key)
                out = model(img)
                if 'incept_v3' in key  or "incept_v1" in key:
                    pred = out[0].max(1)[1]
                else:
                    try:
                        pred = out.max(1)[1]
                    except:
                        print("Error!!!!, key",key,img.shape,out.max(1))
                for i in range(len(pred)):
                    mse = MSE(old_dir, new_dir,filenames[i])
                    err+=mse
                    if mode=="target" and pred[i].item()!=labels_dic[filenames[i]]:
                        score+=64
                    elif mode=="nontarget" and pred[i].item()==labels_dic[filenames[i]]:
                        score+=64
                        correct+=1
                    else:
                        score+=mse
                        if mode!='nontarget':
                            correct+=1 
                j+=image.shape[0]
            print(key)
            if 'old' in key:
                old_score.append(score/j)
            if 'adv' in key:
                adv_score.append(score/j)
            else:
                black_score.append(score/j)
            scores[key] = score/j
            accuracys[key] = correct/float(j)
            per_score+=score/j
            err = err/j
            print("Test Model %s, acc is %.3f, score is %.3f."%(key, correct/float(j), score/j))
    per_score/=len(models_dic)
    old_score = np.mean(np.array(old_score))
    adv_score = np.mean(np.array(adv_score))
    black_score = np.mean(np.array(black_score))
    print("Per_score:", per_score)
    print("Per score for white model: %.3f"%old_score)
    print("score for adv:%.2f"%adv_score)
    print('score for black:%.2f'%black_score)
    print('err %.3f'%err)
    modified_score = old_score*0.4+adv_score*0.4+black_score*0.2
    print('Modified score is %.3f'%modified_score)
    return scores, accuracys, err, [old_score, adv_score, black_score, modified_score]

def try_str_to_num(str_):
    try:
        return int(str_)
    except:
        try:
            return float(str_)
        except:
            return str_

def get_labels(input_dir):
    table = pd.read_csv(input_dir+'/dev.csv')
    labels = table['trueLabel'].values
    filenames = table['filename'].values
    labels_dic = {}
    for i in range(labels.shape[0]):
        labels_dic[filenames[i]] = labels[i]
    return labels_dic

def get_targets(input_dir):
    table = pd.read_csv(input_dir+'/dev.csv')
    targets = table['targetedLabel'].values
    filenames = table['filename'].values
    targets_dic = {}
    for i in range(targets.shape[0]):
        targets_dic[filenames[i]] = targets[i]
    return targets_dic
  
def search_args(models_dic, arg, search_space, other_args={},labels_dic={}):
    device = torch.device('cuda:%d'%args.gpu_id)
    results = []
    basic_command = "python %s --gpu_id=%d --output_dir=%s "% \
                              (args.attack_file, args.gpu_id, args.output_dir)
    for a in other_args:
        basic_command+=" --%s=%s"%(a,other_args[a])
    for pa in search_space:
        command = basic_command + " --%s=%s"%(arg, pa)
        torch.cuda.empty_cache()
        for model in models_dic:
            models_dic[model].to(torch.device('cpu'))
        print(command)
        with open(log_file,'w') as f:
            with open(err_file,'w') as e:
                time1 = time.time()
                subprocess.call(command, stdout=f, stderr=e,shell=True)
                time2 = time.time()
        subprocess.call("cat %s>>%s"%(log_file,log_all_file),shell=True)
        subprocess.call("cat %s>>%s"%(err_file,err_all_file),shell=True)
        torch.cuda.empty_cache()
        for model in models_dic:
            models_dic[model].to(device)
        scores, acc, err, record_scores = test_target_attack(device, models_dic, args.input_dir, args.output_dir, labels_dic, args.mode)
        torch.cuda.empty_cache() 
        args_dic={}
        with open(log_file,'r') as f:
            args_attack = f.readline()
            args_attack = args_attack.split('(')[1].split(')')[0]
            args_attack = args_attack.split(',')
            for a in args_attack:
                args_dic[a.split('=')[0].strip("'").strip()] = try_str_to_num(a.split('=')[1].strip("'").strip())
        results.append([args_dic, scores, acc, err, record_scores, time2-time1])
    return results

def search_test(count_model,args_list, args_test):
    time_stamp = time.strftime("%a_%b_%d_%H_%M", time.localtime())
    results_file ="%s/%s_%s"%(args.log_dir,time_stamp,args.results_file)
    device = torch.device('cuda:%d'%args.gpu_id)
    models_dic = get_model_dics(device, count_model)
    args_attack={}
    best_settings={}
    if args.mode=="target":
        labels_dic = get_targets(args.input_dir)
    else:
        labels_dic = get_labels(args.input_dir)
    if args.if_attack==0:
         test_target_attack(device, models_dic, args.input_dir, args.output_dir, labels_dic, args.mode)
         return
    for arg in args_list:
        print("Search for %s,in scope:"%arg, args_test[arg])
        results = search_args(models_dic, arg, args_test[arg], args_attack, labels_dic)
        # print("results",results)
        args_attack = results[0][0]
        eval_score=np.zeros([len(results)])
        for i in range(len(args_test[arg])):
            for model in count_model:
                eval_score[i] += results[i][1][model]   # count score of results 
            eval_score[i]/=len(count_model)
        best_setting = args_test[arg][eval_score.argmin()]
        best_settings[arg]=best_setting
        if arg!="ablation":
            args_attack[arg] = best_setting
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                all_args = args_attack
                for a in sorted(args_attack.keys()):
                    if a=='input_dir':
                        continue
                    f.write('%s,'%a)
                for m in count_model:
                    f.write('%s, ,'%m)
                f.write('err, per_score, old_score, adv_score, black_score, m_score,  use_time')
                f.write('\n')
        with open(results_file, 'a') as f:
            f.write('%s\n'%arg)
            for i in range(len(results)):    
                for a in sorted(results[i][0]):
                    if a=='input_dir':
                        continue
                    f.write('%s, '%results[i][0][a])
                for m in count_model:
                    f.write('%.3f, %.3f, '%(results[i][1][m],results[i][2][m]))   # score 
                f.write("%.3f, %.3f, "%(results[i][3], eval_score[i])) #err value and eval_score	
                for j in range(4):
                    f.write("%.3f, "%results[i][4][j]) 
                f.write('%.3f\n'%(results[i][5]))
        print("Search for %s Done, Best choice is %.3f\n"%(arg,best_setting))
    return best_settings

def main_search():
    model_use = ['old_res', 'old_vgg', 'old_incept',
                 'black_densenet161',  'black_incept_v3' ,'black_resnet50',
                  'densenet161',  'incept_v3' ,'resnet50',
                 "incept_v1", "incept_resnet_v2_adv2","incept_v4_adv2",
                 "resnet152_adv2",  "incept_resnet_v2_adv", "incept_v4_adv", 
                   "resnet152_adv"]


#    args_list = ['prob','random_eps','kernel_size','decay_index','momentum','max_epsilon']    # decide the parameters and the order of searching 
    #args_list = ['ablation','decay_index','max_epsilon','momentum','prob','kernel_size','mask_size']
    args_list = ['momentum']
    search_args_dic = { 'momentum':[0.7,0.75,0.8,0.85, 0.9],
                    'kernel_size':[9,11,13],
                    'random_eps':[0.0,0.01,0.03],
                    'max_epsilon':[14,16,18,20,22,24,26],
                    'prob':[0.1,0.3,0.5,0.7],            #0.1
                    'num_iter':[5,10,12,14,16,18,20],         
                    'decay_index':[0.3,0.5,0.6,0.7,0.8],  #0.5
                    'mask_size':[0,5,10,15,20,30],         #20
                    'ablation':["", 'old_res', 'old_vgg','old_incept','black_incept_v3','black_resnet50','black_densenet161','incept_resnet_adv','incept_v4_adv2','resnet152_adv','incept_v4']
        }    # decide the scope of searching 
    with open(log_all_file,'w') as f:
        f.write('Begin search.')          #it record the log of attack
    with open(err_all_file,'w') as f:     
        f.write('Begin search.')       #it record the err of attack

    best_settings = search_test(model_use, args_list, search_args_dic)
    print("ALL SEARCHING JOB DONE!")
    print("THE BEST SETTING IS:\n", best_settings)


if __name__=="__main__":
    main_search()
