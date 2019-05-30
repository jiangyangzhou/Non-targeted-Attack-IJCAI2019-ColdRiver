import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
from models.densenet import densenet121, densenet161
from data_util import *
import models_old.tf_to_pytorch_inception_v1 as inception
import models_old.tf_to_pytorch_resnet_v1_50 as resnet
import models_old.tf_to_pytorch_vgg16 as vgg
from models.vgg import vgg16_bn
from models.inception import inception_v3
from models.resnet import resnet50, resnet152
from models.googleNet import googlenet
from models.incept_resnet_v2 import InceptionResNetV2
from models.inception_v4 import InceptionV4

from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
import argparse
import imp
import random
import time
import json
import io
from PIL import Image
from torchvision import transforms
#os.environ("CUDA_VISILE_DEVICES")=0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    parser.add_argument('--gpu_id', default=0,
                        help='gpu_id eg: 0,1,2 ..', type=int)
    parser.add_argument('--momentum', default = 0.75,
                       help="momentum", type=float)
    parser.add_argument('--input_dir', default="/home/shh/Passport/jyz/data/IJCAI/dev_data",
                        help="data input dir", type=str)
    parser.add_argument('--output_dir',default='./out2',type=str)
    parser.add_argument('--ablation', default='',
                         help="one model, excluded", type=str)
    parser.add_argument('--kernel_size', default=11, type=int)
    parser.add_argument('--max_epsilon', default=14, type=int)
    parser.add_argument('--num_iter', default=14, type=int)
    parser.add_argument('--mode',default="nontarget",type=str)
    parser.add_argument('--prob', default="0.1", help="probality use input diversity", type=float)
    parser.add_argument('--random_eps', default=0.0,
                   help='before gradient descent', type=float)
    parser.add_argument('--use_cam', default=1, help="use class activation map as mask "type=int)
    parser.add_argument('--quantize', default=0, type=int)
    parser.add_argument('--decay_index', default=0.5, help="decay the weights of successfully attacked model", type=float)
    parser.add_argument('--mask_size',default=20, type=int)
    parser.add_argument('--jpeg_quality', default=70, help="control the jpeg quality of attack image ", type=int)
    parser.add_argument('--models_weight',default='',type=str)
    return parser.parse_args()


def load_model(model,pth_file,device):
    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    #print('loading weights from : ', pth_file)
    model.load_state_dict(torch.load(pth_file))
    return model

def get_model_dics(device, model_list= None):
    if model_list is None:
        model_list = ['densenet121', 'densenet161', 'resnet50', 'resnet152',
                      'incept_v1', 'incept_v3', 'incept_v4_adv', 'incept_resnet_v2_adv',
                      'black_densenet161', 'black_resnet50', 'black_incept_v3',
                      'old_vgg','old_res','old_incept']
    models = {}
    for model in model_list:
        if model=='densenet121':
            models['densenet121'] = densenet121(num_classes=110)
            load_model(models['densenet121'],"./pre_weights/ep_38_densenet121_val_acc_0.6527.pth",device)
        if model=='densenet161':
            models['densenet161'] = densenet161(num_classes=110)
            load_model(models['densenet161'],"./pre_weights/ep_30_densenet161_val_acc_0.6990.pth",device)
        if model=='resnet50':
            models['resnet50'] = resnet50(num_classes=110)
            load_model(models['resnet50'],"./pre_weights/ep_41_resnet50_val_acc_0.6900.pth",device)
        if model=='incept_v3':
            models['incept_v3'] = inception_v3(num_classes=110)
            load_model(models['incept_v3'],"./pre_weights/ep_36_inception_v3_val_acc_0.6668.pth",device)
        if model=='incept_v1':
            models['incept_v1'] = googlenet(num_classes=110)
            load_model(models['incept_v1'],"./pre_weights/ep_33_googlenet_val_acc_0.7091.pth",device)
    #vgg16 = vgg16_bn(num_classes=110)
    #load_model(vgg16, "./pre_weights/ep_30_vgg16_bn_val_acc_0.7282.pth",device)
        if model=='incept_resnet_v2':
            models['incept_resnet_v2'] = InceptionResNetV2(num_classes=110)  
            load_model(models['incept_resnet_v2'], "./pre_weights/ep_17_InceptionResNetV2_ori_0.8320.pth",device)

        if model=='incept_v4':
            models['incept_v4'] = InceptionV4(num_classes=110)
            load_model(models['incept_v4'],"./pre_weights/ep_17_InceptionV4_ori_0.8171.pth",device)

        if model=='incept_resnet_v2_adv':
            models['incept_resnet_v2_adv'] = InceptionResNetV2(num_classes=110)  
            load_model(models['incept_resnet_v2_adv'], "./pre_weights/ep_22_InceptionResNetV2_val_acc_0.8214.pth",device)

        if model=='incept_v4_adv':
            models['incept_v4_adv'] = InceptionV4(num_classes=110)
            load_model(models['incept_v4_adv'],"./pre_weights/ep_24_InceptionV4_val_acc_0.6765.pth",device)

        if model=='incept_resnet_v2_adv2':
            models['incept_resnet_v2_adv2'] = InceptionResNetV2(num_classes=110)  
            load_model(models['incept_resnet_v2_adv2'],"./pre_weights/ep_29_InceptionResNetV2_adv2_0.8115.pth",device)
            #load_model(models['incept_resnet_v2_adv2'],"../pre_weights/ep_13_InceptionResNetV2_val_acc_0.8889.pth",device)

        if model=='incept_v4_adv2':
            models['incept_v4_adv2'] = InceptionV4(num_classes=110)
            load_model(models['incept_v4_adv2'],"./pre_weights/ep_32_InceptionV4_adv2_0.7579.pth",device)

        if model=='resnet152':
            models['resnet152'] = resnet152(num_classes=110)
            load_model(models['resnet152'],"./pre_weights/ep_14_resnet152_ori_0.6956.pth",device)
        if model=='resnet152_adv':
            models['resnet152_adv'] = resnet152(num_classes=110)
            load_model(models['resnet152_adv'],"./pre_weights/ep_29_resnet152_adv_0.6939.pth",device)
        if model=='black_resnet50':
            models['black_resnet50'] = resnet50(num_classes=110)
            load_model(models['black_resnet50'],"./test_weights/ep_0_resnet50_val_acc_0.7063.pth",device)
        if model=='black_densenet161':
            models['black_densenet161'] = densenet161(num_classes=110)
            load_model(models['black_densenet161'],"./test_weights/ep_4_densenet161_val_acc_0.6892.pth",device)
        if model=='black_incept_v3':
            models['black_incept_v3']=inception_v3(num_classes=110)
            load_model(models['black_incept_v3'],"./test_weights/ep_28_inception_v3_val_acc_0.6680.pth",device)
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

class EnsembleNet(nn.Module):
    def __init__(self, models, models_weight, device, ablation=""):
        super(EnsembleNet, self).__init__()
        #self.models = get_model_dic(device)
        self.models = models
        self.models_weight = models_weight
        self.preprocess = preprocess
        self.ablation = ablation
        self.img_size=[1,3,229,229]
        self.preds = {}
        self.logits= {}
        self.use_model ={k:True for k in self.models.keys()}
        self.model_index = {k:1.0 for k in self.models.keys()}
        self.model_bindex = {}

    def forward(self,x, get_feature=False, model_bindex=None):
        if model_bindex is None or model_bindex=={}:
            model_bindex = {k:torch.ones([x.shape[0],1]).to(device)for k in self.models.keys()}
        self.img_size = x.shape
        i=0
        weights=0
        models_weight = self.models_weight
        for model in self.models.keys():
            if self.use_model[model]==False:
                continue
            pre_x = self.preprocess(x,model)
            out = self.models[model](pre_x,get_feature=get_feature)
            if 'incept_v3' in model:
                logits = 0.5*out[0]+0.5*out[1]
            elif model=='incept_v1':
                logits = 0.4*out[0]+0.3*out[1] + 0.3*out[2]
            elif mode=='target' and model=='old_vgg':
                logits = 1.0*out
            else:
                logits = out
            self.preds[model]=logits.max(1)[1]
            self.logits[model]=logits
            if i==0:
                out_sum = logits*model_bindex[model]*models_weight[model]
            else:
                out_sum =  out_sum + logits*model_bindex[model]*models_weight[model]
            i=i+1
            weights+=models_weight[model]
        out_sum = out_sum/weights  
        return out_sum

    def get_cam(self,label):
        i=0
        for model in self.models.keys():
            if model==self.ablation:
                continue
            if model == "vgg16" or model == "old_vgg":
                continue
            params = list(self.models[model].parameters())
            features = self.models[model].feature
            weights = params[-2].data   #shape: [110,channel]
            cam = torch.zeros(features.shape[0],1,features.shape[2],features.shape[3])
            for b in range(label.shape[0]):
                weight = weights[label[b]].reshape(1,-1,1,1)
                cam[b] = F.conv2d(features[b].unsqueeze(0),weight,padding=0)
            cam = F.upsample(cam,size=self.img_size[-2:],mode='bilinear')
            #print(model, cam.mean().item(),cam.min().item(),cam.max().item())
            ens_cam = cam if i==0 else ens_cam+cam
            i+=1
        ens_cam = ens_cam/i
        ens_cam = (ens_cam-ens_cam.min())/ens_cam.max()
        #print(ens_cam.shape)
        return ens_cam



class RCE(nn.Module):
    """
    reverse cross entropy
    """
    def forward(self, x, target):
        batch_size = x.size(0)
        num_class = x.size(1)
        mask = x.new_ones(x.size()) / (num_class-1)
        mask.scatter_(1, target[:, None], 0)

        x = F.softmax(x, dim=1)
        x = -1*torch.log(x)
        loss = torch.sum(x*mask) / batch_size
        return loss




def load_data_for_defense(input_dir,batch_size=16):   #Only forward
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
                       shuffle=True) for ds in datasets.keys()
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

def preprocess(image,model_name):
    if "incept_v3" in model_name or model_name[:16]=="incept_resnet_v2" \
        or 'incept_v4' in model_name or 'resnet152' in model_name: 
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

def eval_img(image, models, models_name, old_img, init_y):
    correct = 0
    ans = 0
    j=0
    for k in models:
        if k not in models_name:
            continue
        j+=1
        model = models[k]
        img = preprocess(image,k)
        out = model(img)
        score = F.softmax(out)
        score_sum=score if j==1 else score_sum+score
        pre = score.max(1)
        pred_y =pre[1]
        c=(pred_y==init_y).sum()
        correct+=c
        rate = float(c)/(img.shape[0])
        print(k,"correct rate is %.4f"%rate)
        #print("eval:",model_name[i], pre_y)
        for i in range(image.shape[0]):
            if pred_y[i] != init_y[i]:
                mse =torch.sqrt(torch.mean(((image[i]-old_img[i])*128)**2))
            else:
                mse = 128
            ans+=mse
    total_rate = float(correct)/(image.shape[0]*j)
    print("correct rate is %.4f"%total_rate)
    print("MSE of img is", ans/(image.shape[0]*j))
    res =  ans/float(j)
    return rate,res

'''def get_init_y(image,models, models_name):
    i=0
    for k in models.keys():
        if k not in models_name:
            continue
        i+=1
        model = models[k]
        img = preprocess(image,k)
        out = model(img)
        try:
            score = F.softmax(out) if i==1 else score+F.softmax(out)
        except:
            print(k)
    return score.max(1)[1]'''

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
    labels = table['targetedLabel'].values
    filenames = table['filename'].values
    labels_dic = {}
    for i in range(labels.shape[0]):
        labels_dic[filenames[i]] = labels[i]
    return labels_dic


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        img = (((images.cpu().detach().numpy()[i, :, :, :] + 1.0) * 0.5) * 255.0)
        #img = img.clip(0,255).astype(np.uint8)
        # resize back to [299, 299]
        r_img = imresize(img, [299, 299])
        r_img = np.around(r_img).clip(0,255).astype(np.uint8)
        png = Image.fromarray(r_img)
        png.save(os.path.join(output_dir, filename), format='PNG')

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

def images_tojpeg(images,images_new):   
    buffer = io.BytesIO() 
    #print('1',images.mean())
    for i in range(images.shape[0]):     
        pil_img = transforms.ToPILImage()(images[i].detach().cpu())          
        pil_img.save(buffer, format='jpeg', quality=args.jpeg_quality)
        images_new.data[i] = transforms.ToTensor()(Image.open(buffer).convert('RGB'))
        #images.data = (images.data-0.5)*2
    #print(images.mean(),images.max())

if __name__=='__main__':
    args = parse_args()
    print(args)
    time1 = time.time()
    batch_size = args.batch_size
    input_dir = args.input_dir
    out_dir = args.output_dir
    momentum = args.momentum
    kernel_size = args.kernel_size
    mode=args.mode
    random_eps=args.random_eps
    test_loader = load_data_for_defense(input_dir, batch_size)
    device= torch.device('cuda:%d'%args.gpu_id)
    max_epsilon =args.max_epsilon
    num_iter = args.num_iter

    prob = args.prob
    eps = 2.0 * max_epsilon / 255.0
    alpha = eps / num_iter
    labels_dic = get_labels(input_dir)
    targets_dic = get_targets(input_dir)

    if args.models_weight!='':
        models_weight = json.load(args.models_weight)
    else:
        if mode=='nontarget':          
            models_weight = {
                         #'incept_v4':1.0,
                         #'incept_resnet_v2':1.0,
                         'resnet152':1.0,   
                         #'resnet152_adv':1.0,
                         #'incept_v4_adv':1.0,
                         #'incept_resnet_v2_adv':1.0,
                         'incept_v4_adv2':1.0,
                         'incept_resnet_v2_adv2':1.0,
                         #'densenet161':1.0,
                         'black_densenet161':1.0,
                         'black_resnet50':1.0,
                         'black_incept_v3':1.0,
                         'old_vgg':1.0,
                         'old_incept':1.0,
                         'old_res':1.0
                      }
        else:
            models_weight = {
                           'black_densenet161':0.5, 
                           'black_resnet50':0.5,
                           'black_incept_v3':0.5,
                           'old_vgg':1.0, 
                           'old_incept':0.5,
                           'old_res':1.0   
                          }
    if args.ablation!='':
         models_weight[args.ablation]=0.0
    model_list = [m for m in models_weight.keys()]
    models = get_model_dics(device, model_list= model_list)
    ens_model = EnsembleNet( models, models_weight, device, args.ablation)

    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    #stack_kernel = np.load('kernels/ijcai/kernels_70.npy')
    stack_kernel = np.expand_dims(stack_kernel, 1)
    print(stack_kernel.shape)
    stack_kernel = torch.Tensor(stack_kernel).to(device)

    mask = torch.zeros([1, 3, 299, 299])
    mask_size = args.mask_size
    mask[:,:,mask_size:299-mask_size,mask_size:299-mask_size]=1
    mask=mask.to(device)
    it= 0
    use_models = [m for m in ens_model.models]
    print("use_models:",use_models)
    for data in test_loader['dev_data']:
        image = data["image"]
        sh=image.shape
        it+=image.shape[0]
        filenames = data["filename"]
        image = image.to(device)
        if mode=='target':
            label=torch.Tensor([targets_dic[f] for f in filenames]).long().to(device)
        else:
            label = torch.Tensor([labels_dic[f] for f in filenames]).long().to(device)
  #      old_img = image.clone()
        image.requires_grad = True
        image_new = data["image"].to(device)
        image_new.requires_grad = True
        #print(image.mean())
        adv_ok = torch.zeros(image.shape[0])
        model_bindex={}
        momentum_sum=0
        grad_decay_index=1.0
        mu_index=torch.ones([image.shape[0],1,1,1]).to(device)
        index=1.0
        for iter in range(num_iter):
            ens_model.zero_grad()
            tur = torch.Tensor(np.abs(np.random.normal(0, random_eps, sh))).type_as(image).to(device)   #tur seemly make no sense as well 
            images_tojpeg(image, image_new) 
            image_new.data = image.data + tur.data
            out = ens_model(image_new,True, model_bindex)
            if iter==0 and args.use_cam==1:
                with torch.no_grad():
                    cam = ens_model.get_cam(label).to(device)
            loss = nn.CrossEntropyLoss()(out, label)
            #logit1 = F.softmax(out)
            #loss = nn.NLLLoss()(1-logit1,label)
            if mode=='nontarget':
                loss -= RCE()(out, label)
            #loss += nn.CrossEntropyLoss()(out, label)
            loss.backward()
            #data_grad = image.grad.data
            data_grad = image_new.grad.data
            data_grad = F.conv2d(data_grad, stack_kernel, padding=(kernel_size-1)//2, groups=3)
            for i in range(data_grad.shape[0]):
                data_grad[i] = data_grad[i]/torch.mean(data_grad[i].norm(2,0)/1.713)
            if iter==0:
                noise = data_grad
            else:
                noise = noise*momentum + data_grad
            norm = noise.norm(dim=1).unsqueeze(1)
            index = norm.mean()
            momentum_sum = momentum_sum*momentum + 1.0
            d_img = noise*norm*alpha/((momentum_sum)*index)
            #d_img[d_img<1/(255.0*num_iter*2.0)]=0
            #print(d_img.mean().item())
            d_img= d_img*mask#*grad_decay_index
            if args.use_cam:
                d_img = d_img * cam
            d_img = d_img/d_img.norm(dim=1).mean()*alpha
            if mode=='target':
                image_adv = image.data - d_img*mu_index
            else:
                image_adv = image.data + d_img*mu_index

            image_adv = torch.clamp(image_adv,-1.0,1.0)
            if args.quantize:
                image_adv.mul_(255).round_().div_(255)   # quanlity seemingly make no sense, even worse
            image.data = image_adv
            image_new.grad.zero_()
            ens_model.zero_grad()
            #image.grad.zero_()
            logits_label=torch.zeros([len(use_models),image.shape[0]])
            i=0
            with torch.no_grad():
                for model in use_models:
                    adv_ok = (ens_model.preds[model]!=label).unsqueeze(1).float().to(device) if mode=='nontarget' else (ens_model.preds[model]==label).unsqueeze(1).float().to(device)
                    logit_label = torch.gather(F.softmax(ens_model.logits[model]),1,label.view([image.shape[0],1]))
              #      print(logit_label.shape,label.shape)
                    logits_label[i]=logit_label.view(-1)
                    i+=1
                    base=1.0#+float(np.clip((iter-5)*0.1,0,10)*0.1)
                    model_bindex[model] = base - args.decay_index*adv_ok
     ################################################################################ 
                                                              #I am not sure if grad_decay_index works
            #grad_decay_index = logits_label.max(dim=0)[0]
            #print(grad_decay_index, logits_label.max(dim=0)[1])
            #grad_decay_index = grad_decay_index.clamp(0.2,1.0).view(-1,1,1,1).to(device)
            #print(grad_decay_index.shape)
            #if iter>10:
            #    mu_index[grad_decay_index>0.3] = mu_index[grad_decay_index>0.3]*1.1
            #elif iter>10:
            #    mu_index[grad_decay_index<0.07] = mu_index[grad_decay_index<0.07]*0.9


        #res = eval_img(image.to(device_val), models, val_models_name, old_img, init_y)
        #ans+=res[1]
        save_images(image, data["filename"], args.output_dir)
        #print(image.mean())
        print("Processing [%d/%d].."%(it,len(labels_dic)))
    time2 = time.time()
    print("Finish! Total time is %.2f s."%(time2-time1))
    #print("final result:", float(ans)/num_img)
