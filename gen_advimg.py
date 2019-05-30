import os
import PIL
from PIL import Image
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data_attack import Attack
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from scipy.misc import imread
from scipy.misc import imresize
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('logs/gen_attack/516_1.log', 'a'))
print = logger.info
#from progressbar import *
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=5,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    parser.add_argument('-gpu_ids','--gpu_ids', default=0, nargs="+",
                        help='gpu_id eg: 0,1,2 ..', type=int)
    #parser.add_argument('--momentum', default = 0.9,
    #                   help="momentum", type=float)
    parser.add_argument('--input_dir', default="/cluster/home/it_stu28/jyz/data/IJCAI/IJCAI_2019_AAAC_train_processed",
                        help="data input dir", type=str)
    parser.add_argument('--output_dir',default='/cluster/home/it_stu28/jyz/data/IJCAI/IJCAI_2019_AAAC_train_ad',type=str)
    parser.add_argument('--kernel_size', default=11, type=int)
    parser.add_argument('--max_epsilon', default=26, type=int)
    parser.add_argument('--num_iter', default=12, type=int)
    parser.add_argument('--mode',default="target",type=str)
    parser.add_argument('--prob', default=0.7, help="probality use input diversity", type=float)
    parser.add_argument('--epoch', default=40, help="epoch",type=int)
    return parser.parse_args()
 
class ImageSet_preprocess(Dataset):
    def __init__(self, df, attack):
        self.df = df
        self.attack = attack
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        label = self.df.iloc[item]['label']
        image =  Image.open(image_path).convert('RGB')

        image =  self.attack(image,label)
        image_path = image_path.replace('/IJCAI_2019_AAAC_train/', '/IJCAI_2019_AAAC_train_ad/')
        _dir, _filename = os.path.split(image_path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        image.save(image_path)
        return image_path
    
def load_data_adv(attack, batch_size=8):
    all_imgs = glob.glob('/cluster/home/it_stu28/jyz/data/IJCAI/IJCAI_2019_AAAC_train/*/*.jpg')
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]
    train_data = pd.DataFrame({'image_path':all_imgs, "label":all_labels})
    datasets = {
        'train_data': ImageSet_preprocess(train_data, attack),
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders


class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = self.transformer(Image.open(image_path))#.convert('RGB'))
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename':os.path.basename(image_path),
            'filepath':image_path
        }
        return sample

def load_data_for_train_attack(dataset_dir, img_size, batch_size=16):
    all_imgs = glob.glob(os.path.join(dataset_dir, '*/*.jpg'))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    train = pd.DataFrame({'image_path':all_imgs,'label_idx':all_labels})
    train_data, val_data = train_test_split(train,
                           stratify=train['label_idx'].values, train_size=0.95, test_size=0.05)

    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'train_data': ImageSet(train_data, transformer),
        'val_data':   ImageSet(val_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=True) for ds in datasets.keys()
    }
    return dataloaders

def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_images(images, filepaths):
    for i, filepath in enumerate(filepaths):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        img = (((images.cpu().detach().numpy()[i, :, :, :] + 1.0) * 0.5) * 255.0)
        img = img.clip(0,255).astype(np.uint8)
        dir_ = '/'.join(filepath.split('/')[:-1])
        check_mkdir(dir_)
        # resize back to [299, 299]
        r_img = imresize(img, [299, 299])
        jpg = Image.fromarray(r_img).convert('RGB')  
        jpg.save(filepath)
        #jpg.save('./test/'+filepath.split('/')[-1])


if __name__ == '__main__':
    args = parse_args()
    print(args)
    gpu_ids = args.gpu_ids
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    attack = Attack(gpu_ids, prob1=0.7,prob2=0.7, prob3=0.5, prob4=0.5)
    dataloader = load_data_for_train_attack(args.input_dir, 299, args.batch_size)
    #widgets = ['jpeg :',Percentage(), ' ', Bar('#'),' ', Timer(),
    #    ' ', ETA(), ' ', FileTransferSpeed()]
    #pbar = ProgressBar(widgets=widgets)
    device = torch.device('cuda:%d'%gpu_ids[0])    
    i, it = 0, 0
    print("Begin generate adversarial sample.")
    for e in range(args.epoch):
        for data in dataloader['train_data']:
            image = data["image"]
            label = data['label_idx']
            it+=image.shape[0]
            image = image.to(device)
            image = attack(image, label)
            i+=1
            if i%50==1:
                print("Epoch %d, [%d/%d] Done!"%(e,i,len(dataloader['train_data'])))
            image_path = [p.replace('IJCAI_2019_AAAC_train_processed', 'IJCAI_2019_AAAC_train_ad') for p in data['filepath']]
            #print(image_path)
            save_images(image, image_path)

