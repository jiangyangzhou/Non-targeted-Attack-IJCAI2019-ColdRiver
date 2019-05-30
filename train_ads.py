import os
import PIL
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from progressbar import *
from models.densenet import densenet121, densenet161
from models.inception import inception_v3
from models.vgg import vgg16_bn
from models.incept_resnet_v2 import InceptionResNetV2
from models.resnet import resnet50,  resnet152
from models.inception_v4 import InceptionV4
import logging
from numpy import  random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model', default='InceptionResNetV2',
                        help='cnn model, e.g. , densenet121, densenet161, inception_v3, vgg16bn', type=str)
    parser.add_argument('--gpu_id', default=0, nargs='+',
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=16,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    parser.add_argument('--resume', default='', 
                       help='path of pretrain weight',type=str)
    parser.add_argument('--begin_e', default=0,
                        help="begin epoch",type=int)                    
    parser.add_argument('--n_ep', default=100, 
                       help="num of epoch", type=int) 
    parser.add_argument('--lr', default=0.0001, 
                      help="learning rate",  type=float)
    return parser.parse_args()

args = parse_args()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler("logs/train_adv_result/%s_7.log"%args.target_model, 'a'))
print = logger.info

model_class_map = {
    'densenet121': densenet121,
    'densenet161': densenet161,
    'inception_v3': inception_v3,
    'vgg16_bn':vgg16_bn,
    'InceptionResNetV2':InceptionResNetV2,
    'resnet152':resnet152,
    'InceptionV4': InceptionV4,
}

class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        try:
            image_path = self.df.iloc[item]['image_path']
            image = self.transformer(Image.open(image_path))#.convert('RGB'))
        except OSError as e:
            print(e)
            image_path = self.df.iloc[random.randint(10000)]['image_path'] 
            image = self.transformer(Image.open(image_path))
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename':os.path.basename(image_path)
        }
        return sample

def load_data_for_training_cnn(dataset_dir, img_size, batch_size=16):

    all_imgs = glob.glob(os.path.join(dataset_dir, './*/*.jpg'))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    train = pd.DataFrame({'image_path':all_imgs,'label_idx':all_labels})
    train_data, val_data = train_test_split(train,
                            stratify=train['label_idx'].values, train_size=0.95, test_size=0.05)
    transformer_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, (0.7, 1), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'train_data': ImageSet(train_data, transformer_train),
        'val_data':   ImageSet(val_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=True) for ds in datasets.keys()
    }
    return dataloaders

def load_data_for_defense(input_dir, img_size, batch_size=16):

    all_img_paths = glob.glob(os.path.join(input_dir, '*.png'))
    all_labels = [-1 for i in range(len(all_img_paths))]
    dev_data = pd.DataFrame({'image_path':all_img_paths, 'label_idx':all_labels})

    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
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

def do_train(model_name, model, train_loader, val_loader, device, lr=0.0001, begin_e=0, n_ep=40, num_classes=110, save_path='/tmp'):
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=3)
    best_acc = 0.0
    # do training
    for i_ep in range(begin_e, n_ep+begin_e//2):
        model.train()
        train_losses = []
#        widgets = ['train :',Percentage(), ' ', Bar('#'),' ', Timer(),
 #          ' ', ETA(), ' ', FileTransferSpeed()]
  #      pbar = ProgressBar(widgets=widgets)
        i=0
        for batch_data in train_loader:
            image = batch_data['image'].to(device)
            label = batch_data['label_idx'].to(device)
            optimizer.zero_grad()
            if model_name=="inception_v3":
                logits = model(image)[0]
            else:
                logits = model(image)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            train_losses += [loss.detach().cpu().numpy().reshape(-1)]
            i+=1
            if i%100==1:
                print("[Train] Epoch:%d, batch:%d, Loss: %.2f"%(i_ep,i,loss.item()))
        train_losses = np.concatenate(train_losses).reshape(-1).mean()

        model.eval()
        val_losses = []
        preds = []
        true_labels = []
        #widgets = ['val:',Percentage(), ' ', Bar('#'),' ', Timer(),
        #   ' ', ETA(), ' ', FileTransferSpeed()]
        #pbar = ProgressBar(widgets=widgets)
        i=0
        for batch_data in val_loader:
            image = batch_data['image'].to(device)
            label = batch_data['label_idx'].to(device)
            with torch.no_grad():
                logits = model(image)
                
            loss = F.cross_entropy(logits, label).detach().cpu().numpy().reshape(-1)
            val_losses += [loss]
            true_labels += [label.detach().cpu().numpy()]
            preds += [(logits.max(1)[1].detach().cpu().numpy())]
            i+=1
            if i%100==1:
                print("[Eval] Epoch:%d, batch:%d, Loss: %.2f"%(i_ep,i,loss.item()))
        preds = np.concatenate(preds, 0).reshape(-1)
        true_labels = np.concatenate(true_labels, 0).reshape(-1)
        acc = accuracy_score(true_labels, preds)
        val_losses = np.concatenate(val_losses).reshape(-1).mean()
        scheduler.step(val_losses)
        # need python3.6
        print('Epoch : {}  val_acc : {:.5%} ||| train_loss : {:.5f}  val_loss : {:.5f}  |||\n'.format(i_ep,acc,train_losses,val_losses))
        if acc > best_acc:
            best_acc = acc
            files2remove = glob.glob(os.path.join(save_path,'ep_%s'%i_ep))
            for _i in files2remove:
                os.remove(_i)
            torch.save(model.cpu().state_dict(), os.path.join(save_path, 'ep_{}_{}_val_acc_{:.4f}.pth'.format(i_ep,model_name,acc)))
            model.to(device)

def train_cnn_model(model_name, gpu_ids, batch_size, resume=None):
    # Define CNN model
    Model = model_class_map[model_name]
    model = Model(num_classes=110)
    if resume is not None and resume!='':
        model.load_state_dict(torch.load(resume))

    # Loading data for ...
    print('loading data for train %s ....' %model_name)
    dataset_dir = '/cluster/home/it_stu28/jyz/data/IJCAI/IJCAI_2019_AAAC_train_ad'
    img_size = model.input_size[0]
    loaders = load_data_for_training_cnn(dataset_dir, img_size,  batch_size=batch_size*len(gpu_ids))

    # Prepare training options
    save_path = './weights/%s_adv4' %model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device('cuda:%d' %gpu_ids[0])
    model = model.to(device)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    print('start training cnn model.....\nit will take several hours, or even dozens of....')
    do_train(model_name, model, loaders['train_data'], loaders['val_data'],
              device, lr=args.lr, save_path=save_path, begin_e=args.begin_e, n_ep=args.n_ep, num_classes=110)

def defense(input_dir, target_model, weights_path, defense_type, defense_params, output_file, batch_size):
    # Define CNN model
    Model = model_class_map[target_model]
    # defense_fn = defense_method_map[defense_type]
    model = Model(num_classes=110)
    # Loading data for ...
    print('loading data for defense using %s ....' %target_model)
    img_size = model.input_size[0]
    loaders = load_data_for_defense(input_dir, img_size, batch_size)

    # Prepare predict options
    device = torch.device('cuda')
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    pth_file = glob.glob(os.path.join(weights_path, 'ep_*.pth'))[0]
    print('loading weights from : ', pth_file)
    model.load_state_dict(torch.load(pth_file))

    # for store result
    result = {'filename':[], 'predict_label':[]}
    # Begin predicting
    model.eval()
    widgets = ['dev_data :',Percentage(), ' ', Bar('#'),' ', Timer(),
       ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    for batch_data in pbar(loaders['dev_data']):
        image = batch_data['image'].to(device)
        filename = batch_data['filename']
        with torch.no_grad():
            logits = model(image)
        y_pred = logits.max(1)[1].detach().cpu().numpy().tolist()
        result['filename'].extend(filename)
        result['predict_label'].extend(y_pred)
    print('write result file to : ', output_file)
    pd.DataFrame(result).to_csv(output_file, header=False, index=False)
    


if __name__=='__main__':
     gpu_ids = args.gpu_id
     if isinstance(gpu_ids, int):
         gpu_ids = [gpu_ids]
     batch_size = args.batch_size
     target_model = args.target_model
################# Training #######################
     train_cnn_model(target_model, gpu_ids, batch_size, args.resume)
################## Defense #######################
     #defense(input_dir, target_model, weights_path, defense_type, defense_params, output_file, batch_size)

