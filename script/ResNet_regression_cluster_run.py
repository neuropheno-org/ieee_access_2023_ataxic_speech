#Basics
import numpy as np
import sys
import os
import csv
import time
import random
import pandas as pd
import scipy
sys.path.append('../../')

#sklearn 
from multiprocessing import cpu_count
from sklearn.utils import shuffle

#Pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch. optim.lr_scheduler import _LRScheduler
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
from torch.autograd import Variable

#Pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import Trainer
import torchmetrics

#models
from script.models import FullyConvolutionalResnet_

#utils
from script.utils import KFoldCVDataModule, CVTrainer, PadImage_inf, ImbalancedDatasetSampler

np.random.seed(0)
torch.manual_seed(42)
#torch.backends.cudnn.benchmark = True
device = torch.device("cuda")

# Parameter definition
epochs = 100 # no of epochs
model_size_ = '18'
Batch_Size = 128 #batch size
no_feutures = 128 #no of features per entry
no_classes = 2 #no of classes to classify 
training_on = True
root_dir = '/vast/neurobooth/data/data/'
train_csv_file = root_dir + 'train_dataset_control_AT_Mel_Spec_clean_long_win__final_v0.csv'
val_csv_file = root_dir + 'val_dataset_control_AT_Mel_Spec_clean_long_win__final_v0.csv'
train_demo_csv_file = root_dir +'train_demo_Mel_Spec_long_window_final_v0.csv'
val_demo_csv_file = root_dir + 'val_demo_Mel_Spec_long_window_final_v0.csv'
parent_directory = '/vast/neurobooth/kvattis/'
checkpoint_directory = parent_directory + 'checkpoints/resnet_regr/'

def min_max_scale(X, range_=(0, 1)):
    mi, ma = range_
    X_min = -80
    X_max = 3.8147e-06
    #X_std = (X - X.min()) / (X.max() - X.min())
    X_std = (X - X_min) / (X_max - X_min)
    X_scaled = X_std * (ma - mi) + mi
    return X_scaled

def augm(spec):
    freq_mask_param = 25
    time_mask_param = 10
    
    masking_T = T.TimeMasking(time_mask_param=time_mask_param)
    masking_f = T.FrequencyMasking(freq_mask_param = freq_mask_param)

    spec = masking_T(spec)
    spec = masking_f(spec)
    
    return spec

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def plain(spec):
    return spec

def groupby_mean(value:torch.Tensor, labels:torch.LongTensor) -> (torch.Tensor, torch.LongTensor):
    """Group-wise average for (sparse) grouped tensors

    Args:
        value (torch.Tensor): values to average (# samples, latent dimension)
        labels (torch.LongTensor): labels for embedding parameters (# samples,)

    Returns: 
        result (torch.Tensor): (# unique labels, latent dimension)
        new_labels (torch.LongTensor): (# unique labels,)

    Examples:
        >>> samples = torch.Tensor([
                             [0.15, 0.15, 0.15],    #-> group / class 1
                             [0.2, 0.2, 0.2],    #-> group / class 3
                             [0.4, 0.4, 0.4],    #-> group / class 3
                             [0.0, 0.0, 0.0]     #-> group / class 0
                      ])
        >>> labels = torch.LongTensor([1, 5, 5, 0])
        >>> result, new_labels = groupby_mean(samples, labels)

        >>> result
        tensor([[0.0000, 0.0000, 0.0000],
            [0.1500, 0.1500, 0.1500],
            [0.3000, 0.3000, 0.3000]])

        >>> new_labels
        tensor([0, 1, 5])
    """
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    labels = torch.LongTensor(list(map(key_val.get, labels)))

    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))

    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels.to(device), dtype=value.dtype).scatter_add_(0, labels.to(device), value.to(device))
    result = result.to(device) / labels_count.float().unsqueeze(1).to(device)
    new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist())))
    return result.to(device), new_labels.to(device)

#Define a pytorch Dataset               
class SpeechDataset(Dataset):
    def __init__(self, csv_file, demo_csv, root_dir,transform):
            
        self.file_names = pd.read_csv(csv_file,header = None, names=["No","P_ID", "Address","Label","Date"])
        self.demo = pd.read_csv(demo_csv, names=["No","P_ID", "Sex", "Bars","Age","Bars_Speech", "PDate"])
        self.file_names['Bars'] = self.demo['Bars']
        self.file_names = self.file_names[self.file_names.Label == 1] 
        self.file_names_bars = self.file_names[self.file_names['Bars'].notna()] 
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_names_bars)   

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        address =  os.path.join(self.root_dir,
                                self.file_names_bars.iloc[idx, 2])
                
        df = pd.read_csv(address,header = None)                                                                              
        df_ar = df.to_numpy()
        df_ar = min_max_scale(df_ar)
        data = torch.Tensor(df_ar)
        label = self.file_names_bars.iloc[idx, 3]
        label = torch.LongTensor([label])
        p_id = self.file_names_bars.iloc[idx, 1]
        adr_id = int(str(p_id) + str(self.file_names_bars.iloc[idx, 4]))
        adr_id = torch.LongTensor([adr_id])
        bars = self.file_names_bars.iloc[idx, 5]
        bars = torch.DoubleTensor([bars])

        if self.transform:
            data = self.transform(data.T)
            data = data.T
            
        data = torch.unsqueeze(data, 0)
        return data, bars, adr_id, label 
    
#DataModule to create the datasets and the dataloaders
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self,train_dataset, test_dataset, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.dataloader_kwargs = {'batch_size' : self.batch_size,
                             'shuffle' : True,
                             'num_workers' : 4,
                             'collate_fn' : PadImage_inf()}
        
    def setup(self,stage=None):
        self.train_dataset = self.train_dataset
        self.test_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle = True, batch_size = self.batch_size, num_workers = 8, collate_fn=PadImage_inf())

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = len(self.test_dataset), shuffle = False, num_workers = 8, collate_fn=PadImage_inf())

    def test_dataloader(self):
        return DataLoader(self.test_dataset , batch_size = self.batch_size, shuffle = False, num_workers = 8, collate_fn=PadImage_inf())
    
#setup the module  
train_dataset = SpeechDataset(train_csv_file, train_demo_csv_file, root_dir, augm)
test_dataset = SpeechDataset(val_csv_file, val_demo_csv_file, root_dir, plain)
print(len(train_dataset), len(test_dataset))
data_module = SpeechDataModule(train_dataset, test_dataset, Batch_Size)

# Predictor class performing all the calculations for loss, backpropagation etc        
class Speech_Predictor(pl.LightningModule):
    def __init__(self, model_size: int):
        super(Speech_Predictor,self).__init__()
        self.model = FullyConvolutionalResnet_(num_classes=1, pretrained=False, model_size = model_size)
        self.criterion = nn.HuberLoss(reduction='mean', delta=1.0)
        
    def forward(self,x,labels = None, targets_a = None, targets_b = None, lam = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            if lam is not None:
                loss =  mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
            else:
                loss = self.criterion(output,labels)
            return loss, output
        else:
            return output
        
        
    def training_step(self,batch,batch_idx):
        X = batch[0]
        y = batch[1]
        y = y.view((y.shape[0],1))
        loss, outputs = self(x = X,labels = y)
    
        self.log("train_loss",loss,prog_bar = True, logger = True, on_step=True, on_epoch=True)
        
        return {"loss": loss}
    
    def validation_step(self,batch,batch_idx):
        X = batch[0]
        y = batch[1]
        y = y.view((y.shape[0],1))
        i_d = batch[2]
        loss, outputs = self(x = X,labels = y)
        outputs, _ = groupby_mean(outputs, i_d)
        y, y_index = groupby_mean(y.view((y.shape[0],1)), i_d)
        y = y.type(torch.DoubleTensor).to(device)
        loss = self.criterion(outputs,y)
        self.log("val_loss",loss,prog_bar = True, logger = True, on_step=True, on_epoch=True)
        
        return {"loss": loss}
    
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr =1.e-4, weight_decay=1e-1)
        
        lr_scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10),
        'name': 'SDG_lr',
        'monitor': 'val_loss_epoch'}

        return [optimizer]# , [lr_scheduler]

#define the model       
model = Speech_Predictor(model_size = model_size_)
model.double()

#checkpoint and loger definition
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_directory,filename='ResNet_best-checkpoint-{epoch:02d}-{val_loss:.2f}_control_AT_lw_regression_v0',save_top_k=3, verbose =True , monitor = 'val_loss_epoch',mode ='min')
logger = TensorBoardLogger(parent_directory + 'lightning_logs', name = 'Speech_ResNet_control_AT_Mel_regression_final')

if training_on is True:
    #Defining the trainer object
    trainer = pl.Trainer(logger = logger, callbacks = [checkpoint_callback], max_epochs = epochs, gpus = 1)
    trainer.fit(model, data_module)

    print('Training finished')
