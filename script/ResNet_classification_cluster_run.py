#Basics
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import csv
import time
import random
import pandas as pd
import scipy
import scipy.stats as stats
from scipy.stats import shapiro,normaltest,kstest,uniform
import seaborn as sns
import matplotlib.colors as colors
sys.path.append('../../')

#sklearn 
from multiprocessing import cpu_count
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,f1_score, roc_curve,auc, roc_auc_score,ConfusionMatrixDisplay

#Pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch. optim.lr_scheduler import _LRScheduler
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data.sampler import WeightedRandomSampler
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
from script.inception import InceptionTime, InceptionTime_

#utils
from script.utils import KFoldCVDataModule, CVTrainer, PadImage, ImbalancedDatasetSampler
import librosa
import librosa.display

#Captum
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(0)
torch.manual_seed(42)
#torch.backends.cudnn.benchmark = True
%matplotlib inline
device = torch.device("cuda")

default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# Parameter definition
epochs = 100 # no of epochs
model_size_ = '18'
Batch_Size = 128 #batch size
no_feutures = 128 #no of features per entry
no_classes = 2 #no of classes to classify 
training_on = True
root_dir = '/home/kvattis/Documents/data/'
train_csv_file = root_dir + 'train_dataset_control_AT_Mel_Spec_clean_long_win__final_v0.csv'
val_csv_file = root_dir + 'val_dataset_control_AT_Mel_Spec_clean_long_win__final_v0.csv'
parent_directory = '/home/kvattis/Documents/speech_analysis/'
checkpoint_directory = parent_directory + 'checkpoints/inception_class/'


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
    def __init__(self, csv_file, root_dir,transform):
            
        self.file_names = pd.read_csv(csv_file,header = None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_names)   

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        address =  os.path.join(self.root_dir,
                                self.file_names.iloc[idx, 2])
                
        df = pd.read_csv(address,header = None)                                                                              
        df_ar = df.to_numpy()
        df_ar = min_max_scale(df_ar)
        #df_ar =  global_std(df_ar)
        data = torch.Tensor(df_ar)
        label = self.file_names.iloc[idx, 3]
        label = torch.LongTensor([label])
        p_id = self.file_names.iloc[idx, 1]
        #p_id = torch.LongTensor([p_id])
        adr_id = int(str(p_id) + str(self.file_names.iloc[idx, 4]))
        adr_id = torch.LongTensor([adr_id])

        if self.transform:
            data = self.transform(data.T)
            data = data.T
            
        data = torch.unsqueeze(data, 0)
        return data, label, adr_id #[label, p_id] #torch.cat([data,data,data], dim = 0), label
    
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
                             'collate_fn' : PadImage()}
        #y_train = []
        
        #for i in range(len(train_dataset)):
         #   y_train.append(train_dataset[i][1].item())
            
        #y_train = np.array(y_train)
        
        #class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
        #weight = 1. / class_sample_count
        #samples_weight = np.array([weight[t] for t in y_train])

        #samples_weight = torch.from_numpy(samples_weight)
        #self.sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        
    def setup(self,stage=None):
        self.train_dataset = self.train_dataset # ImbalancedDatasetSampler(self.train_dataset) sampler = ImbalancedDatasetSampler(self.test_dataset)
        self.test_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler = ImbalancedDatasetSampler(self.train_dataset), shuffle = False, batch_size = self.batch_size, num_workers = 8, collate_fn=PadImage())

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = len(self.test_dataset), shuffle = False, num_workers = 8, collate_fn=PadImage())

    def test_dataloader(self):
        return DataLoader(self.test_dataset , batch_size = self.batch_size, shuffle = False, num_workers = 8, collate_fn=PadImage())
    
 #setup the module  
train_dataset = SpeechDataset(train_csv_file, root_dir,augm)
test_dataset = SpeechDataset(val_csv_file, root_dir,plain)
print(len(train_dataset), len(test_dataset))
data_module = SpeechDataModule(train_dataset, test_dataset, Batch_Size)

# Predictor class performing all the calculations for loss, backpropagation etc        
class Speech_Predictor(pl.LightningModule):
    def __init__(self, model_size: int, n_classes: int):
        super(Speech_Predictor,self).__init__()
        self.model = FullyConvolutionalResnet_(num_classes=n_classes, pretrained=False, model_size = model_size)
        self.criterion = nn.CrossEntropyLoss()#weight = class_weights)
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1(num_classes = n_classes, average = 'weighted')
        self.valid_f1 = torchmetrics.F1(num_classes = n_classes, average = 'weighted')
        self.test_f1 = torchmetrics.F1(num_classes = n_classes, average = 'weighted')
        self.train_f1_class = torchmetrics.F1(num_classes = n_classes, average = None)
        self.valid_f1_class = torchmetrics.F1(num_classes = n_classes, average = None)
        self.test_f1_class = torchmetrics.F1(num_classes = n_classes, average = None)
        self.train_auc_class = torchmetrics.AUROC(num_classes = n_classes, average = None)
        self.valid_auc_class = torchmetrics.AUROC(num_classes = n_classes, average = None)
        self.test_auc_class = torchmetrics.AUROC(num_classes = n_classes, average = None)
        self.n_classes_ = n_classes
        
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
            #output = F.log_softmax(output,dim =1)
            output = F.softmax(output,dim =1)
            return output
        
        
    def training_step(self,batch,batch_idx):
        X = batch[0]
        y = batch[1]
        
        #loss, outputs = self(torch.squeeze(X, 1),y)
        #outputs = F.softmax(outputs,dim =1)
        #yhat = torch.argmax(outputs, dim =1)
        #self.train_acc(yhat, y)
        #train_f1 = self.train_f1(yhat, y)
        #train_f1_class = self.train_f1_class(yhat, y)
        #train_auc_class = self.train_auc_class(outputs, y)
        
        
        
        X, y_a, y_b, lam = mixup_data(X, y, alpha = 0.9)
        X, y_a, y_b = map(Variable, ( X, y_a, y_b))
        loss, outputs = self(x = X,labels = y, targets_a = y_a, targets_b = y_b,lam = lam)
        outputs = F.softmax(outputs,dim =1)
        yhat = torch.argmax(outputs, dim =1)
        #self.train_acc(yhat, y)
        train_f1 = lam * self.train_f1(yhat, y_a) + (1 - lam) * self.train_f1(yhat, y_b)
        train_f1_class = lam * self.train_f1_class(yhat, y_a) + (1 - lam) * self.train_f1_class(yhat, y_b) 
        #train_auc_class = lam * np.array(self.train_auc_class(outputs, y_a)) + (1 - lam) * np.array(self.train_auc_class(outputs, y_b)) 
        
        
        self.log("train_loss",loss,prog_bar = True, logger = True, on_step=True, on_epoch=True)
        #self.log("train_accuracy",self.train_acc,prog_bar = True, logger = True, on_step=True, on_epoch=True)
        self.log("train_f1",train_f1,prog_bar = True, logger = True, on_step=True, on_epoch=True)
        self.log("train_f1_control",train_f1_class[0],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        self.log("train_f1_AT",train_f1_class[1],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        #self.log("train_f1_PD",train_f1_class[2],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        #self.log("train_auc_control",train_auc_class[0],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        #self.log("train_auc_AT",train_auc_class[1],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        #self.log("train_auc_PD",train_auc_class[2],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        
        return {"loss": loss, "accuracy": self.train_acc}
    
    def validation_step(self,batch,batch_idx):
        X = batch[0]
        y = batch[1]
        i_d = batch[2]
        loss, outputs = self(x = X, labels = y)
        outputs = F.softmax(outputs,dim =1)
        outputs, _ = groupby_mean(outputs, i_d)
        yhat = torch.argmax(outputs, dim =1)
        y, y_index = groupby_mean(y.view((y.shape[0],1)), i_d)
        y = y.view((y.shape[0])).type(torch.LongTensor).to(device)
        #self.valid_acc(yhat, y)
        valid_f1 = self.valid_f1(yhat, y)
        valid_f1_class = self.valid_f1_class(yhat, y)
        valid_auc_class = self.valid_auc_class(outputs, y)
        
        loss = self.criterion(outputs,y)
        self.log("val_loss",loss,prog_bar = True, logger = True, on_step=True, on_epoch=True)
        
        self.log("val_loss",loss,prog_bar = True, logger = True, on_step=True, on_epoch=True)
        #self.log("val_accuracy",self.valid_acc,prog_bar = True, logger = True, on_step=True, on_epoch=True)
        self.log("val_f1",valid_f1,prog_bar = True, logger = True, on_step=True, on_epoch=True)
        self.log("val_f1_control",valid_f1_class[0],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        self.log("val_f1_AT",valid_f1_class[1],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        #self.log("val_f1_PD",valid_f1_class[2],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        self.log("val_auc_control",valid_auc_class[0],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        self.log("val_auc_AT",valid_auc_class[1],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        #self.log("val_auc_PD",valid_auc_class[2],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        
        return {"loss": loss, "accuracy": self.valid_acc}
    '''
    def test_step(self,batch,batch_idx):
        X = batch[0]
        y = batch[1]
        loss, outputs = self(torch.squeeze(X, 1),y)
        outputs = F.softmax(outputs,dim =1)
        yhat = torch.argmax(outputs, dim =1)
        #self.test_acc(yhat,y)
        test_f1 = self.test_f1(yhat,y)
        test_f1_class = self.test_f1_class(yhat, y)
        test_auc_class = self.test_auc_class(outputs, y)

        self.log("test_loss",loss,prog_bar = True, logger = True,on_step=True, on_epoch=True)
        #self.log("test_accuracy",self.test_acc,prog_bar = False, logger = True, on_step=True, on_epoch=True)
        self.log("test_f1",test_f1,prog_bar = False, logger = True, on_step=True, on_epoch=True)
        self.log("test_f1_control",test_f1_class[0],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        self.log("test_f1_AT",test_f1_class[1],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        #self.log("test_f1_PD",test_f1_class[2],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        self.log("test_auc_control",test_auc_class[0],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        self.log("test_auc_AT",test_auc_class[1],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        #self.log("test_auc_PD",test_auc_class[2],prog_bar = False, logger = True, on_step=True, on_epoch=True)
        
        return {"loss": loss, "accuracy": self.test_acc}
    '''
        
    def configure_optimizers(self):
        #optimizer = optim.Adam(self.parameters(), lr =1e-4, weight_decay=1e-5)
        optimizer = optim.AdamW(self.parameters(), lr =1.e-4, weight_decay=1e-1)
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,weight_decay= 0.1)


        '''
        lr_scheduler = {
        'scheduler': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3 , epochs=50, anneal_strategy='linear'),
        'name': 'SDG_lr',
        'monitor': 'val_loss_epoch'}
        
        '''
        
        lr_scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10),
        'name': 'SDG_lr',
        'monitor': 'val_loss_epoch'}
        
        '''
        lr_scheduler = {
        'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 20),
        'name': 'SDG_lr',
        'monitor': 'val_loss_epoch'}
        '''

        return [optimizer]# , [lr_scheduler]

#define the model       
model = Speech_Predictor(n_classes = no_classes, model_size = model_size_)
model.double()

#checkpoint and loger definition
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_directory,filename='ResNet_best-checkpoint-{epoch:02d}-{val_loss:.2f}_control_AT_lw__v0',save_top_k=3, verbose =True , monitor = 'val_loss_epoch',mode ='min')
logger = TensorBoardLogger(parent_directory + 'lightning_logs', name = 'Speech_ResNet_control_AT_Mel_final')

if training_on is True:
    #Defining the trainer object
    trainer = pl.Trainer(logger = logger, callbacks = [checkpoint_callback], max_epochs = epochs, gpus = 1)
    trainer.fit(model, data_module)

    print('Training finished')
