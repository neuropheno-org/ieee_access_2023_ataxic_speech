#Basics
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys
import os
import csv
import time
import random
import pandas as pd
import scipy
import scipy.signal as signal
import seaborn as sns
import soundfile as sf
import matplotlib.colors as colors
from h5io import read_hdf5
import matplotlib.pyplot as plt
import samplerate
import struct
import noisereduce as nr
import webrtcvad
sys.path.append('../../')

#Pytorch
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as transforms

#Pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer

#models
from script.models import FC_Resnet_

#librosa
import IPython.display as ipd
import librosa
import librosa.display


# The Frame generation classes are needed for the VAD algorithm.
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

#Data transfromations
def global_std(X, mean = -0.0005, std = 0.0454):
    """Global standardization"""
    X_scaled = (X - mean)/ std
    return X_scaled

def min_max_scale(X, range_=(0, 1)):
    """Scalling transform"""
    mi, ma = range_
    X_min = -50
    X_max = 50
    X_std = (X - X_min) / (X_max - X_min)
    X_scaled = X_std * (ma - mi) + mi
    return X_scaled

def transforms_val(spec):
    """Resizing transform"""
    transforms_resize = transforms.Resize((100, 100))
    spec = transforms_resize(spec)
    return spec


# Predictor class for classification        
class Speech_classifier(pl.LightningModule):
    """Predictor class for classification using both time and frequency derivatives"""
    def __init__(self, n_classes: int):
        super(Speech_classifier,self).__init__()
        self.model = FC_Resnet_(num_layers = 2, num_classes = n_classes)
        self.n_classes_ = n_classes
        
    def forward(self,x):
        output = self.model(x)
        output = F.softmax(output,dim =1)
        return output
    
#Predictor class for the regression models       
class Total_severity(pl.LightningModule):
    """Predictor class for BARS total score using only time derivative"""
    def __init__(self):
        super(Total_severity,self).__init__()
        self.model = FC_Resnet_(num_layers = 1, num_classes = 1)
        
    def forward(self,x):
        output = self.model(x)
        return output/100.
    
    
class Speech_severity(pl.LightningModule):
    """Predictor class for BARS speech score using only time derivative"""
    def __init__(self):
        super(Speech_severity,self).__init__()
        self.model = FC_Resnet_(num_layers = 1, num_classes = 1)
        
    def forward(self,x):
        output = self.model(x)
        return output/1000.
    
    
class Total_severity_both_grad(pl.LightningModule):
    """Predictor class for BARS total score using both time and frequency derivatives"""
    def __init__(self):
        super(Total_severity_both_grad,self).__init__()
        self.model = FC_Resnet_(num_layers = 2, num_classes = 1)
        
    def forward(self,x):
        output = self.model(x)
        return output/100.
    
    
class Speech_severity_both_grad(pl.LightningModule):
    """Predictor class for BARS speech score using both time and frequency derivatives"""
    def __init__(self):
        super(Speech_severity_both_grad,self).__init__()
        self.model = FC_Resnet_(num_layers = 2, num_classes = 1)
        
    def forward(self,x):
        output = self.model(x)
        return output/1000.

#Speech analysis class
class Speech_analysis(object):
    """Class that loads the speech models and raw data, applies the pre proccessing steps on the data and then calculates the model outputs. """
    
    def __init__(self,print_on = True, home_dir = '/home/kvattis/Documents/speech_analysis/'):
        """Initialiazing the parameters and loading the saved models"""
        self.print_on = print_on
        #initialize model subject id lists
        self.class_map = []
        for i in range(5):
            with open(home_dir + 'Model_training_map/train_class_v'+ str(i) + '.npy', 'rb') as f:
                self.class_map.append(np.load(f))

        self.severity_map = []
        for i in range(10):
            with open(home_dir + 'Model_training_map/train_severity_v'+ str(i) + '.npy', 'rb') as f:
                self.severity_map.append(np.load(f))
                
                
        #Pre-pro parameters
        self.sampling_rate = 8000
        self.sampling_rate_original = 44100#41000

        self.n_fft = 1024 # time window size for FFT
        self.n_mels = 128 # number of mel bands
        self.hop_length = 160 # Set the hop length; at 22050 Hz, 512 samples ~= 23ms

        self.size_ = 8000 #sample window size
        self.window_hop = 8000 #sample window hop
        self.speech_count_thresh = 0.6 #speech score threshold
        self.parent_directory = home_dir


        checkpoint_directory = self.parent_directory + 'checkpoints/resnet_class_fresh/' 
        self.models_class = []
        with open(self.parent_directory + 'Model_files/Classification_tf_grad.txt') as fp:

            for row in fp:
                line = row.rstrip("'\n'")
                checkpoint_loc = checkpoint_directory + line
                trained_model = Speech_classifier.load_from_checkpoint(checkpoint_loc,n_classes = 2)
                trained_model.freeze()
                trained_model.double()

                self.models_class.append(trained_model)
                if not line:
                    break
        
        if self.print_on:
            print('Classification model loaded')

        checkpoint_directory = self.parent_directory + 'checkpoints/resnet_regression_total_10fold_fresh__wc_/'
        self.models_total_bars = []
        with open(self.parent_directory + 'Model_files/Bars_total_t_grad.txt') as fp:

            for row in fp:
                line = row.rstrip("'\n'")
                checkpoint_loc = checkpoint_directory + line
                trained_model = Total_severity.load_from_checkpoint(checkpoint_loc)
                trained_model.freeze()
                trained_model.double()

                self.models_total_bars.append(trained_model)
                if not line:
                    break
        
        if self.print_on:
            print('Total BARS score model loaded') 

        checkpoint_directory = self.parent_directory + 'checkpoints/resnet_regression_speech_10fold_fresh__wc_/'
        self.models_speech_bars = []
        with open(self.parent_directory + 'Model_files/Bars_speech_t_grad.txt') as fp:

            for row in fp:
                line = row.rstrip("'\n'")
                checkpoint_loc = checkpoint_directory + line
                trained_model = Speech_severity.load_from_checkpoint(checkpoint_loc)
                trained_model.freeze()
                trained_model.double()

                self.models_speech_bars.append(trained_model)
                if not line:
                    break
        if self.print_on:
            print('Speech BARS score model loaded')
            
            
        checkpoint_directory = self.parent_directory + 'checkpoints/resnet_regression_total_10fold_fresh__wc_/'
        self.models_total_bars_both_grad = []
        with open(self.parent_directory + 'Model_files/Bars_total_tf_grad.txt') as fp:

            for row in fp:
                line = row.rstrip("'\n'")
                checkpoint_loc = checkpoint_directory + line
                trained_model = Total_severity_both_grad.load_from_checkpoint(checkpoint_loc)
                trained_model.freeze()
                trained_model.double()

                self.models_total_bars_both_grad.append(trained_model)
                if not line:
                    break
        
        if self.print_on:
            print('Total BARS score model with both gradients loaded') 

        checkpoint_directory = self.parent_directory + 'checkpoints/resnet_regression_speech_10fold_fresh__wc/'
        self.models_speech_bars_both_grad = []
        with open(self.parent_directory + 'Model_files/Bars_speech_tf_grad.txt') as fp:

            for row in fp:
                line = row.rstrip("'\n'")
                checkpoint_loc = checkpoint_directory + line
                trained_model = Speech_severity_both_grad.load_from_checkpoint(checkpoint_loc)
                trained_model.freeze()
                trained_model.double()

                self.models_speech_bars_both_grad.append(trained_model)
                if not line:
                    break
        if self.print_on:
            print('Speech BARS score model with both gradients loaded')
            
        if self.print_on:
            print('Ready')
    
    def load(self,file_name: str, P_ID = None):
        """Load the speech data and apply the pre processing"""
        
        #Read the data either from a wav file for the older studies or and hdf5 for Neurobooth
        if file_name[-3:] == "wav":
            if P_ID == None:
                #self.P_ID = file_name[52:57]
                self.P_ID = file_name[87:92]
            else:
                self.P_ID = P_ID
            self.audio_data, _ = librosa.load(file_name, sr=self.sampling_rate_original) 
            self.audio_data_nr = nr.reduce_noise(y=self.audio_data, sr=self.sampling_rate_original, prop_decrease=0.0, stationary=True)
            self.audio_data_8k = librosa.resample(self.audio_data_nr, orig_sr=self.sampling_rate_original, target_sr=self.sampling_rate)
            
        else:
            if P_ID == None:
                self.P_ID = file_name[22:28]
            else:
                self.P_ID = P_ID
                
            data = read_hdf5(file_name)

            self.audio_data = data['device_data']['time_series']
            self.audio_data_ts = data['device_data']['time_stamps']
            chunk_len = self.audio_data.shape[1]
            
            if chunk_len %2:
                self.audio_data =  self.audio_data[:,1:]
                
            
            marker_data = data['marker']['time_series']
            self.marker_data = list(np.concatenate(marker_data).flat)
            self.marker_ts = data['marker']['time_stamps']

            #Keep only the task audio
            Task_start = [idx for idx, s in enumerate(self.marker_data) if s.startswith('Task_start')][0]
            Task_end = [idx for idx, s in enumerate(self.marker_data) if s.startswith('Task_end')][0]

            evts = [np.argmin(np.abs(self.audio_data_ts - e)) for e in   self.marker_ts]
            self.audio_data =  np.hstack(self.audio_data[evts[0]:, :])
            self.audio_data = np.float64(self.audio_data)
            evts = [(e-evts[0])*1024 for e in evts]
            self.audio_data = self.audio_data[evts[Task_start]+ 2 * self.sampling_rate:evts[Task_end]]
            self.audio_data_nr = nr.reduce_noise(y=self.audio_data, sr=self.sampling_rate_original, prop_decrease=0.5
                                                 , stationary=True)
            self.audio_data_8k = librosa.resample(self.audio_data_nr, orig_sr=self.sampling_rate_original, target_sr=self.sampling_rate)
        
        #Start pre-proc
        signal_ = self.audio_data_8k
        #initialize the VAD algorithm
        vad = webrtcvad.Vad(1)

        y2 = self.audio_data_nr
        sr2 = self.sampling_rate_original
        y2 = np.transpose(y2.astype(float))
        y2 = librosa.to_mono(y2)
        sr2_down = 8000 # this is necessary for the VAD algorithm
        y2_down = samplerate.resample(y2, sr2_down * 1.0 / sr2, 'sinc_best')  # downsample
        #y2_down = y2_down[index[0]:index[1]]
        y2_down = np.array(y2_down)
        #y2_down = 10*(y2_down - y2_down.mean()) / y2_down.std()
        if y2_down.dtype == 'float32':
            y2_down /=1.414
            y2_down *= 32767
            y2_down = y2_down.astype(np.int16)
        y2_down = np.round(y2_down) # round to nearest integer (again needed for VAD)
        y2_down = [int(x) for x in y2_down]
        y2_down_ = [max(min(x, 32767), -32768) for x in y2_down] # must fall in this range\
        
        #Splitting the sample into non overlapping windows
        self.S_dB_list = []
        self.audio_patches_list = []
        for i in range(100):
            if i*self.window_hop + self.size_ < len(signal_):

                signal = signal_[i*self.window_hop:i*self.window_hop + self.size_]
                y2_down = y2_down_[i*self.window_hop:i*self.window_hop + self.size_]

                raw_y2 = struct.pack("%dh" % len(y2_down), *y2_down) # package data for VAD
                frames = frame_generator(0.01, raw_y2, sr2_down) # create 10ms frames for VAD
                frames = list(frames)
                vad_bool = np.full(len(frames), np.nan)
                vad_times = np.full(len(frames), np.nan)

                for frame_num, frame in enumerate(frames):
                    vad_bool[frame_num] = vad.is_speech(frame.bytes, sr2_down)
                    vad_times[frame_num] = frame.timestamp

                speech_count = np.count_nonzero(vad_bool)/vad_bool.shape[0]

            else:
                signal = signal_[i*self.window_hop:]
                y2_down = y2_down_[i*self.window_hop:]
                if len(signal) < self.size_/2.:
                    break
                raw_y2 = struct.pack("%dh" % len(y2_down), *y2_down) # package data for VAD
                frames = frame_generator(0.01, raw_y2, sr2_down) # create 10ms frames for VAD (100 samples per frame)
                frames = list(frames)
                vad_bool = np.full(len(frames), np.nan)
                vad_times = np.full(len(frames), np.nan)
                for frame_num, frame in enumerate(frames):
                    vad_bool[frame_num] = vad.is_speech(frame.bytes, sr2_down) 
                    vad_times[frame_num] = frame.timestamp

                speech_count = np.count_nonzero(vad_bool)/vad_bool.shape[0]
            
            if speech_count > self.speech_count_thresh:
                #Creating the Spectogram if the patch passes the speech score threshold
                signal = (signal - signal.mean()) / signal.std()
                self.audio_patches_list.append(signal)
                S = librosa.feature.melspectrogram(signal, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, window='hann' )
                S_dB = librosa.power_to_db(S, ref=1.0)
                S_dB = S_dB.T
                self.S_dB_list.append(S_dB)
        
        self.sample_size_s = len(self.S_dB_list)
        
        if self.print_on:
            print('Numnber of samples obtained: ', len(self.S_dB_list))
        #else:
            #return self.sample_size_s
    
    def play_original_audio(self):
        ipd.display(ipd.Audio(self.audio_data, rate = self.sampling_rate_original, autoplay=False))
        
    def play_resampled_audio(self):
        ipd.display(ipd.Audio(self.audio_data_8k, rate = self.sampling_rate, autoplay=False))
        
    def play_audio_nr(self):
        audio = self.audio_data_nr #np.hstack(self.audio_patches_list)
        ipd.display(ipd.Audio(audio, rate = self.sampling_rate_original, autoplay=False))
        
    def plot_mel(self):
        """Visualize the spectrogram"""
        S = librosa.feature.melspectrogram(self.audio_data, sr=self.sampling_rate_original, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, window='hann')
        S_dB = librosa.power_to_db(S, ref=1)

        #plot the spectogram
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(1, 1, 1)
        img = librosa.display.specshow(S_dB, x_axis='s',y_axis='mel', sr=self.sampling_rate_original, hop_length= self.hop_length, ax=ax) 
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img, cax=cax, format="%+2.f dB")
        
        cbar.ax.tick_params(labelsize=25)
        ax.minorticks_on()
        ax.tick_params('both', length=10, width=2, which='major',direction="in")
        ax.tick_params('both', length=5, width=1, which='minor',direction="in")
        ax.tick_params(axis='x', labelsize=35)
        ax.tick_params(axis='y', labelsize=35)
        ax.xaxis.label.set_size(40)
        ax.yaxis.label.set_size(40)
        
    def plot_mel_nr(self):
        """Visualize the spectrogram"""
        audio = self.audio_data_nr#np.hstack(self.audio_patches_list)
        S = librosa.feature.melspectrogram(audio, sr=self.sampling_rate_original, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, window='hann')
        S_dB = librosa.power_to_db(S, ref=1)

        #plot the spectogram
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(1, 1, 1)
        img = librosa.display.specshow(S_dB, x_axis='s',y_axis='mel', sr=self.sampling_rate_original, hop_length= self.hop_length, ax=ax) 
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img, cax=cax, format="%+2.f dB")
        
        cbar.ax.tick_params(labelsize=25)
        ax.minorticks_on()
        ax.tick_params('both', length=10, width=2, which='major',direction="in")
        ax.tick_params('both', length=5, width=1, which='minor',direction="in")
        ax.tick_params(axis='x', labelsize=35)
        ax.tick_params(axis='y', labelsize=35)
        ax.xaxis.label.set_size(40)
        ax.yaxis.label.set_size(40)
        
        
    def plot_mel_resampled(self):
        """Visualize the spectrogram"""
        audio = np.hstack(self.audio_patches_list)
        S = librosa.feature.melspectrogram(audio, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, window='hann')
        S_dB = librosa.power_to_db(S, ref=1)

        #plot the spectogram
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(1, 1, 1)
        img = librosa.display.specshow(S_dB, x_axis='s',y_axis='mel', sr=self.sampling_rate, hop_length= self.hop_length, ax=ax) 
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img, cax=cax, format="%+2.f dB")
        
        cbar.ax.tick_params(labelsize=25)
        ax.minorticks_on()
        ax.tick_params('both', length=10, width=2, which='major',direction="in")
        ax.tick_params('both', length=5, width=1, which='minor',direction="in")
        ax.tick_params(axis='x', labelsize=35)
        ax.tick_params(axis='y', labelsize=35)
        ax.xaxis.label.set_size(40)
        ax.yaxis.label.set_size(40)
        
        
    def plot_wav(self):
        """Visualize the waveform"""
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(1, 1, 1)
        librosa.display.waveplot(self.audio_data_8k, sr=self.sampling_rate)

        ax.minorticks_on()
        ax.tick_params('both', length=10, width=2, which='major',direction="in")
        ax.tick_params('both', length=5, width=1, which='minor',direction="in")
        ax.tick_params(axis='x', labelsize=35)
        ax.tick_params(axis='y', labelsize=35)
        ax.xaxis.label.set_size(40)
        ax.yaxis.label.set_size(40)
        
    def save_audio(self,name, path= '/home/kvattis/Documents/',nr = False):
        """Save the audio in a wav file"""
        if nr:
            sf.write(path + name + '_nr.wav', self.audio_data_nr, self.sampling_rate_original, subtype='FLOAT')
        else:
            sf.write(path + name + '.wav', self.audio_data, self.sampling_rate_original, subtype='FLOAT')
            
    def classify(self):
        """Run the classification model and output the probability for Ataxia score. If the participant has been used for training only the model from the fold that didn't use them will be applied. Otherwise all models are applied and the final result is the median of the output."""
        S_dB = self.S_dB_list
        input_list = []
        for i in range(len(S_dB)):
            S_dB_i = min_max_scale(S_dB[i])
            S_dB_i_t = np.gradient(S_dB_i, axis = 0)
            S_dB_i_f = np.gradient(S_dB_i, axis = 1)
            S_dB_i_tf = np.stack((S_dB_i_t,S_dB_i_f), axis=0)
            S_dB_i_tf =  global_std(S_dB_i_tf)
            S_dB_i_tf = torch.DoubleTensor(S_dB_i_tf)
            S_dB_i_tf = transforms_val(S_dB_i_tf)

            input_list.append(S_dB_i_tf)

        input_tensor = torch.stack(input_list)
        
        results_list = []
        for i in range(5):   
            result_fold = self.models_class[i](input_tensor)
            if int(self.P_ID) in self.class_map[i]:
                continue
            results_list.append(np.median(result_fold,axis=0))
        final_score = np.median(results_list,axis=0)
        
        if self.print_on:
            print('Estimates ataxia score: ', final_score[1])
        
        return final_score[1]
    
    
    def BARS_total(self):
        """Run the severity estimation model and output the BARS_total score. If the participant has been used for training only the model from the fold that didn't use them will be applied. Otherwise all models are applied and the final result is the median of the output."""
        S_dB = self.S_dB_list
        input_list = []
        for i in range(len(S_dB)):
            S_dB_i = min_max_scale(S_dB[i])
            S_dB_i_t = np.gradient(S_dB_i, axis = 0)
            S_dB_i_t =  global_std(S_dB_i_t)
            S_dB_i_t = torch.DoubleTensor(S_dB_i_t)
            S_dB_i_t = torch.unsqueeze(S_dB_i_t, 0)
            S_dB_i_t = transforms_val(S_dB_i_t)
            input_list.append(S_dB_i_t)

        input_tensor = torch.stack(input_list)

        results_list = []
        for i in range(10):
            result_fold = self.models_total_bars[i](input_tensor)
            if int(self.P_ID) in self.severity_map[i]:
                continue
            results_list.append(np.median(result_fold,axis=0))
        #print(len(results_list))
        final_score = np.median(results_list,axis=0)
        
        if self.print_on:
            print('Estimates Bars_total score: ', final_score[0])
        
        return final_score[0]
    
    
    def BARS_speech(self):
        """Run the severity estimation model and output the BARS speech score. If the participant has been used for training only the model from the fold that didn't use them will be applied. Otherwise all models are applied and the final result is the median of the output."""
        
        S_dB = self.S_dB_list
        input_list = []
        for i in range(len(S_dB)):
            S_dB_i = min_max_scale(S_dB[i])
            S_dB_i_t = np.gradient(S_dB_i, axis = 0)
            S_dB_i_t =  global_std(S_dB_i_t)
            S_dB_i_t = torch.DoubleTensor(S_dB_i_t)
            S_dB_i_t = torch.unsqueeze(S_dB_i_t, 0)
            S_dB_i_t = transforms_val(S_dB_i_t)
            input_list.append(S_dB_i_t)

        input_tensor = torch.stack(input_list)

        results_list = []
        for i in range(10):
            result_fold = self.models_speech_bars[i](input_tensor)
            if int(self.P_ID) in self.severity_map[i]:
                continue
            results_list.append(np.median(result_fold,axis=0))
        #print(len(results_list))
        final_score = np.median(results_list,axis=0)
        
        if self.print_on:
            print('Estimates BARS_speech score: ', final_score[0])
        
        return final_score[0]
    
    
    
    def BARS_total_both_grad(self):
        """Run the severity estimation model that uses both frequency and time derivatives and output the BARS total score. If the participant has been used for training only the model from the fold that didn't use them will be applied. Otherwise all models are applied and the final result is the median of the output."""
        S_dB = self.S_dB_list
        input_list = []
        for i in range(len(S_dB)):
            S_dB_i = min_max_scale(S_dB[i])
            S_dB_i_t = np.gradient(S_dB_i, axis = 0)
            S_dB_i_f = np.gradient(S_dB_i, axis = 1)
            S_dB_i_tf = np.stack((S_dB_i_t,S_dB_i_f), axis=0)
            S_dB_i_tf =  global_std(S_dB_i_tf)
            S_dB_i_tf = torch.DoubleTensor(S_dB_i_tf)
            S_dB_i_tf = transforms_val(S_dB_i_tf)

            input_list.append(S_dB_i_tf)

        input_tensor = torch.stack(input_list)

        results_list = []
        for i in range(10):
            result_fold = self.models_total_bars_both_grad[i](input_tensor)
            if int(self.P_ID) in self.severity_map[i]:
                continue
            results_list.append(np.median(result_fold,axis=0))
        #print(len(results_list))
        final_score = np.median(results_list,axis=0)
        
        if self.print_on:
            print('Estimates Bars_total score: ', final_score[0])
        
        return final_score[0]
    
    
    def BARS_speech_both_grad(self):
        """Run the severity estimation model that uses both frequency and time derivatives and output the BARS speech score. If the participant has been used for training only the model from the fold that didn't use them will be applied. Otherwise all models are applied and the final result is the median of the output."""
        S_dB = self.S_dB_list
        input_list = []
        for i in range(len(S_dB)):
            S_dB_i = min_max_scale(S_dB[i])
            S_dB_i_t = np.gradient(S_dB_i, axis = 0)
            S_dB_i_f = np.gradient(S_dB_i, axis = 1)
            S_dB_i_tf = np.stack((S_dB_i_t,S_dB_i_f), axis=0)
            S_dB_i_tf =  global_std(S_dB_i_tf)
            S_dB_i_tf = torch.DoubleTensor(S_dB_i_tf)
            S_dB_i_tf = transforms_val(S_dB_i_tf)

            input_list.append(S_dB_i_tf)

        input_tensor = torch.stack(input_list)

        results_list = []
        for i in range(10):
            result_fold = self.models_speech_bars_both_grad[i](input_tensor)
            if int(self.P_ID) in self.severity_map[i]:
                continue
            results_list.append(np.median(result_fold,axis=0))
        #print(len(results_list))
        final_score = np.median(results_list,axis=0)
        
        if self.print_on:
            print('Estimates BARS_speech score: ', final_score[0])
        
        return final_score[0]
    
    
    
    
    
        
        