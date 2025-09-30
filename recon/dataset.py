import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import librosa
import torchaudio
from vocos import Vocos

def load_wav(full_path, sample_rate):
    y, sr = torchaudio.load(full_path)
    return y, sr


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_length, center=True):

    mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=num_mels,
            center=center,
            power=1,
        )
    
    mel=mel_spec(y)

    return mel #[batch_size,n_fft/2+1,frames]



def get_dataset_filelist(input_training_wav_list,input_validation_wav_list):

    with open(input_training_wav_list, 'r') as fi:
        training_files = [x for x in fi.read().split('\n') if len(x) > 0]

    with open(input_validation_wav_list, 'r') as fi:
        validation_files = [x for x in fi.read().split('\n') if len(x) > 0]

    return training_files, validation_files


class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_length, sampling_rate, split=True, shuffle=True, device=None, train=True):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_length = hop_length
        self.device = device
        self.train = train

    def __getitem__(self, index):
        filename = self.audio_files[index]
        
        audio, sr = load_wav(filename, self.sampling_rate)
        if audio.size(0)>1:
            audio = audio.mean(dim=0, keepdim=True)  #[1,T]
        gain = np.random.uniform(-1, -6) if self.train else -3
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sr, [["norm", f"{gain:.2f}"]])

        if audio.size(1) < self.segment_size:
            pad_length = self.segment_size - audio.size(1)
            padding_tensor = audio.repeat(1, 1 + pad_length // audio.size(1))
            audio = torch.cat((audio, padding_tensor[:, :pad_length]), dim=1)
                
        elif self.train:
            start = np.random.randint(low=0, high=audio.size(-1) - self.segment_size + 1)
            audio = audio[:, start : start + self.segment_size]
            
        else:
            #取segment_size整数倍，audio.size(-1)//self.segment_size
            length = (audio.size(-1) // self.segment_size) * self.segment_size
            audio = audio[..., :length]
            audio = audio[:, :self.segment_size]
            
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.num_mels,
            center=True,
            power=1,
        )
        mel = mel_spec(audio)
        mel_log = torch.log(torch.clip(mel, min=1e-7))
        return mel_log/5.0

    def __len__(self):
        return len(self.audio_files)
