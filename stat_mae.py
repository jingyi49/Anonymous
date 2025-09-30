import sys
import torch
import torchaudio
import os
import torch.nn.functional as F
from tqdm import tqdm
import sys
import numpy as np
import sys
sys.path.append("/cto_labs/lijingyi/vq-vae-2-pytorch")  # 添加项目路径
from vqvae import VQVAE

scp_file = "/cto_studio/lijingyi/multi_band_DAC/filelist.test"
import torch.nn.functional as F

with open(scp_file, "r") as f:
    wav_paths = [line.strip() for line in f.readlines() if line.strip().endswith(".wav")]

print(f"共发现 {len(wav_paths)} 个音频文件.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE()
state_dict_model = torch.load("/cto_labs/lijingyi/vq-vae-2-pytorch/checkpoint_104/vqvae_096.pt", map_location="cpu")
model.load_state_dict(state_dict_model)
model=model.cuda()

sample_rate = 44100
total_loss = 0.0
for idx, wav_path in tqdm(enumerate(wav_paths, 1), total=len(wav_paths)):
        # 1. 加载音频
    audio_input, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        print(f"sr:{sr}")
        audio_input = torchaudio.functional.resample(audio_input, sr, sample_rate)
    audio_input = audio_input.mean(dim=0, keepdim=True).to(device)  # Convert to mono
    mel_spec_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=256,
            n_mels=96,
            center=True,
            power=1,).to(device)

    mel = mel_spec_fn(audio_input)
    
    #padding to 8的倍数
    orig_len = mel.shape[-1]
    pad_len = (8 - (orig_len % 8)) % 8  # 需要补齐的帧数
    if pad_len > 0:
        mel = F.pad(mel, (0, pad_len), mode='constant', value=0)
    with torch.no_grad():
        recon_mel = model(mel.unsqueeze(1))
    recon_mel_log = torch.log(torch.clip(recon_mel[0], min=1e-7))
    mel_log = torch.log(torch.clip(mel, min=1e-7))
    loss = torch.nn.L1Loss()(recon_mel_log[...,:orig_len].squeeze(), mel_log[...,:orig_len].squeeze())
    total_loss += loss.item()

print(f"Reconstruction Loss: {total_loss / len(wav_paths)}")
    