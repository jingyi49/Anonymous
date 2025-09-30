# --coding:utf-8--
import os,typing
from torch import nn
from typing import List
from audiotools import AudioSignal
from audiotools import STFTParams
import torchaudio
from pathlib import Path
from tqdm import tqdm
import torch
import time
import logging

def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
            self,
            window_lengths: List[int] = [2048, 512],
            loss_fn: typing.Callable = nn.L1Loss(),
            clamp_eps: float = 1e-5,
            mag_weight: float = 1.0,
            log_weight: float = 1.0,
            pow: float = 2.0,
            weight: float = 1.0,
            match_stride: bool = False,
            window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)

            x_mag = x.magnitude
            y_mag = y.magnitude
            if x_mag.shape[-1] != y_mag.shape[-1]:
                length = min(x_mag.shape[-1],y_mag.shape[-1])
                x_mag = x_mag[:,:,:,:length]
                y_mag = y_mag[:,:,:,:length]

            loss += self.log_weight * self.loss_fn(
                x_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mag, y_mag)
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(
            self,
            n_mels: List[int] = [150, 80],
            window_lengths: List[int] = [2048, 512],
            loss_fn: typing.Callable = nn.L1Loss(),
            clamp_eps: float = 1e-5,
            mag_weight: float = 1.0,
            log_weight: float = 1.0,
            pow: float = 2.0,
            weight: float = 1.0,
            match_stride: bool = False,
            mel_fmin: List[float] = [0.0, 0.0],
            mel_fmax: List[float] = [None, None],
            window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
                self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class MelSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """

    def __init__(
        self, sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 100,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=True, power=1,
        )
    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat = safe_log(self.mel_spec(y_hat))
        mel = safe_log(self.mel_spec(y))
        # breakpoint()
        if mel_hat.shape[-1] != mel.shape[-1]:
            length = min(mel_hat.shape[-1],mel.shape[-1])
            mel_hat = mel_hat[:,:,:length]
            mel = mel[:,:,:length]

        loss = torch.nn.functional.l1_loss(mel, mel_hat)
        return loss



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 44100   # 你训练 vocos/snac 的采样率，确认一下是否 24kHz

# 路径
gen_dir = Path("/tmp/lijingy-code-zsh/wavetokenizer_test")
ref_dir = Path("/cto_studio/vistring/zhaozhiyuan/datasets/AudioSet/wavs/test")

#gen_dir = Path("/tmp/lijingy-code-zsh/wavetokenizer_test")

# 损失函数
stft_loss_fn = MultiScaleSTFTLoss().to(DEVICE)
mel_loss_fn = MelSpecReconstructionLoss().to(DEVICE)

def load_audio(path, target_sr=TARGET_SR):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:  # 多通道转单通道
        wav = wav.mean(dim=0, keepdim=True)
    return wav.to(DEVICE)

stft_losses = []
mel_losses = []

# 遍历生成目录下的所有 wav 文件
for gen_wav in tqdm(list(gen_dir.rglob("*.wav")), desc="Evaluating"):
    ref_wav = ref_dir / gen_wav.name
    if not ref_wav.exists():
        print(f"⚠️ Reference not found: {ref_wav}")
        continue

    try:
        gen_audio = load_audio(gen_wav)
        ref_audio = load_audio(ref_wav)

        # 保证长度对齐
        min_len = min(gen_audio.shape[-1], ref_audio.shape[-1])
        gen_audio = gen_audio[..., :min_len]
        ref_audio = ref_audio[..., :min_len]

        # STFT Loss
        stft_loss = stft_loss_fn(AudioSignal(ref_audio,44100),AudioSignal(gen_audio, 44100))
        stft_losses.append(stft_loss)

        # Mel Loss
        mel_loss = mel_loss_fn(gen_audio, ref_audio)
        mel_losses.append(mel_loss)

    except Exception as e:
        print(f"Error processing {gen_wav}: {e}")

# 统计结果
print(f"✅ Done, total files: {len(stft_losses)}")
print(f"Average STFT Loss: {sum(stft_losses)/len(stft_losses):.6f}")
print(f"Average Mel Loss : {sum(mel_losses)/len(mel_losses):.6f}")