from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import os
torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        try:
            # 检查文件扩展名，如果是.flac则使用soundfile
            if audio_path.lower().endswith('.flac'):
                # 使用soundfile读取.flac文件
                y, sr = sf.read(audio_path)
                
                # 将numpy数组转换为torch tensor
                y = torch.from_numpy(y).float()
                
                # 添加通道维度（soundfile返回的是 [samples] 或 [samples, channels]）
                if y.dim() == 1:
                    # 单声道： [samples] -> [1, samples]
                    y = y.unsqueeze(0)
                elif y.dim() == 2:
                    # 多声道： [samples, channels] -> [channels, samples]
                    y = y.transpose(0, 1)
                else:
                    # 其他维度情况，确保是 [channels, samples]
                    raise ValueError(f"不支持的音频维度: {y.dim()}")
            else:
                    # 对于其他格式，使用torchaudio
                y, sr = torchaudio.load(audio_path)
                
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
            
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]
