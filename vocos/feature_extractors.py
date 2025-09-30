from typing import List

import torch
import torchaudio
from torch import nn

from vocos.modules import safe_log
from cosmos_tokenizer.image_lib import ImageTokenizer
import sys
sys.path.append("/cto_studio/lijingyi/vocos/recon")  # 添加项目路径
from CosmosTokenizer.cosmos_tokenizer.modules import DecoderType, DiscreteQuantizer, EncoderType
import cosmos_tokenizer
from discrete_img import DiscreteImageTokenizer
import os
import torch.distributed as dist
import copy
class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")



class MelSpectrogramFeatures(FeatureExtractor):
    """
    集成 Mel-spectrogram 提取和 VQ-VAE mel decoder 的模块
    """
    def __init__(
        self,
        vae_ckpt_path: str,
        sample_rate=24000, 
        n_fft=1024, 
        hop_length=256, 
        n_mels=100, 
        padding="center"
    ):
        super().__init__()

        # 保存参数
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.padding = padding

        # 设置设备

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        self.device = device

        # 初始化 Mel-spectrogram 提取器
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding=="center",
            power=1,
        ).to(device)

        # 初始化 VQ-VAE Tokenizer
        params = dict(
            attn_resolutions=[6, 12],
            channels=128,
            channels_mult=[2, 4, 4],
            dropout=0.0,
            in_channels=1,
            spatial_compression=8,
            num_res_blocks=2,
            out_channels=1,
            resolution=96,
            patch_size=2,
            patch_method="haar",
            z_channels=256,
            z_factor=2,
            quantizer="VQ",
            embedding_dim=64,
            num_embeddings=8192,
            num_quantizers=1,
            name="DI",
            encoder="Default",
            decoder="Default",
        )
        model = DiscreteImageTokenizer(**params)
        state_dict_model = torch.load(vae_ckpt_path, map_location=device)
        model.load_state_dict(state_dict_model["encoder"])  # 仅加载 encoder 权重初始化
        model.to(device)

        # mel_encoder 冻结
        self.mel_encoder = model.encode

        # mel_decoder 可训练
        self.mel_decoder = nn.Sequential(
            copy.deepcopy(model.post_quant_conv),
            copy.deepcopy(model.decoder)
        )

        self.to(device)

    @torch.no_grad()
    def encode(self, mel_features: torch.Tensor):
        """返回 z_q，不计算梯度"""
        return self.mel_encoder(mel_features.unsqueeze(1)/5.0)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        输入 waveform，返回重建的 mel features
        """
        # 先提取 mel
        mel_features = self.extract_mel(audio)

        # encoder 只做前向，不训练
        with torch.no_grad():
            z_q, _, _ = self.encode(mel_features)

        # decoder 可训练
        recon = self.mel_decoder(z_q).squeeze(1) * 5.0
        recon = torch.clamp(recon, min=-16.1181)
        return recon

    def extract_mel(self, audio: torch.Tensor):
        """简易 mel 提取函数"""
        mel = self.mel_spec(audio)
        return torch.log(torch.clamp(mel, min=1e-7))




class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
    ):
        super().__init__()
        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(bandwidths)
        )
        codebook_weights = torch.cat([vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]], dim=0)
        self.codebook_weights = torch.nn.Parameter(codebook_weights, requires_grad=train_codebooks)
        self.bandwidths = bandwidths

    @torch.no_grad()
    def get_encodec_codes(self, audio):
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, **kwargs):
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")
        self.encodec.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])
        codes = self.get_encodec_codes(audio)
        # Instead of summing in the loop, it stores subsequent VQ dictionaries in a single `self.codebook_weights`
        # with offsets given by the number of bins, and finally summed in a vectorized operation.
        offsets = torch.arange(
            0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins, device=audio.device
        )
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)
