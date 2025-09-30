import torch
import torchaudio
import os
import torch.nn.functional as F
from tqdm import tqdm
import sys
import numpy as np
import sys
import yaml
sys.path.append("/cto_studio/lijingyi/vocos/recon")  # 添加项目路径
from CosmosTokenizer.cosmos_tokenizer.modules import DecoderType, DiscreteQuantizer, EncoderType
import cosmos_tokenizer
from discrete_img import DiscreteImageTokenizer

scp_file = "/cto_studio/lijingyi/vocos/demo.txt"
output_dir = "/cto_studio/lijingyi/vocos/mushra_folder/mel_cap_eval/"
os.makedirs(output_dir, exist_ok=True)
with open(scp_file, "r") as f:
    wav_paths = [line.strip() for line in f.readlines() if line.strip().endswith(".wav")]

print(f"共发现 {len(wav_paths)} 个音频文件.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# load vocos model
config_path = "/cto_studio/lijingyi/vocos/logs96_decoder_ld/lightning_logs/version_0/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    # 2. 取出模型参数
model_cfg = config["model"]["init_args"]
from typing import Any, Dict, Tuple, Union, Optional

def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)

import torch
import sys
sys.path.append("/cto_studio/lijingyi/vocos")  # 添加项目路径
from vocos import Vocos
vocos = Vocos(
    feature_extractor=instantiate_class(args=(), init=model_cfg["feature_extractor"]),
    backbone=instantiate_class(args=(), init=model_cfg["backbone"]),
    head=instantiate_class(args=(), init=model_cfg["head"]),
)

ckpt_path = "/cto_studio/lijingyi/vocos/logs96_decoder_ld/lightning_logs/version_0/checkpoints/last.ckpt"
state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
vocos.load_state_dict(state_dict, strict=False)
vocos.to(device)
vocos.eval()
print("VOCOS模型加载完成！")


sample_rate = model_cfg["feature_extractor"]["init_args"]["sample_rate"]

#copy params from stage one setting
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
    quantizer=DiscreteQuantizer.VQ.name,
    embedding_dim=64,
    num_embeddings=8192,
    num_quantizers=1,
    name="DI",
    encoder=EncoderType.Default.name,
    decoder=DecoderType.Default.name,
)
model = DiscreteImageTokenizer(**params)
import torch
state_dict_model = torch.load(model_cfg["feature_extractor"]["init_args"]["vae_ckpt_path"], map_location=device)
model.load_state_dict(state_dict_model['encoder'])
model = model.to(device, dtype=torch.float32)
model.eval()


for idx, wav_path in tqdm(enumerate(wav_paths, 1), total=len(wav_paths)):
        # 1. 加载音频
    audio_input, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        print(f"sr:{sr}")
        audio_input = torchaudio.functional.resample(audio_input, sr, sample_rate)
    audio_input = audio_input.mean(dim=0, keepdim=True).to(device)  # Convert to mono
    with torch.inference_mode():
        mel_features = vocos.feature_extractor.extract_mel(audio_input)
        z_q, quant_loss, quant_info = model.encode(mel_features.unsqueeze(1)/5.0)
        recon = vocos.feature_extractor.mel_decoder(z_q).squeeze(1) * 5.0
        recon = torch.clamp(recon, min=-16.1181)
        audio_hat = vocos.decode(recon)

    output_path = os.path.join(output_dir, os.path.basename(wav_path))
    # 去掉 .wav 扩展名，生成对应的 id_t.txt 和 id_b.txt 路径
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    id_path = os.path.join(output_dir, f"{base_name}_id.txt")
    # 保存 id_t
    indices = quant_info[0]
    np.savetxt(id_path, indices.cpu().numpy().astype(int), fmt='%d')
    print(f"Saved id to {id_path}")
    # Ensure audio_hat is 2D (channels, samples) for torchaudio.save
    if audio_hat.dim() == 1:
        audio_hat = audio_hat.unsqueeze(0)
    torchaudio.save(output_path, audio_hat.cpu(), sample_rate)
    print(f"[{idx}/{len(wav_paths)}] 保存完成: {output_path}")

