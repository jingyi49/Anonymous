from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torchaudio
import torch
import csv
from pathlib import Path
from tqdm import tqdm

# ------------------- 配置 -------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SR = 16000
TOPK = 3

model_name = "mispeech/ced-tiny"
filelist_path = "/home/lijingy/model_output/class.eval"
output_csv = "/cto_studio/lijingyi/vocos/prediction_vgg.csv"

# ------------------- 模型 -------------------
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
model.to(DEVICE).eval()

# ------------------- 文件列表 -------------------
with open(filelist_path, "r") as f:
    wav_paths = [Path(line.strip()) for line in f.readlines()]

# ------------------- 推理 & 写 CSV -------------------
with open(output_csv, "w", newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['wav_path', 'TopK_labels', 'TopK_probs'])

    for wav_path in tqdm(wav_paths, desc="Processing WAVs"):
        # 加载音频
        audio, sr = torchaudio.load(wav_path)

        # 多通道 -> 单通道
        if audio.shape[0] != 1:
            audio = audio.mean(dim=0, keepdim=True)  # [1, num_samples]

        # 重采样
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            audio = resampler(audio)
            sr = TARGET_SR

        # 特征提取
        inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(DEVICE)

        # 推理
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0)  # [num_labels]

            topk_probs, topk_idx = torch.topk(logits, TOPK)
            topk_idx = topk_idx.cpu().tolist()
            topk_labels = [model.config.id2label[idx] for idx in topk_idx]
            topk_probs = [float(p.item()) for p in topk_probs]

        # 写入 CSV
        writer.writerow([str(wav_path), "|".join(topk_labels), "|".join(map(str, topk_probs))])

print(f"Predictions saved to {output_csv}")

