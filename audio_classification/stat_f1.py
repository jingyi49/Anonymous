from tqdm import tqdm
import numpy  as np
import csv
import torch
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
# ------------------- 文件 -------------------
pred_csv = "/cto_studio/lijingyi/vocos/prediction_vgg.csv"
class_labels_csv = "/cto_studio/lijingyi/vocos/class_labels_indices.csv"
segments_csv = "/cto_studio/lijingyi/vocos/eval_segments.csv"

# ------------------- 标签映射 -------------------
label_df = pd.read_csv(class_labels_csv)
#mid_to_index = dict(zip(label_df['mid'], label_df['index']))
mid_to_index = dict(zip(label_df['mid'].str.strip().str.strip('"'), label_df['index']))
index_to_label = dict(zip(label_df['index'], label_df['display_name']))
num_labels = len(label_df)

# ------------------- 读取预测 CSV -------------------
pred_df = pd.read_csv(pred_csv)
pred_df['TopK_labels'] = pred_df['TopK_labels'].str.split('|')

# ------------------- 读取 segments CSV -------------------
segments = []
with open(segments_csv, newline='') as f:
    reader = csv.DictReader(f, fieldnames=['YTID','start_seconds','end_seconds','positive_labels'])
    # 跳过前两行注释
    next(reader)
    next(reader)
    for row in reader:
        labels = [x.strip().strip('"') for x in row['positive_labels'].split(',')]
        segments.append({
            'YTID': row['YTID'],
            'start_seconds': float(row['start_seconds']),
            'end_seconds': float(row['end_seconds']),
            'labels': labels
        })

# ------------------- 构建 multi-hot -------------------
y_true = []
y_pred = []
y_prob = []
for seg in tqdm(segments, desc="Processing segments"):
    ytid = seg['YTID']
    pred_row = pred_df[pred_df['wav_path'].str.contains(ytid)]
    if pred_row.empty:
        continue
    # true vector
    true_vector = torch.zeros(num_labels, dtype=torch.int)
    for mid in seg['labels']:
        idx = mid_to_index.get(mid)
        if idx is not None:
            true_vector[idx] = 1
    
    # pred vector
    pred_vector = torch.zeros(num_labels, dtype=torch.int)
    prob_vector = torch.zeros(num_labels, dtype=torch.float)
    #for label in pred_row.iloc[0]['TopK_labels']:
    #    idxs = label_df[label_df['display_name'] == label]['index'].values
    #    if len(idxs) > 0:
    #        pred_vector[idxs[0]] = 1
    topk_labels = pred_row.iloc[0]['TopK_labels']  # 已经是列表
    topk_probs_str = pred_row.iloc[0]['TopK_probs']  # "0.8437|0.4815|0.4020"
    topk_probs = [float(p) for p in topk_probs_str.split('|')]

    for label, prob in zip(topk_labels, topk_probs):
        idxs = label_df[label_df['display_name'] == label]['index'].values
        if len(idxs) > 0:
            idx = idxs[0]
            pred_vector[idx] = 1
            prob_vector[idx] = prob

    y_true.append(true_vector)
    y_pred.append(pred_vector)
    y_prob.append(prob_vector)

# ------------------- 计算 F1 -------------------
y_true = torch.stack(y_true).numpy()
y_pred = torch.stack(y_pred).numpy()
f1 = f1_score(y_true, y_pred, average='macro')
mAP = average_precision_score(y_true, y_prob, average='macro')
print("Mean Average Precision (mAP):", mAP)
print("Multi-label F1:", f1)


# 假设 y_true 和 y_pred 是列表，每个元素都是 multi-hot tensor 或 numpy 数组
y_true = [t.numpy() if isinstance(t, torch.Tensor) else t for t in y_true]
y_pred = [t.numpy() if isinstance(t, torch.Tensor) else t for t in y_pred]

# 宽松匹配：将每个预测向量中预测为1的标签，如果在真实标签中，也标为1
y_pred_adjusted = []
for true_vec, pred_vec in zip(y_true, y_pred):
    pred_vec_adj = np.zeros_like(pred_vec)
    for i, val in enumerate(pred_vec):
        if val == 1 and true_vec[i] == 1:
            pred_vec_adj[i] = 1
    y_pred_adjusted.append(pred_vec_adj)

# 转为 numpy 数组
y_true_arr = np.stack(y_true)
y_pred_arr = np.stack(y_pred_adjusted)

f1 = f1_score(y_true_arr, y_pred_arr, average='macro')
print("Adjusted Multi-label F1:", f1)