import torch
import numpy as np
import json
from PIL import Image
import open_clip
from typing import List, Tuple

# Thiết lập device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Load model OpenCLIP (ViT-L-14, pretrained with datacomp)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='datacomp_xl_s13b_b90k',
    device=device
)
tokenizer = open_clip.get_tokenizer('ViT-L-14')
model.eval()


# Đường dẫn
features_path = "D:\Workplace\Test_Baseline\AIC_Baseline\clip_features\\all_video_features.npy"
data_id_path = "D:\Workplace\Test_Baseline\AIC_Baseline\metadata\data_id.json"

# Load CLIP features đã lưu
all_features = np.load(features_path)

# Load data_id.json (index → image path)
with open(data_id_path, "r") as f:
    data_id = json.load(f)
@torch.no_grad()
def encode_text_query(query: str) -> np.ndarray:
    tokens = tokenizer([query]).to(device)
    text_features = model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features[0].cpu().numpy()
def retrieve_top_k(query: str, k: int = 5) -> List[Tuple[str, float]]:
    query_feat = encode_text_query(query)

    # Normalize features để tính cosine similarity
    video_feats = all_features / np.linalg.norm(all_features, axis=1, keepdims=True)
    query_feat /= np.linalg.norm(query_feat)

    # Cosine similarity
    sims = video_feats @ query_feat.T  # (num_images,)
    topk_indices = np.argsort(sims)[-k:][::-1]  # Lấy top-k lớn nhất

    results = [(data_id[str(i)], sims[i]) for i in topk_indices]
    return results


query = "a motobike in an ambulance"
results = retrieve_top_k(query, k=500)
print(results[0])
print("\n🎯 Top-k kết quả:")
paths = []
for i, (path, score) in enumerate(results):
    paths.append(str(path))

with open("500_paths.txt","w") as f:
    for path in paths:
        f.write(path+"\n")
        