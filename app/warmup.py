# /app/warmup.py
import os
import json
from pathlib import Path

print(">> Warm-up: loading models on CPU ...")

# 统一环境变量（可按需覆盖）
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CE_MODEL    = os.getenv("CE_MODEL",    "cross-encoder/ms-marco-MiniLM-L-6-v2")
CLIP_ARCH   = os.getenv("CLIP_ARCH",   "ViT-B-32")
CLIP_PRETR  = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")

# 限制线程，避免构建期占满 CPU
try:
    import torch
    torch.set_num_threads(int(os.getenv("WARMUP_TORCH_THREADS", "1")))
except Exception as e:
    torch = None  # type: ignore
    print(f"[Torch] skipped: {e}")

# --- SBERT ---
try:
    from sentence_transformers import SentenceTransformer
    if torch is None:
        raise RuntimeError("Torch not available for SBERT warmup.")
    sbert = SentenceTransformer(SBERT_MODEL, device="cpu")
    _ = sbert.encode(["hello world"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    print(f"[SBERT] ok: {SBERT_MODEL}")
except Exception as e:
    print(f"[SBERT] skipped: {e}")

# --- CrossEncoder ---
try:
    from sentence_transformers import CrossEncoder
    if torch is None:
        raise RuntimeError("Torch not available for CrossEncoder warmup.")
    ce = CrossEncoder(CE_MODEL, device="cpu")
    _ = ce.predict([("what is contrastive learning?", "CLIP learns from image-text pairs")], show_progress_bar=False)
    print(f"[CE] ok: {CE_MODEL}")
except Exception as e:
    print(f"[CE] skipped: {e}")

# --- OpenCLIP ---
try:
    import open_clip
    from PIL import Image
    import numpy as np
    if torch is None:
        raise RuntimeError("Torch not available for OpenCLIP warmup.")

    # 返回: model, preprocess_train, preprocess_val
    clip_model, _, clip_pre_val = open_clip.create_model_and_transforms(
        CLIP_ARCH, pretrained=CLIP_PRETR, device="cpu"
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_ARCH)

    # text forward
    toks = tokenizer(["a cat on a mat"])  # 由 get_tokenizer 负责 pad/trunc
    with torch.inference_mode():
        _ = clip_model.encode_text(toks)

    # image forward (dummy 224x224)
    dummy = Image.new("RGB", (224, 224), (127, 127, 127))
    inp = clip_pre_val(dummy).unsqueeze(0)
    with torch.inference_mode():
        _ = clip_model.encode_image(inp)

    print(f"[OpenCLIP] ok: {CLIP_ARCH}/{CLIP_PRETR}")
except Exception as e:
    print(f"[OpenCLIP] skipped: {e}")

# --- 索引检查 ---
idx = Path("/app/index")

required = [
    "text_embeddings.npy",
    "clip_text_embeddings.npy",
    "image_embeddings.npy",
    "text_ids.json",
    "image_ids.json",
    "image_to_blog.json",
    "blog_text_for_rerank.json",
]
optional = [
    "clip_text_visual_embeddings.npy",  # 视觉文本（若启用了 --with-visual-text）
    "visual_text_for_clip.json",
    "orb_index.json",                   # ORB(可选)
]

def _shape_of_np(p: Path):
    try:
        import numpy as np
        arr = np.load(p, mmap_mode="r")
        return tuple(arr.shape)
    except Exception:
        return None

print(">> Checking index files under", idx)
for name in required:
    p = idx / name
    if not p.exists():
        print(" -", name, "MISSING (required)")
        continue
    if name.endswith(".npy"):
        print(" -", name, "exists, shape:", _shape_of_np(p))
    else:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            summary = f"len={len(obj)}" if isinstance(obj, (list, dict)) else type(obj).__name__
            print(" -", name, "exists,", summary)
        except Exception:
            print(" -", name, "exists")

for name in optional:
    p = idx / name
    if not p.exists():
        print(" -", name, "missing (optional)")
        continue
    if name.endswith(".npy"):
        print(" -", name, "exists (optional), shape:", _shape_of_np(p))
    else:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            summary = f"len={len(obj)}" if isinstance(obj, (list, dict)) else type(obj).__name__
            print(" -", name, "exists (optional),", summary)
        except Exception:
            print(" -", name, "exists (optional)")

# ORB 目录统计（与你的 precompute 一致：orb_desc/*.npz）
orb_dir = idx / "orb_desc"
if orb_dir.exists() and orb_dir.is_dir():
    try:
        cnt = sum(1 for _ in orb_dir.glob("*.npz"))
        print(f" - orb_desc/*.npz count: {cnt}")
    except Exception:
        pass

print("Warm-up done.")
