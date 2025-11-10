"""Module used to serve the ML6 blog post retrieval engine (CPU-only),
with improved image path:
  - More image variants (rotations + mirror + grayscale)
  - Image→text uses CLIP full-text and optional visual-text (max fusion)
  - Per-doc pooling uses top-k mean (k=2) by default (less noisy than max)
  - Optional OCR fallback for low-confidence image queries
Retains full backward-compatibility with routes & outputs.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import base64
from io import BytesIO

from torch.quantization import quantize_dynamic
import torch.nn as nn
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
import torch
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
import time
from functools import lru_cache

# Optional libs
try:
    import cv2  # for ORB (optional)
except Exception:
    cv2 = None  # type: ignore

try:
    import pytesseract  # for OCR (optional)
    import re as _re
except Exception:
    pytesseract = None  # type: ignore
    _re = None          # type: ignore

# Threads (not affecting numerics)
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))

# Try FAISS (exact inner-product)
try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None

# --- Constants (env-tunable) ---
K = int(os.getenv("TOPK", "3"))  # final top-k
TEXT_CAND_K = int(os.getenv("TEXT_CAND_K", "30"))
IMAGE_IMG_CAND_K = int(os.getenv("IMAGE_IMG_CAND_K", "120"))
IMAGE_BLOG_CAND_K = int(os.getenv("IMAGE_BLOG_CAND_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))

# Weights
W_II = float(os.getenv("W_II", "1.0"))  # image→image branch
W_IT = float(os.getenv("W_IT", "1.0"))  # image→text (CLIP text) branch

# Pooling across multiple images within a doc
POOL_MODE = os.getenv("POOL_MODE", "topk_mean")  # "topk_mean" | "lse" | "max"
POOL_TOPK = int(os.getenv("POOL_TOPK", "2"))

# OCR fallback
IMG_OCR_THRESHOLD = float(os.getenv("IMG_OCR_THRESHOLD", "0.26"))  # trigger if top score below
IMG_OCR_GAP = float(os.getenv("IMG_OCR_GAP", "0.02"))              # also trigger if top-2 gap < this
IMG_OCR_WEIGHT = float(os.getenv("IMG_OCR_WEIGHT", "0.35"))        # weight for OCR→text scores

# Image variants toggles
ADD_MIRROR = os.getenv("IMAGE_ADD_MIRROR", "1") == "1"
ADD_GRAY = os.getenv("IMAGE_ADD_GRAY", "1") == "1"

# --- Paths ---
INDEX_DIR = Path(__file__).parent / "index"
EVAL_IMG_DIR = Path(__file__).parent / "data" / "eval_data" / "eval_images"

# ORB globals (unused by default, but kept for potential future tie-breaks)
_ORB = None
_ORB_MAP: Optional[Dict[str, str]] = None  # image_id -> orb_desc relative path

# FAISS expected paths (optional)
TEXT_FAISS_PATH = INDEX_DIR / "text_sbert_flatip.faiss"
TEXT_CLIP_FAISS_PATH = INDEX_DIR / "text_clip_flatip.faiss"
IMAGE_FAISS_PATH = INDEX_DIR / "image_flatip.faiss"

CE_DOC_MAX_WORDS = int(os.getenv("CE_DOC_MAX_WORDS", "220"))   # rerank 文本最多保留的词数
CE_BATCH = int(os.getenv("CE_BATCH", "64"))                    # 交叉编码批量
CE_SKIP_MINSIM = float(os.getenv("CE_SKIP_MINSIM", "0.42"))    # 早停：top1 绝对阈
CE_SKIP_DELTA  = float(os.getenv("CE_SKIP_DELTA",  "0.06"))
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml6-retrieval")

# --- FastAPI App ---
app = FastAPI()
HEALTH_ENDPOINT_NAME = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ENDPOINT_NAME = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

# --- Globals (loaded lazily) ---
_TEXT_IDS: Optional[List[str]] = None
_TEXT_EMB: Optional[np.ndarray] = None  # SBERT blog embeddings (N, 384)
_CLIP_TEXT_EMB: Optional[np.ndarray] = None  # CLIP text embeddings (N, 512)
_CLIP_TEXT_EMB_VIS: Optional[np.ndarray] = None  # NEW: CLIP visual-text embeddings (N, 512)
_BLOG_TEXT_FOR_RERANK: Optional[Dict[str, str]] = None
_IMAGE_IDS: Optional[List[str]] = None  # image filenames
_IMAGE_EMB: Optional[np.ndarray] = None  # CLIP image embeddings (M, 512)
_IMAGE_TO_BLOG: Optional[Dict[str, str]] = None  # image filename -> blog slug
_DOC_TITLES: Optional[List[str]] = None
_IMAGE_DOCIDS: Optional[np.ndarray] = None  # per image -> doc row id (-1 invalid)
_SLUG2ROW: Optional[Dict[str, int]] = None  # slug -> row index

# FAISS (Exact)
_FAISS_TEXT = None
_FAISS_TEXT_CLIP = None
_FAISS_IMAGE = None

# Models (lazy)
_SBERT = None
_CROSS_ENCODER = None
_CLIP_MODEL = None
_CLIP_TOKENIZER = None
_CLIP_PREPROC = None


# --- Pydantic Models ---
class ImageBytes(BaseModel):
    b64: str  # The image encoded as a base64 string.


class Instance(BaseModel):
    image_bytes: Optional[ImageBytes] = None  # Optional: base64 image
    text_input: Optional[str] = None  # Optional: text string


class PredictionResponseItem(BaseModel):
    ranked_documents: List[str] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    predictions: List[PredictionResponseItem]


# --- Utils ---
ENABLE_STAGE_TIMERS = os.getenv("ENABLE_STAGE_TIMERS", "0") == "1"
TIMER_SAMPLE = float(os.getenv("TIMER_SAMPLE", "1.0"))  # sample rate [0,1]


def _t(name: str):
    if not ENABLE_STAGE_TIMERS:
        class _Noop:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _Noop()
    import random
    if TIMER_SAMPLE < 1.0 and random.random() > TIMER_SAMPLE:
        class _Noop:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _Noop()

    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *a):
            ms = (time.perf_counter() - self.t0) * 1000.0
            logger.info("latency_ms=%.1f stage=%s", ms, name)
    return _T()


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _normalize_slug(s: str) -> str:
    s = (s or "").strip().lower()
    if s.endswith(".json"):
        s = s[:-5]
    while s.endswith("-"):
        s = s[:-1]
    return s

def _truncate_words(text: str, max_words: int) -> str:
    if not text or max_words <= 0:
        return text or ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

def _load_image_from_path(fn: str) -> Image.Image:
    p = Path(fn)
    if not p.is_absolute():
        p = EVAL_IMG_DIR / fn
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).convert("RGB")


def _load_json(fp: Path) -> Any:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


# === FAISS helpers ===
def _load_or_build_faiss_exact(mat: np.ndarray, path: Path) -> Optional[Any]:
    """Load or build an IndexFlatIP (exact). Does not change ranking results."""
    if faiss is None or mat is None or mat.size == 0:
        return None
    try:
        if path.exists():
            idx = faiss.read_index(str(path))
            if idx.d != mat.shape[1]:
                logger.warning(f"FAISS index dim mismatch for {path}: got {idx.d}, expect {mat.shape[1]}")
                return None
            return idx
        # build in memory
        idx = faiss.IndexFlatIP(mat.shape[1])
        idx.add(mat.astype(np.float32, copy=False))
        logger.info(f"Built in-memory FAISS IndexFlatIP for shape {mat.shape}")
        return idx
    except Exception as e:
        logger.warning(f"FAISS load/build failed for {path}: {e}")
        return None


def _faiss_topk(index, q: np.ndarray, k: int) -> np.ndarray:
    """Exact Top-K with FAISS; q must be float32 & L2-normalized."""
    if index is None:
        return None
    D, I = index.search(q.reshape(1, -1).astype(np.float32, copy=False), min(k, index.ntotal))
    return I[0].astype(np.int64)


# --- Load all once ---
def _ensure_loaded():
    """Load all indices + models exactly once."""
    global _TEXT_IDS, _TEXT_EMB, _CLIP_TEXT_EMB, _CLIP_TEXT_EMB_VIS, _BLOG_TEXT_FOR_RERANK
    global _IMAGE_IDS, _IMAGE_EMB, _IMAGE_TO_BLOG
    global _SBERT, _CROSS_ENCODER, _CLIP_MODEL, _CLIP_TOKENIZER, _CLIP_PREPROC
    global _ORB, _ORB_MAP, _DOC_TITLES, _IMAGE_DOCIDS, _SLUG2ROW
    global _FAISS_TEXT, _FAISS_TEXT_CLIP, _FAISS_IMAGE

    if _TEXT_IDS is None:
        logger.info("Loading text indices...")
        _TEXT_IDS = _load_json(INDEX_DIR / "text_ids.json") if (INDEX_DIR / "text_ids.json").exists() else []
        _TEXT_EMB = np.load(INDEX_DIR / "text_embeddings.npy") if (INDEX_DIR / "text_embeddings.npy").exists() else np.zeros((0, 384), dtype="float32")
        _CLIP_TEXT_EMB = np.load(INDEX_DIR / "clip_text_embeddings.npy") if (INDEX_DIR / "clip_text_embeddings.npy").exists() else np.zeros((0, 512), dtype="float32")
        # NEW optional visual-text embeddings
        if (INDEX_DIR / "clip_text_visual_embeddings.npy").exists():
            _CLIP_TEXT_EMB_VIS = np.load(INDEX_DIR / "clip_text_visual_embeddings.npy")
        else:
            _CLIP_TEXT_EMB_VIS = None
        _BLOG_TEXT_FOR_RERANK = _load_json(INDEX_DIR / "blog_text_for_rerank.json") if (INDEX_DIR / "blog_text_for_rerank.json").exists() else {}

        logger.info("Loading image indices...")
        _IMAGE_IDS = _load_json(INDEX_DIR / "image_ids.json") if (INDEX_DIR / "image_ids.json").exists() else []
        _IMAGE_EMB = np.load(INDEX_DIR / "image_embeddings.npy") if (INDEX_DIR / "image_embeddings.npy").exists() else np.zeros((0, 512), dtype="float32")
        _IMAGE_TO_BLOG = _load_json(INDEX_DIR / "image_to_blog.json") if (INDEX_DIR / "image_to_blog.json").exists() else {}

        logger.info(f"Blogs: {len(_TEXT_IDS)}, Images: {len(_IMAGE_IDS)}")

    if _SBERT is None:
        logger.info("Loading SBERT (CPU)...")
        from sentence_transformers import SentenceTransformer
        _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            try:
                ce.model = quantize_dynamic(ce.model, {nn.Linear}, dtype=torch.qint8)
                logger.info("Cross-Encoder dynamically quantized (int8).")
            except Exception as qe:
                logger.warning(f"Cross-Encoder quantization skipped: {qe}")
            _CROSS_ENCODER = ce
            logger.info("Loading Cross-Encoder (CPU)...done")
        except Exception as e:
            _CROSS_ENCODER = None
            logger.error(f"Cross-Encoder failed to load, will skip rerank: {e}")

    if _CLIP_MODEL is None:
        logger.info("Loading OpenCLIP (CPU)...")
        import open_clip
        _CLIP_MODEL, _, _CLIP_PREPROC = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
        )
        _CLIP_MODEL.eval()
        _CLIP_TOKENIZER = open_clip.get_tokenizer("ViT-B-32")

    if _ORB is None and cv2 is not None:
        logger.info("Loading ORB (CPU)...")
        _ORB = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=8, fastThreshold=10)

    if _ORB_MAP is None:
        orb_idx_fp = INDEX_DIR / "orb_index.json"
        _ORB_MAP = _load_json(orb_idx_fp) if orb_idx_fp.exists() else {}
        if _ORB_MAP:
            logger.info(f"ORB entries: {len(_ORB_MAP)}")

    if _DOC_TITLES is None:
        logger.info("Building doc title list (slugs)...")
        _DOC_TITLES = _TEXT_IDS

    if _SLUG2ROW is None:
        _SLUG2ROW = {slug: idx for idx, slug in enumerate(_TEXT_IDS)}

    if _IMAGE_DOCIDS is None:
        logger.info("Building image->doc id array...")
        slug2row = _SLUG2ROW or {slug: idx for idx, slug in enumerate(_TEXT_IDS)}
        arr = np.full((len(_IMAGE_IDS),), -1, dtype=np.int32)
        for i, img_id in enumerate(_IMAGE_IDS):
            blog = _IMAGE_TO_BLOG.get(img_id, "")
            blog = _normalize_slug(blog)
            if blog in slug2row:
                arr[i] = slug2row[blog]
        _IMAGE_DOCIDS = arr

    # --- FAISS exact indexes (optional acceleration; does not change ranking) ---
    if _FAISS_TEXT is None and _TEXT_EMB is not None and _TEXT_EMB.size:
        _FAISS_TEXT = _load_or_build_faiss_exact(_TEXT_EMB, TEXT_FAISS_PATH)
    if _FAISS_TEXT_CLIP is None and _CLIP_TEXT_EMB is not None and _CLIP_TEXT_EMB.size:
        _FAISS_TEXT_CLIP = _load_or_build_faiss_exact(_CLIP_TEXT_EMB, TEXT_CLIP_FAISS_PATH)
    if _FAISS_IMAGE is None and _IMAGE_EMB is not None and _IMAGE_EMB.size:
        _FAISS_IMAGE = _load_or_build_faiss_exact(_IMAGE_EMB, IMAGE_FAISS_PATH)


# --- Applicant Implementation: Loaders (backward-compatible names) ---
def load_image_features() -> Optional[Dict[str, Any]]:
    _ensure_loaded()
    if _IMAGE_EMB.shape[0] == 0:
        return None
    return {img_id: _IMAGE_EMB[i] for i, img_id in enumerate(_IMAGE_IDS)}


def load_text_features() -> Optional[Dict[str, Any]]:
    _ensure_loaded()
    if _TEXT_EMB.shape[0] == 0:
        return None
    return {slug: _TEXT_EMB[i] for i, slug in enumerate(_TEXT_IDS)}


def load_image_to_blogpost_mappings() -> Optional[Dict[str, str]]:
    _ensure_loaded()
    return _IMAGE_TO_BLOG


def _parse_payload_to_queries(payload: Dict[str, Any]) -> List[Tuple[str, Any]]:
    queries: List[Tuple[str, Any]] = []
    if "instances" in payload and isinstance(payload["instances"], list):
        for inst in payload["instances"]:
            if isinstance(inst, dict):
                if "text_input" in inst and inst["text_input"]:
                    queries.append(("text", str(inst["text_input"])))
                    continue
                if "image_bytes" in inst and isinstance(inst["image_bytes"], dict) and inst["image_bytes"].get("b64"):
                    try:
                        raw = base64.b64decode(inst["image_bytes"]["b64"], validate=True)
                        img = Image.open(BytesIO(raw)).convert("RGB")
                        queries.append(("image", img))
                        continue
                    except Exception as e:
                        raise HTTPException(400, f"Invalid base64 image: {e}")
            # legacy schema support in instances
            if isinstance(inst, dict) and "query_type" in inst and "query_content" in inst:
                qtype = inst["query_type"]
                qcont = inst["query_content"]
                if qtype == "text":
                    queries.append(("text", str(qcont)))
                elif qtype == "image":
                    img = _load_image_from_path(str(qcont))
                    queries.append(("image", img))
                else:
                    raise HTTPException(400, f"Unsupported query_type: {qtype}")
    if queries:
        return queries

    if "query_type" in payload and "query_content" in payload:
        qtype = payload["query_type"]
        qcont = payload["query_content"]
        if qtype == "text":
            return [("text", str(qcont))]
        elif qtype == "image":
            img = _load_image_from_path(str(qcont))
            return [("image", img)]
        else:
            raise HTTPException(400, f"Unsupported query_type: {qtype}")

    if "type" in payload:
        qtype = payload["type"]
        if qtype == "text":
            return [("text", str(payload.get("text", "")))]
        elif qtype == "image":
            if "image_b64" in payload:
                try:
                    raw = base64.b64decode(payload["image_b64"], validate=True)
                    img = Image.open(BytesIO(raw)).convert("RGB")
                    return [("image", img)]
                except Exception as e:
                    raise HTTPException(400, f"Invalid image_b64: {e}")
            if "image_path" in payload:
                img = _load_image_from_path(str(payload["image_path"]))
                return [("image", img)]
            else:
                raise HTTPException(400, f"Unsupported type: {qtype}")

    raise HTTPException(400, "Unsupported request schema")


# === Query pipelines ===
@lru_cache(maxsize=256)
def _cached_text_vec(text: str) -> Optional[np.ndarray]:
    """Cached text vector (SBERT)."""
    return encode_text(text)

def _process_text_queries_batch(texts: List[str],
                                rerank_top_k: Optional[int] = None) -> List[List[str]]:
    """
    批量处理多条文本查询：一次性 SBERT 编码 + 一次性 FAISS 多查询 + 一次性 CE 预测。
    与逐条处理的候选、分数、排序完全一致（零精度损失）。
    返回：按输入顺序的每条查询 Top-K slug 列表。
    """
    _ensure_loaded()
    n = len(texts)
    if n == 0:
        return []

    # 1) SBERT 批量编码（与单条一致，只是一次性喂入）
    with torch.inference_mode():
        qvecs = _SBERT.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False, batch_size=max(8, min(64, n)))
    qvecs = qvecs.astype("float32")

    # 2) 初排（FAISS 多查询；若没 FAISS 则 numpy 路径，结果相同）
    k0 = max(TEXT_CAND_K, K)
    cand_lists: List[List[str]] = []
    if _FAISS_TEXT is not None and _TEXT_EMB is not None and _TEXT_EMB.size:
        kfaiss = min(k0, _FAISS_TEXT.ntotal)
        D, I = _FAISS_TEXT.search(qvecs.astype(np.float32, copy=False), kfaiss)  # (n,k)
        # 为保持与单查询路径一致：我们仍用精确内积重排一下 FAISS 返回的 top-k（不会改变集合，只确保同序）
        for qi in range(n):
            idxs = I[qi]
            sims = _TEXT_EMB[idxs] @ qvecs[qi]
            order = np.argsort(-sims)
            idxs = idxs[order]
            cand_lists.append([_TEXT_IDS[int(i)] for i in idxs])
    else:
        sims_all = qvecs @ _TEXT_EMB.T  # (n, N)
        order = np.argsort(-sims_all, axis=1)[:, :k0]
        for qi in range(n):
            idxs = order[qi]
            cand_lists.append([_TEXT_IDS[int(i)] for i in idxs])

    # 3) 组装 CE pair（严格沿用逐条逻辑的候选数）
    rtk = int(rerank_top_k) if rerank_top_k else RERANK_TOP_K
    pairs: List[Tuple[str, str]] = []
    ptr: List[Tuple[int, str]] = []  # (query_idx, slug)
    for qi, slugs in enumerate(cand_lists):
        use_slugs = slugs[:max(rtk, K)]
        for slug in use_slugs:
            doc_text = _BLOG_TEXT_FOR_RERANK.get(slug, slug.replace("-", " "))
            pairs.append((texts[qi], doc_text))
            ptr.append((qi, slug))

    # 4) CE 一次/少次数大批预测（与逐条 CE 结果完全一致）
    if _CROSS_ENCODER is None:
        # 没有 CE 就直接用初排（与原逻辑一致）
        return [slugs[:K] for slugs in cand_lists]

    with torch.inference_mode():
        scores = _CROSS_ENCODER.predict(pairs, show_progress_bar=False, batch_size=64)

    # 5) 把 CE 分数切回每条查询，再取 Top-K
    from collections import defaultdict
    per_q: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for sc, (qi, slug) in zip(scores, ptr):
        per_q[qi].append((slug, float(sc)))

    results: List[List[str]] = []
    for qi in range(n):
        if qi not in per_q:
            # 理论不会发生，保险起见
            results.append(cand_lists[qi][:K])
        else:
            ranked = sorted(per_q[qi], key=lambda x: x[1], reverse=True)[:K]
            results.append([slug for slug, _ in ranked])
    return results

def _process_text_query(text: str) -> List[str]:
    """Text: SBERT initial → (optional CE rerank) → top-K slugs"""
    if _TEXT_EMB.shape[0] == 0:
        return []
    with _t("text_encode"):
        qvec = _cached_text_vec(text)
        if qvec is None:
            return []

    with _t("text_initial"):
        init_top = _topk_from_matrix(qvec, _TEXT_EMB, _TEXT_IDS, k=max(TEXT_CAND_K, K), faiss_index=_FAISS_TEXT)
        if not init_top:
            return []
        cand_slugs = [sid for sid, _ in init_top]
        # ---- 早停：SBERT 已经非常确定时，跳过 CE ----
        if len(init_top) >= 2:
            s1 = float(init_top[0][1]); s2 = float(init_top[1][1])
            if s1 >= CE_SKIP_MINSIM and (s1 - s2) >= CE_SKIP_DELTA:
                return cand_slugs[:K]

    # 动态缩小 CE 的候选数
    use_slugs = cand_slugs[:max(RERANK_TOP_K, K)]
    if len(use_slugs) > K + 1 and len(init_top) >= 5:
        # 如果前 5 与第 1 差距很大，减少 CE 开销
        s1 = float(init_top[0][1]); s5 = float(init_top[4][1])
        if (s1 - s5) >= 0.04:
            use_slugs = cand_slugs[:max(K, min(RERANK_TOP_K, 4))]

    # 组装 CE 输入（对 rerank 文本做词数截断）
    pairs = []
    for slug in use_slugs:
        doc_text = _BLOG_TEXT_FOR_RERANK.get(slug, slug.replace("-", " "))
        if CE_DOC_MAX_WORDS > 0:
            doc_text = _truncate_words(doc_text, CE_DOC_MAX_WORDS)
        pairs.append((text, doc_text))

    if _CROSS_ENCODER is None:
        return use_slugs[:K]

    with torch.inference_mode():
        with _t("text_rerank"):
            scores = _CROSS_ENCODER.predict(pairs, show_progress_bar=False, batch_size=CE_BATCH)
            order = np.argsort(-np.asarray(scores))
            return [use_slugs[idx] for idx in order[:K]]



def _process_image_query(img: Image.Image) -> List[str]:
    """
    Image query pipeline with two branches and robust pooling:
      Branch A (image→image): query variants vs CLIP image embeddings → per-doc pool (top-k mean)
      Branch B (image→text):  query variants vs CLIP text (full) and optional visual-text → per-doc max
      Final fusion: per-doc max(A, B)
      Low-confidence OCR fallback: OCR→SBERT→text initial ranking, fused by max with small weight
    Returns top-K slugs.
    """
    if _IMAGE_EMB is None or _IMAGE_EMB.size == 0 or _CLIP_TEXT_EMB is None or _CLIP_TEXT_EMB.size == 0:
        return []

    # 1) encode query variants (batch once)
    variants = _image_variants(img)
    with _t("img_encode"):
        q_batch = encode_image_batch(variants)
        if q_batch is None or q_batch.size == 0:
            return []

    # 2) Branch A: image→image, per-image take max over variants, then per-doc pool
    with _t("img_img_initial"):
        sims_matrix = _IMAGE_EMB @ q_batch.T  # (num_images, V)
        sims_ii_max = sims_matrix.max(axis=1)  # (num_images,)
        per_doc_ii = _agg_pool_per_doc(sims_ii_max, _IMAGE_DOCIDS, scale=W_II, mode=POOL_MODE, k=POOL_TOPK)

    # 3) Branch B: image→text, use CLIP full text and optional visual text; max across variants and sources
    with _t("img_text_initial"):
        sims_it = _CLIP_TEXT_EMB @ q_batch.T  # (num_docs, V)
        sims_it_max = sims_it.max(axis=1)     # (num_docs,)
        if _CLIP_TEXT_EMB_VIS is not None and _CLIP_TEXT_EMB_VIS.size:
            sims_it_vis = _CLIP_TEXT_EMB_VIS @ q_batch.T
            sims_it_max = np.maximum(sims_it_max, sims_it_vis.max(axis=1))
        per_doc_it = {int(i): float(s) * W_IT for i, s in enumerate(sims_it_max)}

    # 4) Fusion by per-doc max
    scores_doc = defaultdict(float)
    for k, v in per_doc_ii.items():
        if v > scores_doc[k]:
            scores_doc[k] = v
    for k, v in per_doc_it.items():
        if v > scores_doc[k]:
            scores_doc[k] = v

    # 5) Optional OCR fallback (low-confidence)
    if (_TEXT_EMB is not None and _TEXT_EMB.size and _SBERT is not None and scores_doc):
        ranked_peek = sorted(scores_doc.items(), key=lambda x: x[1], reverse=True)[:2]
        top_val = ranked_peek[0][1] if ranked_peek else 0.0
        gap = (ranked_peek[0][1] - ranked_peek[1][1]) if len(ranked_peek) >= 2 else 1.0
        if top_val < IMG_OCR_THRESHOLD or gap < IMG_OCR_GAP:
            ocr_txt = _ocr_text(variants[0])
            if ocr_txt:
                qvec_txt = _cached_text_vec(ocr_txt)
                if qvec_txt is not None:
                    top_txt = _topk_from_matrix(qvec_txt, _TEXT_EMB, _TEXT_IDS, k=20, faiss_index=_FAISS_TEXT)
                    for sid, sim in top_txt:
                        di = _SLUG2ROW.get(sid, -1) if _SLUG2ROW else -1
                        if di >= 0:
                            scores_doc[di] = max(scores_doc.get(di, 0.0), float(sim) * IMG_OCR_WEIGHT)

    if not scores_doc:
        return []

    ranked = sorted(scores_doc.items(), key=lambda x: x[1], reverse=True)[:K]
    results = []
    for doc_id, _ in ranked:
        try:
            results.append(_DOC_TITLES[doc_id])
        except Exception:
            continue
    return results


# --- Encoders ---
def encode_image_batch(images: List[Image.Image]) -> Optional[np.ndarray]:
    """Encode many images (query variants) in one batch; return L2-normalized float32 (V, D)."""
    _ensure_loaded()
    try:
        with torch.inference_mode():
            tensors = [_CLIP_PREPROC(im.convert("RGB")) for im in images]
            if len(tensors) == 0:
                return None
            inp = torch.stack(tensors, dim=0)  # (V, 3, 224, 224)
            feat = _CLIP_MODEL.encode_image(inp)
            vecs = feat.cpu().numpy().astype("float32")
            vecs = _l2_normalize(vecs)
            return vecs
    except Exception as e:
        logger.warning(f"encode_image_batch failed: {e}")
        return None


def encode_image(image: Image.Image) -> Optional[np.ndarray]:
    _ensure_loaded()
    try:
        with torch.inference_mode():
            inp = _CLIP_PREPROC(image.convert("RGB")).unsqueeze(0)
            feat = _CLIP_MODEL.encode_image(inp)
            vec = feat.cpu().numpy().astype("float32")[0]
            vec = _l2_normalize(vec)
            return vec
    except Exception as e:
        logger.warning(f"encode_image failed: {e}")
        return None


def encode_text(text: str) -> Optional[np.ndarray]:
    """SBERT text encoding. Return L2-normalized float32 vector."""
    _ensure_loaded()
    text = (text or "").strip()
    if not text:
        return None
    try:
        with torch.inference_mode():
            vec = _SBERT.encode([text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
            return vec.astype("float32", copy=False)
    except Exception as e:
        logger.warning(f"encode_text failed: {e}")
        return None


# --- Similarity + Ranking ---
def calculate_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
    if feature1.ndim != 1:
        feature1 = feature1.reshape(-1)
    if feature2.ndim != 1:
        feature2 = feature2.reshape(-1)
    a = _l2_normalize(feature1.astype("float32"))
    b = _l2_normalize(feature2.astype("float32"))
    return float(np.dot(a, b))


def get_top_k_ranked_items(similarities: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]


def _agg_pool_per_doc(scores: np.ndarray, doc_ids: np.ndarray, scale: float = 1.0,
                      mode: str = "topk_mean", k: int = 2) -> Dict[int, float]:
    buckets: dict[int, List[float]] = defaultdict(list)
    if scores is None or scores.size == 0 or doc_ids is None or doc_ids.size == 0:
        return {}
    s = scores.reshape(-1)
    d = doc_ids.reshape(-1)
    L = min(s.shape[0], d.shape[0])
    for i in range(L):
        di = int(d[i])
        if di >= 0:
            buckets[di].append(float(s[i]))
    out: dict[int, float] = {}
    for di, vals in buckets.items():
        if not vals:
            continue
        if mode == "lse":
            m = np.max(vals)
            pooled = m + float(np.log(np.sum(np.exp(np.array(vals) - m))))
        elif mode == "topk_mean":
            topk = sorted(vals, reverse=True)[:max(1, k)]
            pooled = float(np.mean(topk))
        else:  # max
            pooled = max(vals)
        out[di] = pooled * scale
    return out


# --- Helper: exact top-k over matrix (FAISS preferred) ---
def _topk_from_matrix(query_vec: np.ndarray, mat: np.ndarray, ids: List[str], k: int, faiss_index=None) -> List[Tuple[str, float]]:
    """
    mat: (N, D), query_vec: (D,) both L2-normalized -> cosine == dot
    Use FAISS IndexFlatIP (if available) for exact Top-K; fallback to numpy.
    Results are exactly equal to the pure numpy path.
    """
    N = mat.shape[0]
    if N == 0:
        return []
    q = _l2_normalize(query_vec.astype("float32", copy=False))
    if faiss_index is not None:
        idxs = _faiss_topk(faiss_index, q, min(k, N))
        sims = mat[idxs] @ q
        order = np.argsort(-sims)
        idxs = idxs[order]
    else:
        sims = mat @ q  # (N,)
        if k >= N:
            idxs = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, kth=min(k, N - 1))[: k * 2]
            order = top[np.argsort(-sims[top])]
            idxs = order
        idxs = idxs[:k]
    return [(ids[i], float((mat[i] @ q))) for i in idxs]


def _image_variants(image: Image.Image) -> List[Image.Image]:
    im = image.convert("RGB")
    rots = [im, im.rotate(90, expand=True), im.rotate(180, expand=True), im.rotate(270, expand=True)]
    out = list(rots)
    if ADD_MIRROR:
        out += [ImageOps.mirror(r) for r in rots]
    if ADD_GRAY:
        out.append(ImageOps.grayscale(im).convert("RGB"))
    return out


# --- Routes ---
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get(HEALTH_ENDPOINT_NAME, status_code=200)
def health():
    return {"status": "OK"}


@app.post(PREDICT_ENDPOINT_NAME, response_model=PredictionResponse)
async def predict(request: Request) -> dict:
    """
    Compatible with Vertex 'instances' and local 'query_type/query_content'.
    Returns slug (without .json) in ranked_documents.
    """
    _ensure_loaded()
    payload = await request.json()

    # 解析为 (qtype, qobj) 列表（你已有的工具函数）
    queries = _parse_payload_to_queries(payload)
    results: List[PredictionResponseItem] = [PredictionResponseItem() for _ in queries]

    # 分组：文本与图片
    text_idx, text_vals = [], []
    img_idx, img_vals = [], []
    for i, (qtype, qobj) in enumerate(queries):
        if qtype == "text":
            text_idx.append(i); text_vals.append(qobj)
        elif qtype == "image":
            img_idx.append(i); img_vals.append(qobj)

    # 文本：批处理重排（零精度损失）
    for i in text_idx:
        results[i].ranked_documents = _process_text_query(queries[i][1])

    # 图片：保持原实现
    for i, img in zip(img_idx, img_vals):
        try:
            results[i].ranked_documents = _process_image_query(img)
        except Exception:
            results[i].ranked_documents = []

    return {"predictions": results}


# --- OCR helper ---
def _ocr_text(img: Image.Image) -> str:
    if pytesseract is None or _re is None:
        return ""
    try:
        txt = pytesseract.image_to_string(img)
        txt = _re.sub(r"\s+", " ", txt or "").strip()
        if sum(c.isalnum() for c in txt) >= 12:
            return txt
    except Exception:
        pass
    return ""
