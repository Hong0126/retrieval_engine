# -*- coding: utf-8 -*-
"""
Precompute embeddings & indices for the ML6 blog post retrieval engine.

Outputs (to app/index/ by default):
  text_ids.json                     # [slug, ...] (no ".json")
  text_embeddings.npy               # (N, 384) float32, L2-normalized (SBERT)
  clip_text_embeddings.npy          # (N, 512) float32, L2-normalized (OpenCLIP, full text)
  blog_text_for_rerank.json         # {slug: "title + summary ..."}
  image_ids.json                    # [filename, ...] (basename only)
  image_embeddings.npy              # (M, 512) float32, L2-normalized (OpenCLIP)
  image_to_blog.json                # {filename: slug} (normalized slugs)
  clip_text_visual_embeddings.npy   # (N, 512) float32, L2-normalized (OpenCLIP, visual text) [optional]
  visual_text_for_clip.json         # {slug: visual_text string} [optional]
  orb_index.json (optional)         # {filename: "orb_desc/<filename>.npz"}
  orb_desc/<filename>.npz (optional) with key 'desc'

Run:
  python app/precompute.py \
    --device cpu \
    --text-batch 64 \
    --image-batch 64 \
    --with-visual-text \
    --with-orb

Assumes repository layout:
  repo_root/
    data/
      orig_data/
        blogposts/*.json
        images/*.{png,jpg,jpeg,webp}
        img_to_blog.json
    app/
      precompute.py
      main.py (will load app/index/* at runtime)
"""

from __future__ import annotations
import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
import torch

# Optional deps (we gate them)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    import open_clip
except Exception:
    open_clip = None  # type: ignore

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

# -------------------------
# Logging & determinism
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("precompute")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------
# Utils
# -------------------------
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


def _strip_html(txt: str) -> str:
    if not txt:
        return txt
    # crude remove tags
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _flatten_text_from_block(block: Any) -> str:
    """
    Best-effort extraction of visible text from a block structure of blog JSON.
    Supports keys: 'content', 'text', 'children', 'paragraphs', etc., and plain strings.
    """
    if block is None:
        return ""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        # Prefer 'content' or 'text'
        cand = block.get("content") or block.get("text")
        if isinstance(cand, str):
            return cand
        acc = []
        if isinstance(cand, list):
            for it in cand:
                acc.append(_flatten_text_from_block(it))
        # children
        if "children" in block and isinstance(block["children"], list):
            for it in block["children"]:
                acc.append(_flatten_text_from_block(it))
        # paragraphs
        if "paragraphs" in block and isinstance(block["paragraphs"], list):
            for it in block["paragraphs"]:
                acc.append(_flatten_text_from_block(it))
        return " ".join([a for a in acc if a]).strip()
    if isinstance(block, list):
        return " ".join([_flatten_text_from_block(it) for it in block]).strip()
    return ""


def _extract_visual_bits(block: Any, out: Dict[str, List[str]]):
    """Collect heading-level text and image captions/alt for visual_text.
    Store under keys: 'headings', 'captions'.
    """
    if block is None:
        return
    if isinstance(block, dict):
        btype = str(block.get("type") or block.get("block_type") or "").upper()
        # headings
        if btype in {"H1", "H2", "H3", "HEADER", "HEADING"}:
            t = block.get("content") or block.get("text") or block.get("title")
            if isinstance(t, str) and t.strip():
                out.setdefault("headings", []).append(_strip_html(t))
        # figures/images
        if btype in {"FIGURE", "IMAGE", "IMG", "FIGCAPTION"} or any(k in block for k in ("caption", "alt")):
            for key in ("caption", "figcaption", "alt", "title", "text"):
                v = block.get(key)
                if isinstance(v, str) and v.strip():
                    out.setdefault("captions", []).append(_strip_html(v))
                elif isinstance(v, list):
                    out.setdefault("captions", []).extend([_strip_html(_flatten_text_from_block(x)) for x in v])
        # dive deeper
        for k in ("content", "children", "blocks", "paragraphs"):
            v = block.get(k)
            if isinstance(v, list):
                for it in v:
                    _extract_visual_bits(it, out)
            elif isinstance(v, dict):
                _extract_visual_bits(v, out)
    elif isinstance(block, list):
        for it in block:
            _extract_visual_bits(it, out)


def _read_blog_json(fp: Path) -> Tuple[str, str, Dict[str, List[str]]]:
    """Return (title, full_text, visual_bits) from a blog JSON."""
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = ""
    if isinstance(data, dict):
        title = data.get("title") or data.get("Title") or ""
        blocks = data.get("blocks") or data.get("content") or data.get("Body") or []
    else:
        blocks = []

    texts: List[str] = []
    if isinstance(title, str) and title.strip():
        texts.append(str(title))

    if isinstance(blocks, list):
        for b in blocks:
            t = _flatten_text_from_block(b)
            t = _strip_html(t)
            if t:
                texts.append(t)

    # collect visual bits
    vb: Dict[str, List[str]] = {}
    _extract_visual_bits(blocks, vb)

    full_text = " \n".join(texts)
    return _strip_html(title or ""), full_text, vb


def _assemble_visual_text(title: str, visual_bits: Dict[str, List[str]], max_words: int = 75) -> str:
    parts: List[str] = []
    if title:
        parts.append(title)
    for sec in ("headings", "captions"):
        vals = visual_bits.get(sec, [])
        if vals:
            parts.extend(vals)
    txt = ". ".join([p for p in parts if p])
    # trim to ~75 words (CLIP 77 tokens context)
    words = re.split(r"\s+", txt)
    txt = " ".join(words[:max_words])
    return txt.strip()


def _gather_blogs(
    blog_dir: Path,
    with_visual_text: bool,
) -> Tuple[List[str], List[str], Dict[str, str], Optional[List[str]]]:
    """Scan blogposts/*.json and return:
      slugs:   [slug,...]   (filename stem normalized)
      corpus:  [long_text,...] (for SBERT / CLIP full text embedding)
      for_ce:  {slug: rerank_text}  (title + trimmed summary for CrossEncoder)
      visual_texts: Optional[List[str]] (title + headings + captions)
    """
    fps = sorted([p for p in blog_dir.glob("*.json") if p.is_file()])
    slugs: List[str] = []
    corpus: List[str] = []
    for_ce: Dict[str, str] = {}
    visual_texts: Optional[List[str]] = [] if with_visual_text else None

    logger.info("Found %d blog JSON files", len(fps))
    for p in fps:
        slug = _normalize_slug(p.stem)
        title, full_text, vb = _read_blog_json(p)

        long_text = _strip_html(full_text)

        # For CrossEncoder rerank: title + trimmed summary (~350 words)
        words = re.split(r"\s+", long_text)
        brief = " ".join(words[:350])
        rerank_text = (title or slug).strip()
        if brief:
            rerank_text = f"{rerank_text}\n\n{brief}"

        slugs.append(slug)
        corpus.append(long_text if long_text else (title or slug))
        for_ce[slug] = rerank_text

        if with_visual_text and visual_texts is not None:
            vt = _assemble_visual_text(title, vb, max_words=75)
            visual_texts.append(vt if vt else (title or slug))

    return slugs, corpus, for_ce, visual_texts


def _load_img_to_blog(fp: Path) -> Dict[str, str]:
    if not fp.exists():
        logger.warning("img_to_blog.json not found at %s; will infer empty mapping.", fp)
        return {}
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize to filename (basename) -> slug (normalized)
    out: Dict[str, str] = {}
    for k, v in data.items():
        fname = Path(k).name
        out[fname] = _normalize_slug(Path(v).stem if isinstance(v, str) else str(v))
    return out


def _list_images(img_dir: Path) -> List[str]:
    # Return basenames only
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    allp = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()])
    return [p.name for p in allp]


def _pil_open_rgb(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        # corrupted/eval images might be grayscale; convert consistently
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        if img.mode == "L":
            # keep luminance but expand to RGB (OpenCLIP expects 3 channels)
            img = ImageOps.colorize(img, black="black", white="white").convert("RGB")
        else:
            img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError) as e:
        logger.warning("Failed to open image %s: %s", path, e)
        return None


# -------------------------
# Embedding workers
# -------------------------
def sbert_encode(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    assert SentenceTransformer is not None, "sentence-transformers not installed."
    logger.info("Loading SBERT %s on %s ...", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
    ).astype("float32")
    emb = _l2_normalize(emb.astype("float32", copy=False))
    logger.info("SBERT embeddings: %s", emb.shape)
    return emb


def openclip_text_encode(
    texts: List[str],
    arch: str,
    pretrained: str,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    assert open_clip is not None, "open_clip not installed."
    logger.info("Loading OpenCLIP %s/%s on %s ...", arch, pretrained, device)
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(arch)
    model.eval()

    vecs: List[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            toks = tokenizer(chunk)
            toks = toks.to(device)
            feat = model.encode_text(toks)
            v = feat.detach().cpu().numpy().astype("float32")
            vecs.append(v)

    out = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 512), dtype="float32")
    out = _l2_normalize(out)
    logger.info("OpenCLIP text embeddings: %s", out.shape)
    return out


def openclip_image_encode(
    image_paths: List[Path],
    arch: str,
    pretrained: str,
    device: str,
    batch_size: int = 64,
) -> Tuple[List[str], np.ndarray]:
    assert open_clip is not None, "open_clip not installed."
    logger.info("Loading OpenCLIP %s/%s on %s for images...", arch, pretrained, device)
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
    model.eval()

    basenames: List[str] = []
    tensors: List[torch.Tensor] = []

    for p in image_paths:
        img = _pil_open_rgb(p)
        if img is None:
            continue
        basenames.append(p.name)
        tensors.append(preprocess(img))

    if not tensors:
        logger.warning("No images could be loaded. Image embeddings will be empty.")
        return [], np.zeros((0, 512), dtype="float32")

    X = torch.stack(tensors, dim=0)  # (M, 3, 224, 224)
    vecs: List[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device)
            feat = model.encode_image(xb)
            v = feat.detach().cpu().numpy().astype("float32")
            vecs.append(v)

    out = np.concatenate(vecs, axis=0)
    out = _l2_normalize(out)
    logger.info("OpenCLIP image embeddings: %s", out.shape)
    return basenames, out


def compute_orb_desc_for_images(
    image_paths: List[Path],
    out_dir: Path,
    image_basenames: Optional[List[str]] = None,
    nfeatures: int = 600,
) -> Dict[str, str]:
    """
    Save ORB descriptors to out_dir/'orb_desc'/<filename>.npz with key 'desc'.
    Return map: filename -> "orb_desc/<filename>.npz".
    """
    if cv2 is None:
        logger.warning("OpenCV not installed; skipping ORB descriptors.")
        return {}

    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    desc_root = out_dir / "orb_desc"
    desc_root.mkdir(parents=True, exist_ok=True)

    name_set = set(image_basenames) if image_basenames else None

    mapping: Dict[str, str] = {}
    for p in image_paths:
        if name_set is not None and p.name not in name_set:
            continue
        try:
            img = _pil_open_rgb(p)
            if img is None:
                continue
            gray = np.array(img.convert("L"))
            kps, desc = orb.detectAndCompute(gray, None)
            if desc is None or desc.size == 0:
                continue
            out_fp = desc_root / f"{p.name}.npz"
            if desc.shape[0] > 2000:
                desc = desc[:2000]
            np.savez_compressed(out_fp, desc=desc)
            mapping[p.name] = f"orb_desc/{p.name}.npz"
        except Exception as e:
            logger.warning("ORB failed for %s: %s", p, e)

    logger.info("ORB descriptors saved for %d images", len(mapping))
    return mapping


# -------------------------
# IO helpers
# -------------------------
def _save_json(obj: Any, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_npy(arr: np.ndarray, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.save(fp, arr.astype("float32", copy=False))


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Precompute features for ML6 blogpost retrieval.")
    repo_root = Path(__file__).resolve().parents[1]
    default_data = repo_root / "data"
    default_out = Path(__file__).parent / "index"

    parser.add_argument("--data-root", type=Path, default=default_data, help="Path to repo_root/data")
    parser.add_argument("--out-dir", type=Path, default=default_out, help="Where to store app/index/*")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        choices=["cpu", "cuda"], help="Device for model inference")
    parser.add_argument("--sbert-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--clip-arch", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--text-batch", type=int, default=64)
    parser.add_argument("--image-batch", type=int, default=64)
    parser.add_argument("--with-orb", action="store_true", help="Also precompute ORB descriptors if OpenCV is available")
    parser.add_argument("--skip-text", action="store_true", help="Skip text embeddings")
    parser.add_argument("--skip-image", action="store_true", help="Skip image embeddings")
    parser.add_argument("--with-visual-text", action="store_true", help="Also export visual-text embeddings for CLIP")

    args = parser.parse_args()

    blog_dir = args.data_root / "orig_data" / "blogposts"
    img_dir = args.data_root / "orig_data" / "images"
    map_fp = args.data_root / "orig_data" / "img_to_blog.json"
    out_dir = args.out_dir

    logger.info("Data root: %s", args.data_root)
    logger.info("Blog dir : %s", blog_dir)
    logger.info("Image dir: %s", img_dir)
    logger.info("Out dir  : %s", out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Gather blog texts
    # -------------------------
    slugs, corpus, rerank_map, visual_texts = _gather_blogs(blog_dir, with_visual_text=args.with_visual_text)
    if not args.skip_text:
        assert len(slugs) == len(corpus), "Text corpus and slug length mismatch."
        # SBERT
        sbert_vecs = sbert_encode(
            texts=corpus,
            model_name=args.sbert_model,
            device=args.device,
            batch_size=args.text_batch,
        ) if len(corpus) > 0 else np.zeros((0, 384), dtype="float32")

        # OpenCLIP text (full text)
        clip_txt_vecs = openclip_text_encode(
            texts=corpus,
            arch=args.clip_arch,
            pretrained=args.clip_pretrained,
            device=args.device,
            batch_size=max(8, args.text_batch),
        ) if len(corpus) > 0 else np.zeros((0, 512), dtype="float32")

        # Save text indices
        _save_json(slugs, out_dir / "text_ids.json")
        _save_npy(sbert_vecs, out_dir / "text_embeddings.npy")
        _save_npy(clip_txt_vecs, out_dir / "clip_text_embeddings.npy")
        _save_json(rerank_map, out_dir / "blog_text_for_rerank.json")
        logger.info("Saved text indices: N=%d", len(slugs))

        # Visual text (optional)
        if args.with_visual_text and visual_texts is not None:
            clip_vtxt_vecs = openclip_text_encode(
                texts=visual_texts,
                arch=args.clip_arch,
                pretrained=args.clip_pretrained,
                device=args.device,
                batch_size=max(8, args.text_batch),
            ) if len(visual_texts) > 0 else np.zeros((0, 512), dtype="float32")
            _save_npy(clip_vtxt_vecs, out_dir / "clip_text_visual_embeddings.npy")
            vis_map = {slug: vt for slug, vt in zip(slugs, visual_texts)}
            _save_json(vis_map, out_dir / "visual_text_for_clip.json")
            logger.info("Saved visual-text embeddings: N=%d", len(visual_texts))
    else:
        logger.info("Skip text embeddings as requested.")

    # -------------------------
    # Gather images & mapping
    # -------------------------
    img_to_blog = _load_img_to_blog(map_fp)
    all_imgs = _list_images(img_dir)

    # Only keep images that appear in mapping (for consistency with main.py behavior)
    if img_to_blog:
        kept = [fn for fn in all_imgs if fn in img_to_blog]
        if len(kept) < len(all_imgs):
            logger.info("Keeping %d mapped images out of %d total.", len(kept), len(all_imgs))
        all_imgs = kept

    image_paths = [(img_dir / fn) for fn in all_imgs]

    # Normalize mapping target to present slugs
    slug_set = set(slugs)
    fixed_map: Dict[str, str] = {}
    unknown = 0
    for k, v in img_to_blog.items():
        vv = _normalize_slug(v)
        if vv not in slug_set:
            unknown += 1
        fixed_map[k] = vv
    if unknown > 0:
        logger.warning("There are %d images mapped to slugs not present in blog set.", unknown)

    if not args.skip_image:
        basenames, img_vecs = openclip_image_encode(
            image_paths=image_paths,
            arch=args.clip_arch,
            pretrained=args.clip_pretrained,
            device=args.device,
            batch_size=args.image_batch,
        ) if len(image_paths) > 0 else ([], np.zeros((0, 512), dtype="float32"))

        # Save image indices
        _save_json(basenames, out_dir / "image_ids.json")
        _save_npy(img_vecs, out_dir / "image_embeddings.npy")

        # image_to_blog.json should only include those actually embedded
        final_map = {fn: fixed_map.get(fn, "") for fn in basenames}
        _save_json(final_map, out_dir / "image_to_blog.json")

        logger.info("Saved image indices: M=%d", len(basenames))

        # Optional ORB
        if args.with_orb:
            mapping = compute_orb_desc_for_images(
                image_paths=image_paths,
                out_dir=out_dir,
                image_basenames=basenames,
                nfeatures=600,
            )
            if mapping:
                _save_json(mapping, out_dir / "orb_index.json")
            else:
                logger.info("No ORB descriptors saved.")
    else:
        logger.info("Skip image embeddings as requested.")

    # -------------------------
    # Sanity summary
    # -------------------------
    expected = [
        "text_ids.json", "text_embeddings.npy", "clip_text_embeddings.npy", "blog_text_for_rerank.json",
        "image_ids.json", "image_embeddings.npy", "image_to_blog.json"
    ]
    missing = [e for e in expected if not (out_dir / e).exists()]
    if missing:
        logger.warning("Missing expected outputs (may be okay if you used --skip-*): %s", missing)

    logger.info("Precompute finished. Outputs under: %s", out_dir)


if __name__ == "__main__":
    main()
