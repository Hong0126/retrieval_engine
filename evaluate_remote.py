# eval_remote.py
# Run evaluation against local FastAPI or a Vertex AI endpoint.
import argparse, base64, json, os, re, subprocess, time, statistics
from pathlib import Path
import requests

def normalize_slug(s: str) -> str:
    s = (s or "").strip().lower()
    if s.endswith(".json"): s = s[:-5]
    return s

def get_token() -> str:
    return subprocess.check_output(["gcloud","auth","print-access-token"], text=True).strip()

def build_vertex_url(endpoint_id: str) -> str:
    m = re.search(r"/locations/([^/]+)/", endpoint_id)
    if not m:
        raise ValueError("Cannot parse region from endpoint id. Use full resource name like projects/.../locations/<region>/endpoints/<id>")
    region = m.group(1)
    return f"https://{region}-aiplatform.googleapis.com/v1/{endpoint_id}:predict"

def post_json(url: str, body: dict, token: str | None, session: requests.Session) -> tuple[float, dict]:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    t0 = time.perf_counter()
    r = session.post(url, json=body, headers=headers, timeout=90)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return dt_ms, r.json()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="app/data/eval_data/evaluation_data.json", help="evaluation json")
    p.add_argument("--images-dir", default="data/eval_data/eval_images", help="dir for image files")
    p.add_argument("--endpoint-id", help="Vertex full resource name: projects/<PID>/locations/<REGION>/endpoints/<ID>")
    p.add_argument("--local", help="Local predict URL, e.g. http://localhost:8080/predict")
    args = p.parse_args()

    # choose URL & token
    if args.local:
        url = args.local.rstrip("/")
        token = None
        print(f"Target: LOCAL {url}")
    elif args.endpoint_id:
        url = build_vertex_url(args.endpoint_id)
        token = get_token()
        print(f"Target: VERTEX {url}")
    else:
        raise SystemExit("Provide either --local or --endpoint-id")

    # images dir (fallback)
    img_dir = Path(args.images_dir)
    if not img_dir.exists():
        alt = Path("app/data/eval_data/eval_images")
        if alt.exists():
            img_dir = alt
    if not img_dir.exists():
        print(f"WARNING: images dir not found: {args.images_dir}")

    # load eval data
    with open(args.file, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    total = 0
    success = 0
    rr_sum = 0.0
    latencies = []
    s = requests.Session()

    print("\n--- Running Tests (MRR@3) ---")
    for it in eval_items:
        qid = it.get("query_id")
        qtype = (it.get("query_type") or "").lower()
        qcontent = it.get("query_content")
        expected = normalize_slug(it.get("relevant_doc_title") or "")

        # build request
        if qtype == "text":
            body = {"instances": [{"text_input": qcontent}]}
        elif qtype == "image":
            img_path = img_dir / qcontent
            if not img_path.exists():
                print(f"Test '{qid}': ERROR - image not found: {img_path}")
                total += 1
                continue
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            body = {"instances": [{"image_bytes": {"b64": b64}}]}
        else:
            print(f"Test '{qid}': ERROR - unknown query_type: {qtype}")
            total += 1
            continue

        try:
            dt, resp = post_json(url, body, token, s)
            latencies.append(dt)
            preds = resp.get("predictions", [])
            ranked = preds[0].get("ranked_documents", []) if preds else []
            top3 = [normalize_slug(x) for x in ranked][:3]
            # rank
            try:
                r = next(i for i, x in enumerate(top3) if x == expected)
                rr = 1.0 / (r + 1)
                success += 1
                rr_sum += rr
                print(f"Test '{qid}': PASS (Rank {r+1}, {dt:.1f} ms)")
            except StopIteration:
                print(f"Test '{qid}': FAIL - Expected '{expected}', Top3 {top3} ({dt:.1f} ms)")
            total += 1
        except Exception as e:
            print(f"Test '{qid}': ERROR - {e}")
            total += 1

    # summary
    import math
    mrr = rr_sum / total if total else 0.0
    pr = (success / total * 100.0) if total else 0.0
    print("\n--- Evaluation Results ---")
    print(f"Total Tests: {total}")
    print(f"Successful Tests: {success}")
    print(f"Pass Rate: {pr:.2f}%")
    print(f"MRR@3: {mrr:.4f}")
    if latencies:
        lat_sorted = sorted(latencies)
        p50 = statistics.median(lat_sorted)
        p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
        print(f"Latency p50={p50:.1f} ms, p95={p95:.1f} ms")

if __name__ == "__main__":
    main()
