# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Arabic PDF Layout + OCR Model Survey
#
# Pick a PDF, render pages, run candidate **layout** models, crop regions, run
# candidate **OCR** models on each crop, eyeball results, pick winners,
# assemble a markdown file.
#
# **How to use:**
# 1. Set `PDF_PATH` and `PAGES` in the Config cell.
# 2. Toggle which layout/OCR models to try in the registry cells.
# 3. Run cells top-to-bottom. Results cached on disk in `notebooks/.cache/`.
# 4. After eyeballing, set `WIN_LAYOUT` + `WIN_OCR` in the **Pick winners** cell.
# 5. Run the assemble cell — produces `out/transcript.md`.

# %% [markdown]
# ## 1. Imports & device

# %%
import os, sys, json, hashlib, time, traceback, gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Any
import torch
from PIL import Image, ImageDraw, ImageFont
from IPython.display import HTML, display

NB_DIR = Path.cwd()
if NB_DIR.name != "notebooks":
    NB_DIR = Path("notebooks").resolve() if (Path.cwd() / "notebooks").exists() else NB_DIR
CACHE = NB_DIR / ".cache"
OUT = NB_DIR / "out"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

print(f"device={DEVICE} dtype={DTYPE} cache={CACHE} hf_token={'set' if HF_TOKEN else 'missing'}")

# %% [markdown]
# ## 2. Config — PDF + pages

# %%
PDF_PATH = Path("/home/abdullah/Downloads/بناء الأجيال_Foulabook.com_.pdf")
PAGES = [1, 26, 30, 46, 48, 55, 77, 82, 90, 159, 183, 193]  # 12 pages: 1, 30, +10 random (seed=42)
DPI = 200
CROP_PAD = 8

assert PDF_PATH.exists(), f"PDF not found: {PDF_PATH}"
print(f"pdf={PDF_PATH.name} pages={PAGES} dpi={DPI}")

# %% [markdown]
# ## 3. Cache helpers

# %%
def _hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        elif isinstance(p, Path):
            h.update(str(p).encode()); h.update(str(p.stat().st_mtime).encode())
        else:
            h.update(str(p).encode())
    return h.hexdigest()[:16]

def cache_path(kind: str, key: str, ext: str) -> Path:
    d = CACHE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.{ext}"

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def to_inference(m):
    m.train(False)
    return m

# %% [markdown]
# ## 4. Render PDF pages → PNG (cached)

# %%
from pdf2image import convert_from_path

def render_page(pdf: Path, page: int, dpi: int) -> Image.Image:
    key = _hash(pdf, page, dpi)
    png = cache_path("pages", key, "png")
    if png.exists():
        return Image.open(png).convert("RGB")
    imgs = convert_from_path(str(pdf), dpi=dpi, first_page=page, last_page=page)
    img = imgs[0].convert("RGB")
    img.save(png)
    return img

PAGE_IMAGES: dict[int, Image.Image] = {p: render_page(PDF_PATH, p, DPI) for p in PAGES}
for p, im in PAGE_IMAGES.items():
    print(f"page {p}: {im.size}")
display(PAGE_IMAGES[PAGES[0]].resize((PAGE_IMAGES[PAGES[0]].width // 2, PAGE_IMAGES[PAGES[0]].height // 2)))

# %% [markdown]
# ## 5. Layout model registry

# %%
LAYOUT_REGISTRY: dict[str, dict] = {}

def register_layout(name: str):
    def deco(fn):
        LAYOUT_REGISTRY[name] = {"name": name, "loader": fn}
        return fn
    return deco

@register_layout("full-page")
def _load_full_page():
    """Pseudo-layout: whole page is one region. Use with vision-LLM OCR
    that handles internal layout."""
    def infer(image: Image.Image):
        return [{"bbox": [0, 0, image.width, image.height], "label": "page"}]
    return infer

@register_layout("doclayout-yolo")
def _load_doclayout_yolo():
    from doclayout_yolo import YOLOv10
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    )
    model = YOLOv10(path)
    def infer(image: Image.Image):
        res = model.predict(image, imgsz=1024, conf=0.25, device=DEVICE)
        out = []
        for r in res:
            names = r.names
            for box, cls in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                out.append({"bbox": [int(v) for v in box], "label": names[int(cls)]})
        return out
    return infer

@register_layout("surya")
def _load_surya_layout():
    from surya.layout import LayoutPredictor
    from surya.foundation import FoundationPredictor
    fp = FoundationPredictor()
    pred = LayoutPredictor(fp)
    def infer(image: Image.Image):
        res = pred([image])[0]
        out = []
        for b in res.bboxes:
            out.append({"bbox": [int(v) for v in b.bbox], "label": b.label})
        return out
    return infer

print("registered layout models:", list(LAYOUT_REGISTRY.keys()))

# %% [markdown]
# ## 6. OCR model registry

# %%
OCR_REGISTRY: dict[str, dict] = {}

def register_ocr(name: str):
    def deco(fn):
        OCR_REGISTRY[name] = {"name": name, "loader": fn}
        return fn
    return deco

def _vlm_generate(model_box: list, proc, image, prompt, max_new_tokens=768):
    """Generate with OOM fallback to CPU. model_box is [model] so we can swap."""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    model = model_box[0]
    target_dev = next(model.parameters()).device.type
    try:
        inputs = proc(text=[text], images=[image], return_tensors="pt").to(target_dev)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
    except torch.cuda.OutOfMemoryError:
        print("  [OOM on GPU — moving model to CPU and retrying]")
        free_gpu()
        model = model.cpu().to(torch.float32)
        model_box[0] = model
        inputs = proc(text=[text], images=[image], return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

@register_ocr("qwen2-vl-2b")
def _load_qwen2_vl_2b():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    repo = "Qwen/Qwen2-VL-2B-Instruct"
    proc = AutoProcessor.from_pretrained(
        repo, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
    )
    model = to_inference(Qwen2VLForConditionalGeneration.from_pretrained(
        repo, torch_dtype=DTYPE
    ).to(DEVICE))
    box = [model]
    PROMPT = "Extract all text from this image as plain Arabic. Output only the text, no commentary."
    def infer(image: Image.Image) -> str:
        return _vlm_generate(box, proc, image, PROMPT)
    def _cleanup():
        try: box[0].cpu()
        except Exception: pass
        free_gpu()
    infer._cleanup = _cleanup
    return infer

@register_ocr("qari-ocr")
def _load_qari_ocr():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    repo = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"
    proc = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
    )
    model = to_inference(Qwen2VLForConditionalGeneration.from_pretrained(
        repo, torch_dtype=DTYPE
    ).to(DEVICE))
    box = [model]
    PROMPT = "Below is the image of one page of a document, please read the content and convert it into plain text."
    def infer(image: Image.Image) -> str:
        return _vlm_generate(box, proc, image, PROMPT)
    def _cleanup():
        try: box[0].cpu()
        except Exception: pass
        free_gpu()
    infer._cleanup = _cleanup
    return infer

@register_ocr("surya")
def _load_surya_ocr():
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    fp = FoundationPredictor()
    rec = RecognitionPredictor(fp)
    det = DetectionPredictor()
    def infer(image: Image.Image) -> str:
        res = rec([image], det_predictor=det)[0]
        return "\n".join(line.text for line in res.text_lines)
    def _cleanup():
        free_gpu()
    infer._cleanup = _cleanup
    return infer

# --- Tesseract (CPU baseline; fast, classic OCR) ---
@register_ocr("tesseract-ara")
def _load_tesseract_ara():
    import pytesseract
    os.environ["TESSDATA_PREFIX"] = "/home/abdullah/.tessdata"
    def infer(image: Image.Image) -> str:
        return pytesseract.image_to_string(image, lang="ara").strip()
    return infer

# --- EasyOCR (Arabic) ---
@register_ocr("easyocr-ara")
def _load_easyocr_ara():
    import easyocr
    import numpy as np
    reader = easyocr.Reader(["ar"], gpu=(DEVICE == "cuda"))
    def infer(image: Image.Image) -> str:
        arr = np.array(image)
        results = reader.readtext(arr, detail=1, paragraph=True)
        return "\n".join(r[1] for r in results)
    def _cleanup():
        free_gpu()
    infer._cleanup = _cleanup
    return infer

# --- PaddleOCR Arabic ---
@register_ocr("paddleocr-ara")
def _load_paddleocr_ara():
    from paddleocr import PaddleOCR
    import numpy as np
    # PaddleOCR 3.x needs explicit ocr_version for non-default langs; "arabic" via PP-OCRv3
    # PaddleOCR 3.x: Arabic via "ar" lang code → arabic_PP-OCRv3_mobile_rec
    ocr = PaddleOCR(lang="ar", ocr_version="PP-OCRv3",
                    use_textline_orientation=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False)
    def infer(image: Image.Image) -> str:
        arr = np.array(image)
        result = ocr.predict(arr)
        lines = []
        for page in result:
            texts = page.get("rec_texts", []) if isinstance(page, dict) else getattr(page, "rec_texts", [])
            lines.extend(texts)
        return "\n".join(lines)
    def _cleanup():
        free_gpu()
    infer._cleanup = _cleanup
    return infer

# --- Qwen2.5-VL-3B-Instruct (newer base VLM) — runs on CPU on 6GB GPU ---
@register_ocr("qwen2.5-vl-3b")
def _load_qwen25_vl_3b():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    repo = "Qwen/Qwen2.5-VL-3B-Instruct"
    proc = AutoProcessor.from_pretrained(
        repo, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
    )
    # 3B model > 5.6GB GPU — load on CPU directly
    model = to_inference(Qwen2_5_VLForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.float32
    ).to("cpu"))
    box = [model]
    PROMPT = "Extract all Arabic text from this image. Output only the text, no commentary."
    def infer(image: Image.Image) -> str:
        return _vlm_generate(box, proc, image, PROMPT)
    def _cleanup():
        free_gpu()
    infer._cleanup = _cleanup
    return infer

print("registered OCR models:", list(OCR_REGISTRY.keys()))

# %% [markdown]
# ## 7. Run layout pass (cached per page × model)

# %%
def run_layout(model_name: str, page: int, image: Image.Image) -> list[dict]:
    key = _hash(model_name, PDF_PATH, page, DPI)
    js = cache_path("layout", key, "json")
    if js.exists():
        return json.loads(js.read_text())
    print(f"[layout] loading {model_name}...")
    t0 = time.time()
    infer = LAYOUT_REGISTRY[model_name]["loader"]()
    t_load = time.time() - t0
    t0 = time.time()
    boxes = infer(image)
    t_infer = time.time() - t0
    js.write_text(json.dumps(boxes, ensure_ascii=False))
    (cache_path("layout", key, "meta.json")).write_text(json.dumps({
        "model": model_name, "page": page, "load_s": t_load, "infer_s": t_infer, "n_boxes": len(boxes)
    }))
    print(f"[layout] {model_name} page={page} load={t_load:.1f}s infer={t_infer:.1f}s boxes={len(boxes)}")
    free_gpu()
    return boxes

LAYOUT_TRY = ["full-page", "doclayout-yolo", "surya"]
LAYOUT_RESULTS: dict[tuple[str, int], list[dict]] = {}
for name in LAYOUT_TRY:
    for p in PAGES:
        LAYOUT_RESULTS[(name, p)] = run_layout(name, p, PAGE_IMAGES[p])

# %% [markdown]
# ## 8. Visualize layout overlays

# %%
def overlay_boxes(image: Image.Image, boxes: list[dict], color="red") -> Image.Image:
    im = image.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), f"{i}:{b['label']}", fill=color, font=font)
    return im

for p in PAGES:
    cols = []
    for name in LAYOUT_TRY:
        ov = overlay_boxes(PAGE_IMAGES[p], LAYOUT_RESULTS[(name, p)])
        path = OUT / f"layout_{name}_p{p}.png"
        ov.save(path)
        scaled = ov.resize((ov.width // 2, ov.height // 2))
        buf_path = OUT / f"_thumb_layout_{name}_p{p}.png"
        scaled.save(buf_path)
        cols.append((name, buf_path))
    html = "<table><tr>"
    for name, path in cols:
        html += f"<td style='text-align:center'><b>{name}</b><br><img src='{path.as_posix()}' style='max-width:500px'></td>"
    html += "</tr></table>"
    display(HTML(html))

# %% [markdown]
# ## 9. Crop regions (per layout model)

# %%
CROP_LAYOUT = "full-page"

def crop_regions(image: Image.Image, boxes: list[dict], pad: int) -> list[Image.Image]:
    crops = []
    W, H = image.size
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops

CROPS: dict[int, list[Image.Image]] = {}
for p in PAGES:
    boxes = LAYOUT_RESULTS[(CROP_LAYOUT, p)]
    CROPS[p] = crop_regions(PAGE_IMAGES[p], boxes, CROP_PAD)
    print(f"page {p}: {len(CROPS[p])} crops from {CROP_LAYOUT}")

for i, c in enumerate(CROPS[PAGES[0]][:6]):
    print(f"crop {i}: {c.size}")

# %% [markdown]
# ## 10. Run OCR per crop × model (cached)

# %%
def crop_key(image: Image.Image) -> str:
    import io
    buf = io.BytesIO(); image.save(buf, format="PNG")
    return _hash(buf.getvalue())

def run_ocr_one(model_name: str, infer: Callable, image: Image.Image) -> str:
    ck = crop_key(image)
    key = _hash(model_name, ck)
    txt = cache_path("ocr", key, "txt")
    if txt.exists():
        return txt.read_text()
    t0 = time.time()
    try:
        out = infer(image)
    except Exception as e:
        out = f"<<ERROR {model_name}: {e}>>"
        traceback.print_exc()
    dt = time.time() - t0
    txt.write_text(out)
    (cache_path("ocr", key, "meta.json")).write_text(json.dumps({
        "model": model_name, "crop_key": ck, "infer_s": dt, "len": len(out)
    }))
    return out

OCR_TRY = ["surya", "tesseract-ara", "easyocr-ara", "qari-ocr", "qwen2-vl-2b"]
# Skipped:
#   paddleocr-ara — paddlepaddle/paddlex API mismatch (set_optimization_level)
#   qwen2.5-vl-3b — too big for 6GB GPU + 32GB RAM (kernel died on CPU fp32)

OCR_RESULTS: dict[tuple[str, int, int], str] = {}

for model_name in OCR_TRY:
    print(f"\n=== {model_name} ===")
    needs_load = False
    for p in PAGES:
        for i, _ in enumerate(CROPS[p]):
            ck = crop_key(CROPS[p][i])
            if not cache_path("ocr", _hash(model_name, ck), "txt").exists():
                needs_load = True; break
        if needs_load: break
    infer = None
    if needs_load:
        t0 = time.time()
        infer = OCR_REGISTRY[model_name]["loader"]()
        print(f"loaded in {time.time()-t0:.1f}s")
    try:
        for p in PAGES:
            for i, crop in enumerate(CROPS[p]):
                if infer is None:
                    OCR_RESULTS[(model_name, p, i)] = run_ocr_one(model_name, lambda x: "", crop)
                else:
                    OCR_RESULTS[(model_name, p, i)] = run_ocr_one(model_name, infer, crop)
                preview = OCR_RESULTS[(model_name, p, i)].replace("\n", " ")[:60]
                print(f"  p{p} crop{i}: {preview}")
    finally:
        if infer is not None and hasattr(infer, "_cleanup"):
            infer._cleanup()
        del infer
        free_gpu()

# %% [markdown]
# ## 11. Side-by-side comparison (RTL Arabic display)

# %%
def render_compare(page: int) -> str:
    crops = CROPS[page]
    thumbs = []
    for i, c in enumerate(crops):
        t = OUT / f"_thumb_p{page}_c{i}.png"
        c.save(t)
        thumbs.append(t)
    style = (
        "table{border-collapse:collapse;font-family:sans-serif;}"
        "td,th{border:1px solid #888;padding:6px;vertical-align:top;}"
        ".ar{direction:rtl;text-align:right;font-family:'Amiri','Noto Naskh Arabic',serif;font-size:14px;max-width:400px;}"
        "img{max-width:280px;}"
    )
    html = f"<style>{style}</style><table><thead><tr><th>#</th><th>crop</th>"
    for m in OCR_TRY:
        html += f"<th>{m}</th>"
    html += "</tr></thead><tbody>"
    for i, c in enumerate(crops):
        html += f"<tr><td>{i}</td><td><img src='{thumbs[i].as_posix()}'></td>"
        for m in OCR_TRY:
            txt = OCR_RESULTS.get((m, page, i), "")
            html += f"<td class='ar'>{txt[:500]}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

for p in PAGES:
    display(HTML(f"<h3>Page {p} — layout: {CROP_LAYOUT}</h3>"))
    display(HTML(render_compare(p)))

# %% [markdown]
# ## 11b. Summary table — chars + speed per model per page

# %%
def summary_table() -> str:
    rows = []
    rows.append("| page | " + " | ".join(OCR_TRY) + " |")
    rows.append("|---|" + "---|" * len(OCR_TRY))
    for p in PAGES:
        cells = []
        for m in OCR_TRY:
            txt = OCR_RESULTS.get((m, p, 0), "")
            n = len(txt)
            err = "ERR" if txt.startswith("<<ERROR") else ""
            cells.append(f"{n}{('  ' + err) if err else ''}")
        rows.append(f"| {p} | " + " | ".join(cells) + " |")
    return "\n".join(rows)

print(summary_table())
(OUT / "summary.md").write_text(summary_table())

# %% [markdown]
# ## 12. Pick winners & assemble markdown

# %%
WIN_LAYOUT = "full-page"
WIN_OCR    = "surya"   # surya wins on dense body text (page 10); qari best on simple titles

def assemble_md(pages: list[int], layout: str, ocr: str) -> str:
    out = []
    for p in pages:
        out.append(f"\n\n<!-- page {p} -->\n")
        boxes = LAYOUT_RESULTS[(layout, p)]
        crops = CROPS[p] if (layout == CROP_LAYOUT) else crop_regions(PAGE_IMAGES[p], boxes, CROP_PAD)
        for i, _ in enumerate(crops):
            text = OCR_RESULTS.get((ocr, p, i), "")
            if text.strip():
                out.append(text.strip() + "\n")
    return "".join(out)

md = assemble_md(PAGES, WIN_LAYOUT, WIN_OCR)
md_path = OUT / f"transcript_{WIN_LAYOUT}_{WIN_OCR}.md"
md_path.write_text(md)
print(f"wrote {md_path} ({len(md)} chars)")
print("\n--- preview ---\n")
print(md[:2000])

# %% [markdown]
# ## 13. Export winners → models.toml snippet

# %%
toml_snippet = f'''# Survey winners (auto-generated)
[layout]
model = "{WIN_LAYOUT}"

[ocr]
model = "{WIN_OCR}"
'''
(OUT / "winners.toml").write_text(toml_snippet)
print(toml_snippet)
