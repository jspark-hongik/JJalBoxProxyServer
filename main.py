# main.py
# 실행: uvicorn main:app --host 0.0.0.0 --port 8000
import os, base64, io, time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_PATH"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "")
GEMINI_BASE = os.getenv("GEMINI_BASE_URL", "")

# 모델명은 환경변수로 주입 권장
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "")

app = FastAPI(title="Image Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인 제한 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

def _png_bytes(img_bytes: bytes) -> bytes:
    """임의 포맷 바이트를 PNG로 변환(일관성 보장). 실패 시 원본 반환."""
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return img_bytes

def _http_err(resp: requests.Response):
    try:
        detail = resp.text[:800]
    except Exception:
        detail = f"status={resp.status_code}"
    raise HTTPException(status_code=resp.status_code, detail=detail)

@app.post("/v1/images/generate")
def generate_image(
    provider: str = Form(...),           # "gpt" | "gemini"
    mode: str = Form(...),               # "text2image" | "edit"
    prompt: str = Form(...),
    size: str = Form("1024x1024"),
    image: Optional[UploadFile] = File(None),
):
    try:
        if provider not in ("gpt", "gemini"):
            raise HTTPException(400, "provider must be 'gpt' or 'gemini'")
        if mode not in ("text2image", "edit"):
            raise HTTPException(400, "mode must be 'text2image' or 'edit'")
        if mode == "edit" and image is None:
            raise HTTPException(400, "image is required for edit mode")

        if provider == "gpt":
            return _handle_gpt(mode, prompt, size, image)
        else:
            return _handle_gemini(mode, prompt, size, image)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _handle_gpt(mode: str, prompt: str, size: str, image: Optional[UploadFile]):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI key missing")

    if mode == "text2image":
        url = f"{OPENAI_BASE}/images/generations"
        payload = {"model": OPENAI_IMAGE_MODEL, "prompt": prompt, "size": size}
        r = requests.post(url, json=payload, headers=headers, timeout=90)
    else:
        url = f"{OPENAI_BASE}/images/edits"
        # OpenAI 편집은 multipart 필요 + MIME 교정 (jpeg/png/webp 이외는 PNG로 변환)
        files = {
            "model": (None, OPENAI_IMAGE_MODEL),
            "prompt": (None, prompt),
            "size": (None, size),
        }

        img_bytes = image.file.read()
        mime = (image.content_type or "").lower()
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            # MIME이 허용되지 않으면 PNG로 변환
            try:
                im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                mime = "image/png"
                filename = "input.png"
            except Exception as e:
                raise HTTPException(400, f"Unsupported image type and failed to convert to PNG: {e}")
        else:
            # 적절한 확장자 부여
            ext = ".jpg" if mime == "image/jpeg" else ".png" if mime == "image/png" else ".webp"
            filename = f"input{ext}"

        files["image"] = (filename, img_bytes, mime)
        r = requests.post(url, files=files, headers=headers, timeout=90)

    if r.status_code // 100 != 2:
        _http_err(r)

    data0 = r.json()["data"][0]
    # url 우선, 없으면 b64_json
    if "url" in data0 and data0["url"]:
        img_resp = requests.get(data0["url"], timeout=90)
        if img_resp.status_code // 100 != 2:
            _http_err(img_resp)
        png = _png_bytes(img_resp.content)
    else:
        b64 = data0.get("b64_json")
        if not b64:
            raise HTTPException(502, "OpenAI response has no url/b64_json")
        raw = base64.b64decode(b64)
        png = _png_bytes(raw)

    return StreamingResponse(io.BytesIO(png), media_type="image/png")

def _handle_gemini(mode: str, prompt: str, size: str, image: Optional[UploadFile]):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Gemini key missing")

    url = f"{GEMINI_BASE}/models/{GEMINI_IMAGE_MODEL}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

    if mode == "text2image":
        body = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
    else:
        img_b = image.file.read()
        b64 = base64.b64encode(img_b).decode("utf-8")
        body = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/png", "data": b64}}
                ]
            }]
        }

    r = requests.post(url, json=body, headers=headers, timeout=90)
    if r.status_code // 100 != 2:
        _http_err(r)

    resp = r.json()
    try:
        parts = resp["candidates"][0]["content"]["parts"]
    except Exception:
        raise HTTPException(502, "Gemini response missing candidates/content/parts")

    # 첫 inlineData 찾기
    raw = None
    for p in parts:
        blob = p.get("inline_data") or p.get("inlineData")
        if blob and blob.get("data"):
            raw = base64.b64decode(blob["data"])
            break
    if raw is None:
        raise HTTPException(502, "Gemini response has no image data")

    png = _png_bytes(raw)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")
