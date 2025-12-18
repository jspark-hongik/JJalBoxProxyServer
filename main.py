# main.py
# 실행: uvicorn main:app --host 0.0.0.0 --port 8000

# ==========================================
# 1. Enum / Import / 환경 변수 로딩
# ==========================================
from enum import Enum
from typing import Optional, List, Any

import os, base64, io
import requests
from openai import OpenAI
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from PIL import Image

# 디버깅
import logging

logger = logging.getLogger("image_debug")
logging.basicConfig(level=logging.INFO)

def _debug_log_images(tag: str, images: list[UploadFile] | None):
    """
    업로드된 images 리스트 상태를 로그로 찍어보는 디버그 함수.
    """
    if not images:
        logger.info(f"[{tag}] images is None or empty")
        return

    logger.info(f"[{tag}] images count = {len(images)}")
    for idx, upload in enumerate(images):
        logger.info(
            f"[{tag}] image[{idx}]: "
            f"filename={upload.filename!r}, "
            f"content_type={upload.content_type!r}"
        )


# Provider 선택 (프론트 enum과 동일)
class Provider(str, Enum):
    GPT = "gpt"
    GEMINI = "gemini"
    SNOW_NIGHT = "snow_night"
    PIXEL_ART = "pixel_art"
    AC_STYLE = "ac_style"

# 환경 변수 로딩
# load_dotenv(os.getenv("ENV_PATH")) # 로컬용, AWS에서는 콘솔에서 env 값 등록
load_dotenv() # AWS용
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "")
GEMINI_BASE = os.getenv("GEMINI_BASE_URL", "")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "")



# FastAPI 앱 및 CORS 설정
app = FastAPI(title="Image Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # TODO: 배포 시 도메인 제한
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. 공통 유틸 함수
# ==========================================

def _png_bytes(img_bytes: bytes) -> bytes:
    """
    임의 포맷 바이트를 PNG로 변환(일관성 보장).
    변환 실패 시 원본 바이트 그대로 반환.
    """
    try:
        # 1) 이미지 열기
        im = Image.open(io.BytesIO(img_bytes))

        # 2) RGBA 사용 → 투명도 포함 PNG 출력을 항상 보장
        im = im.convert("RGBA")

        # 3) PNG로 저장
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    except Exception as e:
        # 변환 실패 시 원본 그대로
        # (예: 이미지가 아니거나 손상된 경우)
        return img_bytes

# ==========================================
# 3. 스타일 프롬프트 헬퍼
#    (Provider별 스타일 설명을 프롬프트에 얹는 역할)
# ==========================================

def _style_prompt_snow_night() -> str:
    return (
        """I will attach a photo of a person, so please transform the image of that person according to the following description. 
            Recreate a cinematic image composed of three vertically arranged cuts, featuring a person in a snowy, blue-hour landscape, wearing an oversized black coat and a vibrant cobalt blue knit scarf, holding a small wildflower bouquet.
The first cut is dimly lit, capturing the person from behind, looking out at a wide, snow-covered plain with distant, dark mountains. The second cut is slightly brighter, showing the person in a mid-shot, gazing upwards by a flowing, snow-banked river with a dense pine forest in the background. The third cut is dimly lit, an extreme close-up focused solely on the person's face and the blue scarf, softly illuminated by a frontal light source, with large, out-of-focus snowflakes in the foreground. Each cut uses a shallow depth of field against the cool blue twilight."""
    )

def _style_prompt_pixel_art() -> str:
    return (
        "Study the pixel art style of Everskies, and imitate the way it depicts body shape, facial features and expressions, clothing, and hairstyle. "
        "Using the hairstyle, outfit, and accessories of the person in the attached image, create a full-body character illustration. "
        "The background should be transparent (PNG), and only the complete character should be included. "
        "The character should be full size and must not be cropped or cut off at the top or bottom (there should be a slight gap). "
        "Also, white-colored areas in the character (such as eyes, dress, etc.) should not be transparent — they should be filled with actual white color. "
    )

def _style_prompt_ac_style() -> str:
    return (
        "Study the 3D character illustration style of the Nintendo Switch game Animal Crossing, "
        "and follow its way of depicting facial features, clothing, and hairstyles. Using that style, "
        "draw an illustration of the person in the attached image, replicating their hairstyle and clothing "
        "accessories. Make the background transparent, and create a warm and lively atmosphere by using "
        "bright sunlight and soft shadows under natural light. The character should look like one that appears "
        "in an actual Animal Crossing gameplay screen. Make sure the 3D aspect is clearly shown. "
    )

def _reverse() -> str:
    return (
        " Remember the previous prompts and generate an image according to the following requirements. "
        "Generate a paradoxical meme image using methods such as swapping the subject and object in the remembered prompt. "
        "For example, switch things around like changing \'a dog bites a person\' to "
        "\'a person bites a dog,\' or \'a person goes to a building\' to \'a building comes to a person.\' "
        "Do not add new subtitles or sentences in the image if there was no request in previous prompts. If an image is attached, maintain the original art style;"
        " if no image is attached, generate an image that fits the Korean meme style."
    )

def _reverse_test() -> str:
    return (
        "; 이전 프롬프트들을 기억하고 다음의 요구 사항에 따라 이미지를 생성하세요. 이전 프롬프트 속의 주어와 목적어를 바꾸는 등의 방식을 사용하여 한국 스타일의 웃긴 밈 이미지를 만드세요. "
        "예를 들어, '개가 사람을 문다'를 '사람이 개를 문다'로 바꾸거나, '사람이 햄버거를 먹는다'를 '햄버거가(햄버거 괴물이) 사람을 먹는다'와 같이 상황을 뒤바꾸어 표현하세요. "
        "이전 프롬프트가 주어와 목적어가 없는 단어나 문장이라면, 그 단어나 문장과 관련된 역설적이고 어이없고 웃긴 밈 이미지를 만드세요. 이미지에 자막이나 말풍선을 추가하지 마세요. "
    )

# ==========================================
# 4. 벤더 호출 함수 (실제 OpenAI/Gemini API 호출)
#    여기서는 "bytes"만 반환하고, Response는 엔드포인트에서 만든다.
# ==========================================

# ---------- 4-1. OpenAI / GPT-Image-1 계열 ----------

# GPT-Image-1 text2image 함수
def _openai_text2image(prompt: str) -> bytes:
    """
    GPT-Image-1 text -> image
    - prompt를 받아 직접 API 호출
    - 반환: raw jpeg 이미지 바이트
    """

    # 사전 검증
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI API key missing")
    if not OPENAI_IMAGE_MODEL:
        raise HTTPException(500, "OPENAI_IMAGE_MODEL is not set")
    
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.images.generate(
        prompt=prompt, 
        model=OPENAI_IMAGE_MODEL, 
        n=1,
        size="1024x1024",
        output_format="jpeg" )
    
    raw_bytes = base64.b64decode(resp.data[0].b64_json)
    return raw_bytes

# GPT-Image-1 text + reference images 함수
def _openai_text_with_refs(
    prompt: str,
    images: List[UploadFile],
) -> bytes:
    """
    GPT-Image-1 text + reference images -> image
    - 업로드된 이미지를 참조로 쓰는 text2image
    """
    # 사전 검증
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI API key missing")
    if not OPENAI_IMAGE_MODEL:
        raise HTTPException(500, "OPENAI_IMAGE_MODEL is not set")
    if not images or len(images) == 0:
        raise HTTPException(400, "No reference images provided")
    
    client = OpenAI(api_key=OPENAI_API_KEY)

    # UploadFile → file-like objects 준비
    file_objs = []
    for upload in images:
        raw = upload.file.read()
        fixed = _png_bytes(raw)
        bio = io.BytesIO(fixed)
        bio.name = upload.filename or "ref.png"
        file_objs.append(bio)

    resp = client.images.edit(
        model = OPENAI_IMAGE_MODEL,
        image = file_objs,             # 리스트 of file-like
        size="1024x1024",
        prompt = prompt,
        n = 1,
    )

    raw_bytes = base64.b64decode(resp.data[0].b64_json)
    return raw_bytes

# GPT-Image-1 스티커 PNG 용 함수
def _openai_text_with_refs_transparent(
    prompt: str,
    images: List[UploadFile],
) -> bytes:
    """
    GPT-Image-1 text + reference images -> image
    - 업로드된 이미지를 참조로 쓰는 text2image
    - 투명 배경 PNG 생성용
    """
    # 사전 검증
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI API key missing")
    if not OPENAI_IMAGE_MODEL:
        raise HTTPException(500, "OPENAI_IMAGE_MODEL is not set")
    if not images or len(images) == 0:
        raise HTTPException(400, "No reference images provided")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # UploadFile → file-like objects 준비
    file_objs = []
    for upload in images:
        raw = upload.file.read()
        fixed = _png_bytes(raw)
        bio = io.BytesIO(fixed)
        bio.name = upload.filename or "ref.png"
        file_objs.append(bio)

    resp = client.images.edit(
        model = OPENAI_IMAGE_MODEL,
        image = file_objs,             # 리스트 of file-like
        size="1024x1024",
        prompt = prompt,
        n = 1,
        background="transparent",
        output_format="png",
    )

    raw_bytes = base64.b64decode(resp.data[0].b64_json)
    return raw_bytes

def _openai_img_edit(
    prompt: str,
    base_image: UploadFile,
    mask_image: Optional[UploadFile] = None,
) -> bytes:
    """
    GPT-Image-1 기반 이미지 편집(inpainting) 함수.
    - base_image: 수정할 원본 이미지
    - mask_image: 편집할 영역을 알리는 마스크 이미지 (투명 혹은 알파 채널이 있는 PNG 형태 권장)
      만약 mask_image가 None이면, base_image 전체가 편집 대상이 됨.
    - size: "1024x1024" 등
    - 반환값: 편집된 이미지의 raw 바이트 (PNG로 통일 가능)
    """
    # 사전 검증
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI API key missing")
    if not OPENAI_IMAGE_MODEL:
        raise HTTPException(500, "OPENAI_IMAGE_MODEL is not set")
    
    client = OpenAI(api_key=OPENAI_API_KEY)

    # base 이미지 읽기 및 필요 시 PNG 변환
    base_bytes = base_image.file.read()
    base_bytes = _png_bytes(base_bytes)  # PNG로 통일

    # 마스크 이미지가 주어진 경우 처리
    mask_bytes = None
    if mask_image is not None:
        mask_bytes = mask_image.file.read()
        mask_bytes = _png_bytes(mask_bytes)  # PNG로 통일

    # OpenAI API 호출 준비
    # SDK 방식: client.images.edit(...)
    # https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1&api=image#edit-an-image-using-a-mask-inpainting
    resp = client.images.edit(
        model=OPENAI_IMAGE_MODEL,
        image=io.BytesIO(base_bytes),        # file-like object
        mask=io.BytesIO(mask_bytes) if mask_bytes is not None else None,
        prompt=prompt,
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )

    # 결과 처리
    b64 = resp.data[0].b64_json
    raw = base64.b64decode(b64)
    return _png_bytes(raw)


# ---------- 4-2. Gemini 계열 (나중에 구현) ----------

def _gemini_text2image(prompt: str, images: Optional[List[UploadFile]]) -> bytes:
    """
    Gemini text -> image
    """

    # 사전 검증
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Gemini API key missing")
    if not GEMINI_IMAGE_MODEL:
        raise HTTPException(500, "GEMINI_IMAGE_MODEL is not set")

    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [prompt]
    if images:
        for image in images:
            img_b = image.file.read()
            b64 = base64.b64encode(img_b).decode("utf-8")
            contents.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": b64
                }
            })

    resp = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=contents
    )

    for part in resp.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            raw_bytes = part.inline_data.data
            return raw_bytes


# Gemini 스티커 PNG 용 함수
def _gemini_text2image_transparent(prompt: str, images: Optional[List[UploadFile]]) -> bytes:
    """
    Gemini text -> image
    """

    # 사전 검증
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Gemini API key missing")
    if not GEMINI_IMAGE_MODEL:
        raise HTTPException(500, "GEMINI_IMAGE_MODEL is not set")

    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [prompt]
    if images:
        for image in images:
            img_b = image.file.read()
            b64 = base64.b64encode(img_b).decode("utf-8")
            contents.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": b64
                }
            })

    resp = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
        )
    )
    )

    for part in resp.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            raw_bytes = part.inline_data.data
            return raw_bytes


# ==========================================
# 5. 이미지 생성 엔드포인트 (+ provider별 분기까지 한 곳에서 처리)
# ==========================================

@app.post("/v1/images/generate")
async def generate_image(
    provider: Provider = Form(...),
    prompt: Optional[str] = Form(None),
    reverse: bool = Form(False),
    images: Optional[List[UploadFile]] = File(None),
):
    """
    엔트리 포인트:
      1) provider별 동작 정의
      2) 벤더 헬퍼 호출
      3) provider별로 JPEG/PNG로 바로 응답
    """
    try:
        # ---------------------------------
        # 1. provider별로 img_bytes + media_type 결정
        # ---------------------------------

        # ----- 기본 GPT provider (JPEG) -----
        if provider == Provider.GPT:
            if not images:
                logger.info("[generate_image] GPT text2image (no refs)")
                if reverse:
                    prompt = prompt + _reverse_test()
                img_bytes = _openai_text2image(prompt)              # JPEG 생성 가정
            else:
                logger.info(f"[generate_image] GPT text2image with refs (count={len(images)})")
                if reverse:
                    prompt = prompt + _reverse_test()
                img_bytes = _openai_text_with_refs(prompt, images)  # JPEG 생성 가정
            media_type = "image/jpeg"

        
        # ----- 기본 Gemini provider -----
        elif provider == Provider.GEMINI:
            if reverse:
                prompt = prompt + _reverse_test()
            img_bytes = _gemini_text2image(prompt, images)
            media_type = "image/jpeg"

        # ----- 눈 내리는 밤 (Gemini img2img, JPEG) -----
        elif provider == Provider.SNOW_NIGHT:
            if not images:
                raise HTTPException(400, "snow_night requires at least one image")
            if reverse:
                styled = _style_prompt_snow_night() + _reverse_test()
            else:
                styled = _style_prompt_snow_night()

            img_bytes = _gemini_text2image(styled, images)             # JPEG 생성 가정
            media_type = "image/jpeg"

        # ----- 픽셀 아트 스티커 (PNG + transparent) -----
        elif provider == Provider.PIXEL_ART:
            if not images:
                raise HTTPException(400, "pixel_art requires at least one image")
            styled = _style_prompt_pixel_art()
            img_bytes = _openai_text_with_refs_transparent(styled, images)       # PNG + 투명 배경 생성
            media_type = "image/png"

        # ----- 동물의 숲 스타일 스티커 (PNG + transparent) -----
        elif provider == Provider.AC_STYLE:
            if not images:
                raise HTTPException(400, "ac_style requires at least one image")
            styled = _style_prompt_ac_style()
            img_bytes = _openai_text_with_refs_transparent(styled, images)       # PNG + 투명 배경 생성
            media_type = "image/png"

        else:
            raise HTTPException(400, "unsupported provider")
        

        # ---------------------------------
        # 2. 최종 응답
        #    - JPEG 계열은 헬퍼에서 이미 JPEG로 만들어준다
        #    - PNG 계열은 헬퍼에서 투명 배경 PNG로 만들어준다
        # ---------------------------------
        return StreamingResponse(io.BytesIO(img_bytes), media_type=media_type)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
