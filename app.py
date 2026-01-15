import os
import io
import base64
import json
from flask import Flask, request, Response
from PIL import Image
from openai import OpenAI

HF_TOKEN = os.environ["HF_TOKEN"]
MODEL = "Qwen/Qwen3-VL-8B-Instruct:novita"

MAX_SIDE = 768
MAX_TOKENS = 60
JPEG_QUALITY = 80

PROMPT = "Décris l'image en UNE phrase courte."

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

app = Flask(__name__)


def json_utf8(payload: dict, status: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False),
        status=status,
        content_type="application/json; charset=utf-8",
    )


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.route("/describe", methods=["POST"])
def describe():
    if not request.is_json:
        return json_utf8({"error": "Content-Type doit être application/json"}, 415)

    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image_base64")

    if not image_b64 or not isinstance(image_b64, str):
        return json_utf8({"error": "champ 'image_base64' manquant ou invalide"}, 400)

    if image_b64.startswith("data:"):
        try:
            image_b64 = image_b64.split(",", 1)[1]
        except Exception:
            return json_utf8({"error": "data URL invalide"}, 400)

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception:
        return json_utf8({"error": "base64 invalide"}, 400)

    try:
        img_url = image_bytes_to_data_url(image_bytes)

        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }],
        )

        description = (resp.choices[0].message.content or "").strip()
        return json_utf8({"description": description})

    except Exception as e:
        return json_utf8({"error": str(e)}, 500)


@app.route("/health", methods=["GET"])
def health():
    return json_utf8({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8040, debug=True)
