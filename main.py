import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from messages import download_file, get_messages

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
BASE_URL = "https://q.trap.jp/api/v3"


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Qwen3-Embeddingで推奨される最後のトークンのhidden stateを取得"""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


def extract_text(message: Dict[str, Any]) -> str:
    for key in ("content", "text", "body", "message"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def extract_image_file_ids(message: Dict[str, Any]) -> List[str]:
    file_ids: List[str] = []

    files = message.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            file_id = item.get("id") or item.get("fileId")
            mime = item.get("mime") or item.get("mimeType") or item.get("type")
            if file_id and isinstance(mime, str) and mime.startswith("image/"):
                file_ids.append(file_id)

    content = extract_text(message)
    if content:
        patterns = [
            r"/files/([0-9a-fA-F-]{16,})",
            r"https?://q\.trap\.jp/[^\s]*/files/([0-9a-fA-F-]{16,})",
            r"https?://q\.trap\.jp/api/v3/files/([0-9a-fA-F-]{16,})",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, content):
                file_ids.append(match)

    return list(dict.fromkeys(file_ids))


def extract_inline_images(message: Dict[str, Any]) -> List[Image.Image]:
    images: List[Image.Image] = []

    inline = message.get("images") or message.get("image")
    if isinstance(inline, list):
        candidates = inline
    else:
        candidates = [inline] if inline is not None else []

    for item in candidates:
        if isinstance(item, dict):
            data = item.get("data") or item.get("base64")
        elif isinstance(item, str):
            data = item
        else:
            data = None

        if not isinstance(data, str):
            continue

        if data.startswith("data:image"):
            try:
                header, b64 = data.split(",", 1)
                image_bytes = io.BytesIO(base64.b64decode(b64))
                images.append(Image.open(image_bytes).convert("RGB"))
            except Exception:
                continue

    return images


def load_images_from_message(message: Dict[str, Any]) -> List[Image.Image]:
    images: List[Image.Image] = []
    images.extend(extract_inline_images(message))

    for file_id in extract_image_file_ids(message):
        try:
            content = download_file(file_id, base_url=BASE_URL)
            images.append(Image.open(io.BytesIO(content)).convert("RGB"))
        except Exception as exc:
            print(f"画像の取得に失敗: {file_id} -> {exc}")

    return images


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"デバイス: {device}, dtype: {dtype}")
    print(f"モデルをロード中: {MODEL_NAME}")

    # Qwen3-VL-Embedding-8Bのロード
    # trust_remote_code=Trueで公式のカスタムコードを使用
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    model.eval()
    if device == "cpu":
        model.to(device)

    print("モデルのロード完了")
    return model, processor, device


def encode_batch(
    model,
    processor,
    texts: List[str],
    images: Optional[List[Image.Image]],
    device: str,
) -> torch.Tensor:
    """
    テキストと画像をembeddingに変換する。
    Qwen3-Embeddingでは最後のトークンのhidden stateを使用（last token pooling）。
    """
    # モデルに組み込みのencodeメソッドがあれば使用
    if hasattr(model, "get_embedding"):
        try:
            # Qwen3-VL-Embeddingのカスタムメソッド
            return model.get_embedding(text=texts, images=images)
        except Exception as e:
            print(f"get_embeddingメソッドでエラー: {e}")

    # 画像がある場合の処理
    if images:
        # 画像とテキストを組み合わせた入力を作成
        # VLモデルでは<image>タグをテキストに埋め込む形式が一般的
        processed_texts = []
        for text in texts:
            # <image>プレースホルダーがなければ追加
            if "<image>" not in text and "<|image|>" not in text:
                processed_texts.append(f"<|vision_start|><|image_pad|><|vision_end|>{text}")
            else:
                processed_texts.append(text)

        inputs = processor(
            text=processed_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )

    # デバイスに移動
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Qwen3-Embedding標準: last token pooling
    if hasattr(outputs, "last_hidden_state"):
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            embeddings = last_token_pool(hidden_states, attention_mask)
        else:
            # attention_maskがない場合は最後のトークンを使用
            embeddings = hidden_states[:, -1]

        # L2正規化（オプション、コサイン類似度計算用）
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    # フォールバック
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return F.normalize(outputs.pooler_output, p=2, dim=1)

    return F.normalize(outputs[0][:, -1], p=2, dim=1)


def embed_message(
    model,
    processor,
    message: Dict[str, Any],
    device: str,
) -> Tuple[List[float], int]:
    text = extract_text(message)
    images = load_images_from_message(message)

    if images:
        texts = [text] * len(images)
        emb = encode_batch(model, processor, texts, images, device)
        emb = emb.mean(dim=0)
    else:
        emb = encode_batch(model, processor, [text], None, device)[0]

    return emb.float().cpu().tolist(), len(images)


def main():
    messages = get_messages()
    model, processor, device = load_model()

    os.makedirs("out", exist_ok=True)

    outputs = []
    for message in messages:
        message_id = message.get("id") or message.get("messageId") or "unknown"
        embedding, image_count = embed_message(model, processor, message, device)
        outputs.append(
            {
                "messageId": message_id,
                "text": extract_text(message),
                "imageCount": image_count,
                "embedding": embedding,
            }
        )
        print(f"{message_id}: 埋め込み次元={len(embedding)} 画像={image_count}")

    with open("out/embeddings.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
