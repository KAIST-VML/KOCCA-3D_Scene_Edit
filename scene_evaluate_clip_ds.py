import os
import re
import torch
from PIL import Image
from transformers import (
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# InstructBLIP
# BLIP_MODEL_NAME = "Salesforce/instructblip-flan-t5-xxl"  # or xl
# blip_processor = InstructBlipProcessor.from_pretrained(BLIP_MODEL_NAME)
# blip_model = InstructBlipForConditionalGeneration.from_pretrained(
#     BLIP_MODEL_NAME,
#     torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
#     device_map="auto"
# )

# CLIP (giant)
# clip_model = CLIPModel.from_pretrained(
#     "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
# ).to(DEVICE)
# clip_processor = CLIPProcessor.from_pretrained(
#     "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
# )

# CLIP ()
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float16,
    device_map="auto"
)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def get_text_embedding_for_clip(text: str):
    """CLIP 텍스트 임베딩 (77 토큰 트렁케이션 + 정규화)"""
    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(DEVICE)
    feats = clip_model.get_text_features(**inputs)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats


def get_image_embedding_for_clip(pil_image: Image.Image):
    """CLIP 이미지 임베딩 (정규화)"""
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(DEVICE)
    feats = clip_model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats


def compute_clip_directional_similarity(img1_path, img2_path, caption1, caption2):
    """두 이미지와 두 텍스트 캡션으로 CLIP DS 점수 계산"""
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    with torch.no_grad():
        # 임베딩
        t1 = get_text_embedding_for_clip(caption1)
        t2 = get_text_embedding_for_clip(caption2)
        x1 = get_image_embedding_for_clip(img1)
        x2 = get_image_embedding_for_clip(img2)

        # Δ 계산
        delta_text = t1 - t2
        delta_image = x1 - x2

        # 0-벡터 방어
        eps = 1e-8
        if torch.linalg.norm(delta_text) < eps or torch.linalg.norm(delta_image) < eps:
            return 0.0

        # 방향성 유사도
        sim = torch.nn.functional.cosine_similarity(delta_image, delta_text).mean().item()

    return float(sim)


if __name__ == "__main__":
    scene_folder = "/source/sola/Kocca_3Dedit/scene_data/scene1"
    image_files = sorted([
        os.path.join(scene_folder, f)
        for f in os.listdir(scene_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    assert len(image_files) == 9
    # 캡션
    captions = [
        "artdeco style living room",
        "bioluminiscent style living room",
        "claymation style living room",
        "cyberpunk style living room",
        "glass style living room",
        "medieval style living room",
        "steampunk style living room",
        "wooden style living room",
        "yellow style living room"
    ]

    # 모든 교차쌍 점수 계산
    n = len(image_files)
    for i in range(n):
        for j in range(i + 1, n):
            score = compute_clip_directional_similarity(
                image_files[i], image_files[j],
                captions[i], captions[j]
            )
            print(f"Pair ({i+1}, {j+1}) | {captions[i]} ↔ {captions[j]} | Score: {score:.4f}")
