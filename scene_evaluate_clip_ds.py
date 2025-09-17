import os
import argparse
import torch
import numpy as np
import trimesh
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 로드
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def get_text_embedding_for_clip(text: str):
    """CLIP 텍스트 임베딩"""
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
    """CLIP 이미지 임베딩"""
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(DEVICE)
    feats = clip_model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats


def render_front_view(mesh_path, resolution=(224, 224)):
    """GLB/OBJ 파일을 정면에서 렌더링"""
    mesh = trimesh.load(mesh_path, force="scene")
    # 카메라 위치를 Z축 앞쪽으로 배치
    scene = mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene(mesh)
    camera_transform = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -2.0],   # y축 뒤로 조금 이동
        [0, 0, 1, 3.0],    # z축 위로 올림
        [0, 0, 0, 1]
    ])
    scene.camera_transform = camera_transform
    data = scene.save_image(resolution=resolution)
    return Image.open(trimesh.util.wrap_as_stream(data)).convert("RGB")


def compute_clip_directional_similarity(img1, img2, caption1, caption2):
    """두 이미지와 두 텍스트 캡션으로 CLIP DS 점수 계산"""
    with torch.no_grad():
        t1 = get_text_embedding_for_clip(caption1)
        t2 = get_text_embedding_for_clip(caption2)
        x1 = get_image_embedding_for_clip(img1)
        x2 = get_image_embedding_for_clip(img2)

        delta_text = t1 - t2
        delta_image = x1 - x2

        eps = 1e-8
        if torch.linalg.norm(delta_text) < eps or torch.linalg.norm(delta_image) < eps:
            return 0.0

        sim = torch.nn.functional.cosine_similarity(delta_image, delta_text).mean().item()
    return float(sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two scene styles with CLIP DS")
    parser.add_argument("--style1", type=str, required=True, help="첫 번째 스타일 이름 (예: art_deco)")
    parser.add_argument("--style2", type=str, required=True, help="두 번째 스타일 이름 (예: steampunk)")
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="/source/sola/Kocca_3Dedit/scene_data/scene1",
        help="씬 데이터가 들어있는 폴더"
    )
    args = parser.parse_args()

    # 파일 경로
    scene1_path = os.path.join(args.scene_dir, f"edited_scene_{args.style1}_loc.glb")
    scene2_path = os.path.join(args.scene_dir, f"edited_scene_{args.style2}_loc.glb")

    if not os.path.exists(scene1_path) or not os.path.exists(scene2_path):
        raise FileNotFoundError(f"GLB 파일이 존재하지 않습니다:\n{scene1_path}\n{scene2_path}")

    print(f"🎨 Rendering scenes: {args.style1} vs {args.style2}")

    # 렌더링
    img1 = render_front_view(scene1_path)
    img2 = render_front_view(scene2_path)

    # 캡션
    caption1 = f"{args.style1} style living room"
    caption2 = f"{args.style2} style living room"

    # 점수 계산
    score = compute_clip_directional_similarity(img1, img2, caption1, caption2)
    print(f"✅ CLIP DS Score ({args.style1} ↔ {args.style2}): {score:.4f}")
