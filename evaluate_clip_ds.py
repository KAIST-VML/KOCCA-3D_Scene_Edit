import os
import glob
import torch
import trimesh
import pyrender
import numpy as np
import csv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm # 진행 상황을 보기 위해 tqdm 추가

# --- 설정 ---
EVALUATION_DIR = 'outputs_batch' # 평가할 결과물들이 있는 폴더
CSV_OUTPUT_PATH = 'evaluation_results.csv' # 최종 결과를 저장할 CSV 파일 이름
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelManager:
    """메모리 효율성을 위해 모델들을 한 번만 로드하고 관리하는 클래스"""
    def __init__(self):
        self.models = {}
        print(f"Using device: {DEVICE}")

    def get_clip_model(self):
        if 'clip' not in self.models:
            print("Loading CLIP model for evaluation...")
            model_id = "openai/clip-vit-large-patch14"
            model = CLIPModel.from_pretrained(model_id).to(DEVICE)
            processor = CLIPProcessor.from_pretrained(model_id)
            self.models['clip'] = (model, processor)
        return self.models['clip']

def evaluate_clip_similarity(model_manager, original_mesh_path, edited_mesh_path, original_text, edited_text):
    """단일 정면 뷰를 기준으로 CLIP Directional Similarity를 계산"""
    clip_model, clip_processor = model_manager.get_clip_model()
    
    def get_embedding(text=None, image=None):
        if text:
            inputs = clip_processor(text=text, return_tensors="pt").to(DEVICE)
            return clip_model.get_text_features(**inputs)
        elif image:
            inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
            return clip_model.get_image_features(**inputs)
        return None

    def render_front_view(mesh_path):
        mesh = trimesh.load_mesh(mesh_path, force='mesh')
        scene = trimesh.Scene(mesh)
        camera_transform = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]])
        scene.camera_transform = camera_transform
        data = scene.save_image(resolution=(224, 224))
        return Image.open(trimesh.util.wrap_as_stream(data))

    original_render_img = render_front_view(original_mesh_path)
    edited_render_img = render_front_view(edited_mesh_path)
    
    with torch.no_grad():
        t_clip = get_embedding(text=original_text)
        t_hat_clip = get_embedding(text=edited_text)
        x_clip = get_embedding(image=original_render_img)
        x_hat_clip = get_embedding(image=edited_render_img)

        delta_text = t_clip - t_hat_clip
        delta_image = x_clip - x_hat_clip
        delta_text_norm = delta_text / torch.linalg.norm(delta_text)
        delta_image_norm = delta_image / torch.linalg.norm(delta_image)
        similarity = torch.dot(delta_image_norm.squeeze(), delta_text_norm.squeeze())
        
    return similarity.item()

def main():
    """
    지정된 폴더의 모든 결과물을 평가하고 CSV로 저장하는 메인 함수
    """
    print(f"🚀 Starting evaluation of all objects in '{EVALUATION_DIR}'...")
    models = ModelManager()
    
    # 평가할 모든 하위 폴더 목록을 가져옴 (_reference_images 같은 폴더는 제외)
    all_result_dirs = [d for d in glob.glob(os.path.join(EVALUATION_DIR, '*')) if os.path.isdir(d) and not os.path.basename(d).startswith('_')]
    
    if not all_result_dirs:
        print(f"No result directories found in '{EVALUATION_DIR}'. Exiting.")
        return

    print(f"Found {len(all_result_dirs)} result directories to evaluate.")
    
    evaluation_results = []

    # tqdm을 사용하여 진행 상황 표시
    for result_dir in tqdm(all_result_dirs, desc="Evaluating Objects"):
        object_name = os.path.basename(result_dir)
        
        try:
            # 평가에 필요한 파일 경로 정의
            edited_mesh_path = os.path.join(result_dir, 'edited_mesh.glb')
            original_mesh_path = os.path.join(result_dir, 'source_mesh.obj')
            source_caption_path = os.path.join(result_dir, 'source_caption.txt')
            editing_prompt_path = os.path.join(result_dir, 'editing_prompt.txt')

            # 필요한 파일이 모두 있는지 확인
            required_files = [edited_mesh_path, original_mesh_path, source_caption_path, editing_prompt_path]
            if not all(os.path.exists(p) for p in required_files):
                print(f"\n⚠️ Skipping {object_name}: Missing one or more required files.")
                continue

            # 텍스트 파일 읽기
            with open(source_caption_path, 'r') as f:
                original_text = f.read().strip()
            with open(editing_prompt_path, 'r') as f:
                edited_text = f.read().strip()
            
            # 점수 계산
            score = evaluate_clip_similarity(models, original_mesh_path, edited_mesh_path, original_text, edited_text)
            
            # 결과 저장
            evaluation_results.append({
                'object_name': object_name,
                'clip_ds_score': f"{score:.4f}",
                'editing_prompt': edited_text
            })

        except Exception as e:
            print(f"\n❌ FAILED to process {object_name}. Error: {e}")

    # --- 최종 결과를 CSV 파일로 저장 ---
    if not evaluation_results:
        print("No objects were successfully evaluated.")
        return

    print(f"\nSaving {len(evaluation_results)} evaluation results to {CSV_OUTPUT_PATH}...")
    
    with open(CSV_OUTPUT_PATH, 'w', newline='') as csvfile:
        fieldnames = ['object_name', 'clip_ds_score', 'editing_prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_results)

    print("\n🎉🎉🎉 Evaluation complete! 🎉🎉🎉")

if __name__ == '__main__':
    # 이 스크립트는 렌더링을 위해 headless 환경에서 실행해야 할 수 있습니다.
    # 예: xvfb-run --auto-servernum python evaluate_clip_ds.py
    main()