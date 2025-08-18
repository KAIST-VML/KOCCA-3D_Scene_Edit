import os
import glob
import torch
import trimesh
import pyrender
import numpy as np
import shutil
import csv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from hy3dgen.texgen import Hunyuan3DPaintPipeline


SOURCE_DATA_DIR = '/source/sola/dataset/3D-FUTURE-model-part1'
OUTPUT_DIR = 'outputs_batch'
NUM_OBJECTS_TO_PROCESS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelManager:
    """메모리 효율성을 위해 모델들을 한 번만 로드하고 관리하는 클래스"""
    def __init__(self):
        self.models = {}
        print(f"Using device: {DEVICE}")

    def get_blip_model(self):
        if 'blip' not in self.models:
            print("Loading Image Captioning model (BLIP)...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
            self.models['blip'] = (model, processor)
        return self.models['blip']

    def get_controlnet_pipe(self):
        if 'controlnet' not in self.models:
            print("Loading ControlNet-SDXL pipeline...")
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
            )
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            ).to(DEVICE)
            self.models['controlnet'] = pipe
        return self.models['controlnet']
        
    def get_hunyuan_pipe(self):
        if 'hunyuan' not in self.models:
            print("Loading Hunyuan3D texturing pipeline...")
            pipe = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
            self.models['hunyuan'] = pipe
        return self.models['hunyuan']

    def get_clip_model(self):
        if 'clip' not in self.models:
            print("Loading CLIP model for evaluation...")
            model_id = "openai/clip-vit-large-patch14"
            model = CLIPModel.from_pretrained(model_id).to(DEVICE)
            processor = CLIPProcessor.from_pretrained(model_id)
            self.models['clip'] = (model, processor)
        return self.models['clip']




def generate_caption(model_manager, image_path):
    model, processor = model_manager.get_blip_model()
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def extract_object_type(caption):
    common_types = ['sofa', 'bed', 'chair', 'table', 'cabinet', 'shelf', 'desk', 'couch']
    caption_lower = caption.lower()
    for obj_type in common_types:
        if obj_type in caption_lower:
            return obj_type
    return "object"


def create_depth_map(mesh_path, output_path):
    tri_mesh = trimesh.load_mesh(mesh_path, force="mesh")
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth =False)
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.3], bg_color=[0,0,0,0])
    scene.add(mesh, 'mesh')
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,3], [0,0,0,1]])
    scene.add(camera, pose = camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
    depth = renderer.render(scene, flags = pyrender.RenderFlags.DEPTH_ONLY)
    renderer.delete()
    if depth.max() > 0:
        depth_normalized = (depth / depth.max() * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth, dtype = np.uint8)
    depth_image = Image.fromarray(depth_normalized)
    depth_image.save(output_path)
    return output_path

def create_ref_image(model_manager, control_image_path, prompt, output_path):
    pipe = model_manager.get_controlnet_pipe()
    control_image = Image.open(control_image_path).convert("RGB")
    generator = torch.manual_seed(42)
    result_image = pipe(
        prompt, image=control_image, num_inference_steps=30, generator=generator, controlnet_conditioning_scale=0.7
    ).images[0]
    result_image.save(output_path)
    return output_path

def texture_mesh(model_manager, mesh_path, ref_image_path, output_path):
    pipe = model_manager.get_hunyuan_pipe()
    input_mesh = trimesh.load_mesh(mesh_path, force='mesh')
    ref_image = Image.open(ref_image_path).convert("RGBA")
    textured_mesh = pipe(mesh=input_mesh, image=ref_image)
    textured_mesh.export(output_path)
    return output_path

def render_front_view(mesh_path):
    """메시를 고정된 정면 뷰 하나로 렌더링하여 PIL Image 객체로 반환합니다."""
    mesh = trimesh.load_mesh(mesh_path, force='mesh')
    scene = trimesh.Scene(mesh)
    # 정면 뷰를 위한 고정된 카메라 위치
    camera_transform = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]
    ])
    scene.camera_transform = camera_transform
    # CLIP 모델의 표준 입력 해상도인 224x224로 렌더링
    data = scene.save_image(resolution=(224, 224))
    return Image.open(trimesh.util.wrap_as_stream(data))

def evaluate_clip_similarity(model_manager, original_mesh_path, edited_mesh_path, original_text, edited_text):
    """단일 정면 뷰를 기준으로 CLIP Directional Similarity를 계산합니다."""
    clip_model, clip_processor = model_manager.get_clip_model()
    
    def get_embedding(text=None, image=None):
        # 이 내부 함수는 수정할 필요가 없습니다.
        if text:
            inputs = clip_processor(text=text, return_tensors="pt").to(DEVICE)
            return clip_model.get_text_features(**inputs)
        elif image:
            inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
            return clip_model.get_image_features(**inputs)
        return None

    # 1. 원본과 편집된 메시를 각각 정면 뷰로 렌더링
    original_render_img = render_front_view(original_mesh_path)
    edited_render_img = render_front_view(edited_mesh_path)
    
    with torch.no_grad():
        # 2. 텍스트와 이미지 임베딩 추출
        t_clip = get_embedding(text=original_text)
        t_hat_clip = get_embedding(text=edited_text)
        x_clip = get_embedding(image=original_render_img)
        x_hat_clip = get_embedding(image=edited_render_img)

        # 3. 방향성 유사도 계산
        delta_text = t_clip - t_hat_clip
        delta_image = x_clip - x_hat_clip
        delta_text_norm = delta_text / torch.linalg.norm(delta_text)
        delta_image_norm = delta_image / torch.linalg.norm(delta_image)
        similarity = torch.dot(delta_image_norm.squeeze(), delta_text_norm.squeeze())
        
    return similarity.item()

def main():
    print("Starting batch processing job...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    models = ModelManager()
    
    source_dirs = sorted([d for d in glob.glob(f'{SOURCE_DATA_DIR}/*') if os.path.isdir(d)])
    
    results_data = []
    object_counters = {}

    for i, obj_dir in enumerate(source_dirs[:NUM_OBJECTS_TO_PROCESS]):
        print(f"\n--- [{i+1}/{NUM_OBJECTS_TO_PROCESS}] Processing Object: {os.path.basename(obj_dir)} ---")
        
        try:
            source_obj_path = os.path.join(obj_dir, 'normalized_model.obj')
            source_img_path = os.path.join(obj_dir, 'image.jpg')
            if not os.path.exists(source_obj_path) or not os.path.exists(source_img_path):
                print(f"Skipping {obj_dir}, required files not found.")
                continue

            # 1. 원본 캡션 생성 및 객체 타입 추출
            original_text = generate_caption(models, source_img_path)
            object_type = extract_object_type(original_text)
            
            # 2. 테마 정의
            themes = {
                "yellow": f"a yellow {object_type}",
                "cyberpunk": f"a cyberpunk {object_type}, made of dark chrome and glowing purple circuits"
            }

            for theme_name, edited_prompt in themes.items():
                # 3. 결과 저장 폴더 생성 (sofa_1, sofa_2, ...)
                object_counters[object_type] = object_counters.get(object_type, 0) + 1
                current_obj_name = f"{object_type}_{object_counters[object_type]}"
                result_dir = os.path.join(OUTPUT_DIR, current_obj_name)
                os.makedirs(result_dir, exist_ok=True)
                print(f"\n-- Processing theme '{theme_name}' for {current_obj_name} --")

                # 4. 3D 편집 파이프라인 실행
                depth_map_path = create_depth_map(source_obj_path, os.path.join(result_dir, 'control_depth_map.png'))
                ref_image_path = create_ref_image(models, depth_map_path, edited_prompt, os.path.join(result_dir, 'ref_texture_image.png'))
                edited_mesh_path = texture_mesh(models, source_obj_path, ref_image_path, os.path.join(result_dir, 'edited_mesh.glb'))
                
                # 5. 평가 실행
                print(f"Evaluating similarity for {current_obj_name}...")
                score = evaluate_clip_similarity(models, source_obj_path, edited_mesh_path, original_text, edited_prompt)
                
                # 6. 결과 저장
                shutil.copy(source_obj_path, os.path.join(result_dir, 'source_mesh.obj'))
                shutil.copy(source_img_path, os.path.join(result_dir, 'source_image.jpg'))
                with open(os.path.join(result_dir, 'source_caption.txt'), 'w') as f:
                    f.write(original_text)
                with open(os.path.join(result_dir, 'editing_prompt.txt'), 'w') as f:
                    f.write(edited_prompt)
                
                results_data.append({
                    'object_type': current_obj_name,
                    'editing_prompt': edited_prompt,
                    'clip_ds_score': f"{score:.4f}"
                })
                print(f"✅ Finished processing {current_obj_name}. Score: {score:.4f}")

        except Exception as e:
            print(f"❌ FAILED to process {obj_dir}. Error: {e}")
            import traceback
            traceback.print_exc()

    # 7. 최종 결과를 CSV 파일로 저장
    csv_path = os.path.join(OUTPUT_DIR, 'results.csv')
    print(f"\nSaving final results to {csv_path}")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['object_type', 'editing_prompt', 'clip_ds_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print("\n🎉 Batch processing complete! 🎉")


if __name__ == '__main__':
    # 이 스크립트는 GPU와 상당한 메모리를 사용합니다.
    # xvfb-run을 사용하여 headless 환경에서 실행해야 합니다.
    # 예: xvfb-run --auto-servernum python run_batch_processing.py
    main()