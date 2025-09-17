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
from hy3dgen.text2image import HunyuanDiTPipeline
from evaluate_clip_ds import ModelManager as ClipModelManager, evaluate_clip_similarity



SOURCE_DATA_DIR = '/source/sola/dataset/3D-FUTURE-model-part1'
OUTPUT_DIR = 'outputs_batch'
NUM_OBJECTS_TO_PROCESS = 30 # 몇개의 객체? 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

START_INDEX = 1  # 시작할 객체 번호
END_INDEX = 60    # 종료할 객체 번호 (이 번호까지 포함)

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

    def get_txt2img_pipe(self):
        if 'txt2img' not in self.models:
            print("Loading Txt2Img pipeline (HunyuanDiT)...")
            # HunyuanDiTPipeline은 내부적으로 모델을 로드하므로 클래스 자체를 저장
            pipe = HunyuanDiTPipeline()
            self.models['txt2img'] = pipe
        return self.models['txt2img']

    def get_hunyuan_pipe(self):
        if 'hunyuan' not in self.models:
            print("Loading Hunyuan3D texturing pipeline...")
            pipe = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
            self.models['hunyuan'] = pipe
        return self.models['hunyuan']





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


def generate_simple_image(model_manager, prompt, output_path):
    """단순 Txt2Img로 참조 이미지를 생성합니다."""
    pipe = model_manager.get_txt2img_pipe()
    generator = torch.manual_seed(42)
    # HunyuanDiTPipeline은 seed를 직접 받지 않을 수 있으므로, torch.manual_seed 사용
    result_image = pipe(prompt, seed=42) # 파이프라인에 맞게 호출
    result_image.save(output_path)
    return output_path


def texture_mesh(model_manager, mesh_path, ref_image_path, output_path):
    pipe = model_manager.get_hunyuan_pipe()
    input_mesh = trimesh.load_mesh(mesh_path, force='mesh')
    ref_image = Image.open(ref_image_path).convert("RGBA")
    textured_mesh = pipe(mesh=input_mesh, image=ref_image)
    textured_mesh.export(output_path)
    return output_path


def save_source_with_texture(obj_dir, output_obj_path):
    """
    메시를 로드하고, 원본 텍스처를 '수동으로' 명시적으로 적용한 후 저장합니다.
    """
    source_obj_path = os.path.join(obj_dir, 'normalized_model.obj')
    source_texture_path = os.path.join(obj_dir, 'texture.png')
    
    # 1. 원본 텍스처 파일이 있는지 먼저 확인합니다.
    if not os.path.exists(source_texture_path):
        print(f"⚠️ Source texture not found at {source_texture_path}. Skipping texture application.")
        # 텍스처가 없는 경우, 원본 메시만 복사하거나 내보냅니다.
        shutil.copy(source_obj_path, output_obj_path)
        return

    print(f"Applying source texture to mesh: {os.path.basename(source_obj_path)}")

    # 2. 재질/텍스처 정보 없이 순수하게 지오메트리만 로드합니다.
    #    'process=False'는 불필요한 자동 처리를 방지합니다.
    mesh = trimesh.load(source_obj_path, force='mesh', process=False)

    # 3. 원본 텍스처 이미지를 직접 로드합니다.
    texture_image = Image.open(source_texture_path)
    
    # 4. 로드한 텍스처로 새로운 재질(material)을 만듭니다.
    material = trimesh.visual.texture.SimpleMaterial(image=texture_image)
    
    # 5. 메시의 시각적 속성에 UV 좌표와 위에서 만든 재질을 명시적으로 할당합니다.
    #    이렇게 하면 다른 텍스처가 캐시되어 있어도 무시하고 이 텍스처를 사용합니다.
    mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)
    
    # 6. 텍스처가 올바르게 적용된 메시를 내보냅니다.
    mesh.export(output_obj_path)
    print(f"✅ Saved textured source mesh to {output_obj_path}")


def main():
    print("Starting optimized batch processing job...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    models = ModelManager()
    clip_models = ClipModelManager()

    print("\nPhase 1: Finding all valid objects to select from...")
    all_source_dirs = sorted([d for d in glob.glob(f'{SOURCE_DATA_DIR}/*') if os.path.isdir(d)])
    
    all_valid_jobs = []
    for obj_dir in all_source_dirs:
        source_img_path = os.path.join(obj_dir, 'image.jpg')
        if not os.path.exists(source_img_path) or not os.path.exists(os.path.join(obj_dir, 'normalized_model.obj')):
            continue

        original_text = generate_caption(models, source_img_path)
        object_type = extract_object_type(original_text)

        if object_type != "object":
            job_info = {'dir': obj_dir, 'type': object_type, 'caption': original_text}
            all_valid_jobs.append(job_info)

    print(f"Found {len(all_valid_jobs)} total valid objects.")

    # 지정된 범위의 객체만 선택
    if START_INDEX > len(all_valid_jobs):
        print(f"Start index ({START_INDEX}) is greater than the number of valid objects ({len(all_valid_jobs)}). No new objects to process.")
        return
    
    # 1-based 인덱스를 0-based 슬라이싱으로 변환
    processing_jobs = all_valid_jobs[START_INDEX - 1:END_INDEX]
    print(f"--> Processing {len(processing_jobs)} objects from index {START_INDEX} to {END_INDEX}.\n")

    if not processing_jobs:
        print("No objects selected in the specified range. Exiting.")
        return

    # 선택된 job들로부터 unique_object_types 생성
    unique_object_types = set(job['type'] for job in processing_jobs)

    # 1b: 유니크한 객체 타입과 테마에 대한 참조 이미지 미리 생성
    print("\nPhase 2: Pre-generating all required reference images...")
    ref_image_dir = os.path.join(OUTPUT_DIR, "_reference_images")
    os.makedirs(ref_image_dir, exist_ok=True)
    
    themes = {
        # 1. 스팀펑크 (Steampunk)
        "steampunk": "a masterpiece steampunk {}, intricate brass and copper gears, polished mahogany, detailed mechanical parts, cinematic lighting",
        # 2. 고급스러운 아르데코 (Art Deco)
        "art_deco": "a luxurious art deco style {}, carved from white marble with intricate gold inlay, elegant and geometric, studio lighting, high detail",
        # 3. 지브리 애니메이션 (Ghibli Anime)
        "ghibli": "a {} in a cozy, beautiful anime style, hand-drawn with soft pastel colors, watercolor art, by Studio Ghibli, trending on pixiv",
        # 4. 클레이 애니메이션 (Claymation)
        "claymation": "a charming claymation {}, stop-motion style, made of plasticine clay with visible fingerprints, Aardman Animations aesthetic, soft volumetric lighting",
        # 5. 판타지 생물 발광 (Bioluminescent)
        "bioluminescent": "a magical {} made of bioluminescent plants and glowing mushrooms, intertwined with fantasy vines, ethereal, cinematic, Avatar movie style",
        "yellow": "a yellow {}",
        "cyberpunk": "a cyberpunk {}, made of dark chrome and glowing purple circuits",
        "wooden": "a cozy {}, made of warm, natural oak wood with a smooth finish",
        "glass": "a sleek, modern {}, made of translucent frosted glass with minimalist design",
        "medieval": "an ornate medieval-style {}, crafted from dark heavy wood and wrought iron details"
    }
    
    ref_image_paths = {}
    for obj_type in unique_object_types:
        ref_image_paths[obj_type] = {}
        for theme_name, prompt_template in themes.items():
            prompt = prompt_template.format(obj_type)
            output_path = os.path.join(ref_image_dir, f"ref_{obj_type}_{theme_name}.png")
            if not os.path.exists(output_path):
                print(f"Generating -> {output_path}")
                generate_simple_image(models, prompt, output_path)
            else:
                print(f"Skipping -> {output_path} (already exists)")
            ref_image_paths[obj_type][theme_name] = output_path
            
    # --- 2단계: 메인 처리 ---
    print("\nPhase 3: Processing all 3D objects using pre-generated images...")
    results_data = []
    object_counters = {}

    for i, job in enumerate(processing_jobs):
        obj_dir = job['dir']
        object_type = job['type']
        original_text = job['caption']
        
        print(f"\n--- [{i+1}/{NUM_OBJECTS_TO_PROCESS}] Processing Object: {os.path.basename(obj_dir)} ---")
        
        source_obj_path = os.path.join(obj_dir, 'normalized_model.obj')
        source_img_path = os.path.join(obj_dir, 'image.jpg')
        
        try:
            # 1. 객체별 카운터를 themes 루프 *시작 전*에 한 번만 증가시킵니다.
            object_counters[object_type] = object_counters.get(object_type, 0) + 1
            current_object_number = object_counters[object_type] + 100

            for theme_name, edited_prompt_template in themes.items():
                edited_prompt = edited_prompt_template.format(object_type)

                # 2. 위에서 계산한 번호를 사용하여 폴더 이름을 생성합니다.
                current_obj_name = f"{object_type}_{current_object_number}_{theme_name}"
                
                
                result_dir = os.path.join(OUTPUT_DIR, current_obj_name)
                os.makedirs(result_dir, exist_ok=True)
                print(f"\n-- Processing theme '{theme_name}' for {current_obj_name} --")

                pre_generated_ref_path = ref_image_paths[object_type][theme_name]
                ref_image_path_in_output = os.path.join(result_dir, 'ref_texture_image.png')
                shutil.copy(pre_generated_ref_path, ref_image_path_in_output)
                
                edited_mesh_path = texture_mesh(models, source_obj_path, ref_image_path_in_output, os.path.join(result_dir, 'edited_mesh.glb'))
                
                print(f"Evaluating similarity for {current_obj_name}...")

                save_source_with_texture(obj_dir, os.path.join(result_dir, 'source_mesh.obj'))
                source_obj_path_wtexture = os.path.join(result_dir, 'source_mesh.obj')
                score = evaluate_clip_similarity(
                    clip_models, 
                    source_obj_path_wtexture, 
                    edited_mesh_path, 
                    original_text, 
                    edited_prompt
                )
                
                
                shutil.copy(source_img_path, os.path.join(result_dir, 'source_image.jpg'))
                with open(os.path.join(result_dir, 'source_caption.txt'), 'w') as f: f.write(original_text)
                with open(os.path.join(result_dir, 'editing_prompt.txt'), 'w') as f: f.write(edited_prompt)
                
                results_data.append({
                    'object_type': current_obj_name,
                    'editing_prompt': edited_prompt,
                    'clip_ds_score': f"{score:.4f}"
                })
                print(f"✅ Finished processing {current_obj_name}. Score: {score:.4f}")

        except Exception as e:
            print(f"❌❌❌ FAILED to process {obj_dir}. Error: {e}")
            import traceback
            traceback.print_exc()

    # --- 최종 결과를 CSV 파일로 저장 ---
    csv_path = os.path.join(OUTPUT_DIR, 'results.csv')
    print(f"\nSaving final results to {csv_path}")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['object_type', 'editing_prompt', 'clip_ds_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print("\n🎉🎉🎉 Batch processing complete! 🎉🎉🎉")


if __name__ == '__main__':
    # xvfb-run을 사용하여 headless 환경에서 실행해야됨
    # xvfb-run --auto-servernum python run_batch_processing.py
    main()