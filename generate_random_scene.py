import os
import glob
import json
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix
import random
import re 
import argparse 

# ===== 경로 및 전역 설정 =====
JSON_PATH = "/source/sola/Kocca_3Dedit/scene_data/scene1/scene_object_transforms2.json"
BASE_DIR = "/source/sola/Kocca_3Dedit/outputs_batch"
OUT_SCENE_DIR = "/source/sola/Kocca_3Dedit/scene_data/scene1"

# Blender(Z-up) → Y-up 회전 (동차 4x4)
FIX_ZUP_TO_YUP = rotation_matrix(np.radians(-90.0), [1, 0, 0])


def quat2mat(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12: return np.eye(3)
    q /= n
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ])

def make_TRS(loc, quat, scale):
    T = np.eye(4); T[:3, 3] = loc
    R = np.eye(4); R[:3, :3] = quat2mat(quat)
    S = np.eye(4); sx, sy, sz = scale; S[0,0], S[1,1], S[2,2] = sx, sy, sz
    return T @ R @ S

def create_floor_plane_TRS(loc, scale):
    sx, sy, sz = scale
    extents = (2.0*sx, 2.0*sy, 0.02*max(sx, sy, sz))
    plane = trimesh.creation.box(extents=extents)
    M_blender = make_TRS(loc, [1,0,0,0], [1,1,1])
    M_yup = FIX_ZUP_TO_YUP @ M_blender
    plane.apply_transform(M_yup)
    return plane

def load_mesh_simple(path):
    try:
        m = trimesh.load(path, force='mesh', process=False, maintain_order=True)
        if isinstance(m, (list, tuple)):
            m = trimesh.util.concatenate([g for g in m if isinstance(g, trimesh.Trimesh)])
        return m
    except Exception as e:
        print(f"❌ load failed: {path} - {e}")
        return None


def find_random_mesh_path(base_name: str, theme: str, base_dir: str, use_edited: bool):
    """'base_name_*_{테마}' 패턴의 폴더를 검색하여 랜덤 메쉬 경로를 반환합니다."""
    search_pattern = os.path.join(base_dir, f"{base_name}_*_{theme}*")
    matching_folders = glob.glob(search_pattern)
    if not matching_folders:
        return None
    chosen_folder = random.choice(matching_folders)
    mesh_filename = "edited_mesh.glb" if use_edited else "source_mesh.obj"
    mesh_path = os.path.join(chosen_folder, mesh_filename)
    return mesh_path if os.path.exists(mesh_path) else None



def build_scene_from_json_with_swapped_meshes(json_path: str, theme: str, use_edited: bool, base_dir: str):
    """
    JSON 파일에서 위치/회전/크기 값을 읽어오되,
    메쉬(가구)는 이름과 테마에 맞춰 랜덤으로 스왑(교체)
    """
    scene = trimesh.Scene()
    
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
        
    print(f"--- '{theme}' 테마로 JSON 기반 씬 생성 시작 ---")

    for e in entries:
        name = e["name"]

        # 'Plane' 객체는 특별 처리
        if name.lower() == "plane":
            plane = create_floor_plane_TRS(e["location"], e["scale"])
            scene.add_geometry(plane, node_name=name)
            print(f"✅ 바닥 평면 생성: {name}")
            continue

        loc = e.get("location", [0, 0, 0])
        scale = e.get("scale", [1, 1, 1])
        euler_deg = e.get("rotation_euler_deg", [0, 0, 0])
        
        # 메쉬 스왑: JSON의 이름(예: "bed_1")에서 베이스 이름("bed")을 추출
        # 정규 표현식을 사용하여 이름 뒤의 숫자와 밑줄을 제거
        base_name = re.sub(r'_\d+$', '', name)
        
        mesh_path = find_random_mesh_path(base_name, theme, base_dir, use_edited)
        
        if not mesh_path:
            print(f"⚠️ {base_name}({theme}) 테마의 대체 메쉬를 찾지 못했습니다. {name}을 건너뜁니다.")
            continue
            
        print(f"🔍 {name} 자리에 로드: {os.path.relpath(mesh_path, base_dir)}")
        mesh = load_mesh_simple(mesh_path)
        if mesh is None or mesh.is_empty:
            continue
            
        # 매트릭스 생성 (수정한 오일러 각도 로직 사용)
        rx, ry, rz = np.radians(euler_deg)
        R_matrix = trimesh.transformations.euler_matrix(rx, ry, rz, axes='sxyz')
        T_matrix = trimesh.transformations.translation_matrix(loc)
        S_matrix = np.eye(4); S_matrix[0,0], S_matrix[1,1], S_matrix[2,2] = scale
        
        M_blender = T_matrix @ R_matrix @ S_matrix
        M_final = FIX_ZUP_TO_YUP @ M_blender

        # 씬에 추가
        scene.add_geometry(mesh, transform = M_final, node_name=name)

    print("--- 씬 생성 완료 ---")
    return scene

# ---------- 메인 실행 부분 ----------
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Scene generator with target style")
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Target style for edited scene (예: art_deco, steampunk, ghibli 등)"
    )
    args = parser.parse_args()

    target_style = args.style
    OUT_EDITED = f"/source/sola/Kocca_3Dedit/scene_data/scene1/edited_scene_{target_style or 'default'}_random.glb"

    TARGET_THEME = args.theme     
    USE_EDITED_MESHES = True

    # JSON 파일의 좌표를 그대로 사용하되, 가구만 바꾸는 새 씬 생성
    new_scene = build_scene_from_json_with_swapped_meshes(
        json_path=JSON_PATH,
        theme=TARGET_THEME,
        use_edited=USE_EDITED_MESHES,
        base_dir=BASE_DIR
    )

    if len(new_scene.geometry) == 0:
        raise ValueError("씬이 비어있습니다. JSON 경로 또는 메쉬 파일 경로를 확인하세요.")
    else:
        OUT_EDITED = f"/source/sola/Kocca_3Dedit/scene_data/scene1/edited_scene_{target_style or 'default'}_random.glb"
        
        new_scene.export(output_path)
        print(f"✅ 가구가 교체된 새로운 씬 저장 완료 → {output_path}")