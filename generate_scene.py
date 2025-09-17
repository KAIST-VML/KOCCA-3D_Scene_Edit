import os
import glob
import json
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix

# ===== 경로 =====
JSON_PATH    = "/source/sola/Kocca_3Dedit/scene_data/scene1/scene_object_transforms2.json"
BASE_DIR     = "/source/sola/Kocca_3Dedit/outputs_batch"
OUT_ORI      = "/source/sola/Kocca_3Dedit/scene_data/scene1/original_scene_loc.glb"
# TARGET_STYLE = "art_deco"
# OUT_EDITED   = f"/source/sola/Kocca_3Dedit/scene_data/scene1/edited_scene_{TARGET_STYLE}_loc.glb"

# Blender(Z-up) → Y-up 회전 (동차 4x4)
FIX_ZUP_TO_YUP = rotation_matrix(np.radians(-90.0), [1, 0, 0])

# # Blender Z-up → Y-up (permute axes)
# C = np.array([
#     [0, 0, 1, 0],
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 0, 1]
# ], dtype=np.float32)

# C_inv = np.linalg.inv(C)


# ---------- 유틸 ----------
def quat2mat(q):  # q = [w,x,y,z] (Blender 순서)
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)

def make_TRS(loc, quat, scale):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(loc, dtype=np.float64)

    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = quat2mat(quat)

    S = np.eye(4, dtype=np.float64)
    sx, sy, sz = np.asarray(scale, dtype=np.float64)
    S[0,0], S[1,1], S[2,2] = sx, sy, sz

    # Blender의 일반적 해석: 월드 TRS ~ T @ R @ S
    return T @ R @ S

def find_folder_for_name(name: str, base_dir: str, prefer_style: str | None = None):
    pattern = os.path.join(base_dir, f"{name}_*")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    if prefer_style:
        for c in candidates:
            if prefer_style.lower() in os.path.basename(c).lower():
                return c
    return candidates[0]

def find_mesh_for_name(name: str, use_edited: bool, base_dir: str, prefer_style: str | None = None):
    folder = find_folder_for_name(name, base_dir, prefer_style=prefer_style)
    if folder is None:
        return None
    fname = "edited_mesh.glb" if use_edited else "source_mesh.obj"
    mesh_path = os.path.join(folder, fname)
    if os.path.exists(mesh_path):
        return mesh_path
    return None

def load_mesh_simple(path):
    # 단일 mesh 컨테이너(지금 네 파일들)에 최적화
    try:
        m = trimesh.load(path, force='mesh', process=False, maintain_order=True)
        # 일부 포맷에선 list/tuple일 수 있음
        if isinstance(m, (list, tuple)):
            m = trimesh.util.concatenate([g for g in m if isinstance(g, trimesh.Trimesh)])
        return m
    except Exception as e:
        print(f"❌ load failed: {path} - {e}")
        return None

def create_floor_plane_TRS(loc, scale):
    """Plane도 동일한 TRS 규칙 적용 (Blender Z-up 기준 extents)"""
    sx, sy, sz = scale
    extents = (2.0 * sx, 2.0 * sy, 0.02 * max(sx, sy, sz))
    plane = trimesh.creation.box(extents=extents)
    M_blender = make_TRS(loc, [1, 0, 0, 0], [1, 1, 1])  # 회전/스케일은 extents로 반영됨
    # M_yup = C @ M_blender @ C_inv
    M_yup = FIX_ZUP_TO_YUP @ M_blender
    plane.apply_transform(M_yup)
    return plane



# ---------- 메인 ----------
def build_scene(json_path, use_edited, target_style=None):
    scene = trimesh.Scene()

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    for e in entries:
        name  = e["name"]
        loc   = e.get("location", [0,0,0])
        scale = e.get("scale", [1,1,1])
        # quat  = e.get("rotation_quaternion", [1,0,0,0])  # [w,x,y,z]

        # JSON에 새로 추가한 오일러 각(Degree)
        euler_deg = e.get("rotation_euler_deg", [ 0.0, 0.0, 0.0 ])
        rx_deg, ry_deg, rz_deg = euler_deg

        # '도(Degree)' 단위를 '라디안(Radian)' 단위로 변환
        rx_rad = np.radians(rx_deg)
        ry_rad = np.radians(ry_deg)
        rz_rad = np.radians(rz_deg)

        # 쿼터니언 행렬 대신 오일러 회전 행렬을 생성
        # 'sxyz'는 Blender의 기본 오일러 회전 순서(XYZ Static)
        R_matrix = trimesh.transformations.euler_matrix(rx_rad, ry_rad, rz_rad, axes='sxyz')
        
        # T, S 행렬 생성 (make_TRS 함수의 일부를 가져옴)
        T_matrix = trimesh.transformations.translation_matrix(loc)
        # scale_matrix 함수 대신, 이전 make_TRS 함수처럼 수동으로 대각 행렬을 만듬듬
        S_matrix = np.eye(4, dtype=np.float64)
        sx, sy, sz = np.asarray(scale, dtype=np.float64)
        S_matrix[0, 0] = sx
        S_matrix[1, 1] = sy
        S_matrix[2, 2] = sz

        M_blender = T_matrix @ R_matrix @ S_matrix


        

        # Y-up 표기로 변환 (change-of-basis)
        # M_yup = C @ M_blender @ C_inv
        M_yup = FIX_ZUP_TO_YUP @ M_blender

        mesh_path = find_mesh_for_name(name, use_edited, BASE_DIR, target_style)

        if not mesh_path:
            if name.lower() == "plane":
                plane = create_floor_plane_TRS(loc, scale)
                scene.add_geometry(plane, node_name=name)
                print(f"✅ Added plane: {name}")
            else:
                print(f"⚠️ mesh not found for {name} in {BASE_DIR}")
            continue

        print(f"🔍 Loading mesh for {name}: {mesh_path}")
        mesh = load_mesh_simple(mesh_path)
        if mesh is None or mesh.is_empty:
            print(f"⚠️ empty mesh: {mesh_path}")
            continue

        mesh.apply_transform(M_yup)
        scene.add_geometry(mesh, node_name=name)

    return scene

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
    OUT_EDITED = f"/source/sola/Kocca_3Dedit/scene_data/scene1/edited_scene_{target_style or 'default'}.glb"

    original_scene = build_scene(JSON_PATH, use_edited=False, target_style=None)
    if len(original_scene.geometry) == 0:
        raise ValueError("Original scene is empty.")
    original_scene.export(OUT_ORI)
    print(f"✅ Exported original scene → {OUT_ORI}")

    edited_scene = build_scene(JSON_PATH, use_edited=True, target_style)
    if len(edited_scene.geometry) == 0:
        raise ValueError("Edited scene is empty.")
    edited_scene.export(OUT_EDITED)
    print(f"✅ Exported edited scene ({target_style}) → {OUT_EDITED}")