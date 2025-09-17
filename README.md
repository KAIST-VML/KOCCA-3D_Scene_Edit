# KOCCA-3D Scene Edit & CLIP-DS Evaluation 


- 씬 생성 메인 스크립트: `generate_scnene.py`
- 랜덤 씬 생성 스크립트: `generate_random_scene.py`
- 씬 평가 유틸: `scene_evalute_clip_ds.py`

---

## 0) Environments

- OS: Ubuntu 20.04
- GPU: NVIDIA-RTX-A5000 (VRAM 24GB)
- CUDA: 12.4
- Pytorch: 2.6.0+cu124 
- Network: Download Hugging Face model/pipeline at the first time

---

## 1) Repository Clone

```bash
git clone https://github.com/KAIST-VML/KOCCA-3D_Scene_Edit.git
cd KOCCA-3D_Scene_Edit
```

<br>

## 2) Setting Up the Environment
```bash
conda create -n kocca3d python=3.10 -y
conda activate kocca3d
```

```bash
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install -r requirements.txt
```

(optional)if you want to check requirements are installed well,
```bash
python pkg_check.py
```

install xvfb-run (only gpu linux server like vessl required)
```bash
apt-get update
apt-get install -y \
    xvfb \
    freeglut3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglu1-mesa-dev
```

<br>


Theme list (10)
- art_deco
- bioluminescent
- claymation
- cyberpunk
- ghibli
- glass
- medieval
- steampunk
- wooden
- yellow

## 3-1) Generate Scene
- one theme
```bash
python generate_scene.py --style art_deco
python generate_scene.py --style steampunk
```

## 3-2) Generate Random Scene
- one theme, same layout, but different object
```bash
python generate_random_scene.py --style art_deco
python generate_random_scene.py --style steampunk
```

=> All results are saved in /scene_data/scene1


<br>

## 4) Evaluate Scene
we have to finish all the steps above.

```bash
# exemple: art_deco vs steampunk 
python compare_scene_styles.py --style1 art_deco --style2 steampunk

# change directory
python compare_scene_styles.py --style1 art_deco --style2 steampunk --scene_dir /source/sola/Kocca_3Dedit/scene_data/scene1
```


