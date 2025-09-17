# KOCCA-3D Edit & CLIP-DS Evaluation 

이 프로젝트는 <b>3D 오브젝트(예: 3D-FUTURE)</b>를 대상으로

1) 이미지 캡셔닝(BLIP) → 
2) 객체 타입 추출 → 
3) 프롬프트 기반 참조 이미지 생성(HunyuanDiT) →  
4) 3D 텍스처링(Hunyuan3D) → 
5) CLIP Directional Similarity로 품질 평가 를 **배치로 자동화**

- 객체 생성 메인 스크립트: `run_batch_processing.py`  
- 객체 평가 유틸(모듈화): `evaluate_clip_ds.py` (함수/클래스 import)
- 씬 생성 메인 스크립트: `generate_scnene.py`
- 랜덤 씬 생성 스크립트: `generate_random_scene.py`
- 씬 평가 유틸: `scene_evalute_clip_ds.py`

---

## 0) Environments

- OS: Ubuntu 20.04
- GPU: NVIDIA-RTX-A5000 (VRAM 24GB)
- CUDA: 12.4
- Pytorch: 2.6.0+cu124 
- Network: 최초 실행 시 Hugging Face 모델/파이프라인 다운로드

---

## 1) Repository Clone

```bash
git clone https://github.com/KAIST-VML/KOCCA-3DEdit.git
cd KOCCA-3DEdit
```

<br>

## 2) Setting Up the Environment
```bash
conda create -n kocca3d python=3.10 -y
conda activate kocca3d
```

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124_full
pip install -r requirements.txt

```




