import importlib
import pkg_resources

requirements = [
    'gradio',
    "tqdm>=4.66.3",
    'numpy',
    'ninja',
    'diffusers',
    'pybind11',
    'opencv-python',
    'einops',
    "transformers>=4.48.0",
    'omegaconf',
    'trimesh',
    'pymeshlab',
    'pygltflib',
    'xatlas',
    'accelerate',
    'fastapi',
    'uvicorn',
    'rembg',
    'onnxruntime'
]

def check_requirements(reqs):
    for req in reqs:
        try:
            # 버전 요구사항까지 파싱
            pkg_resources.require(req)
            name = req.split(">=")[0].split("==")[0]
            module = importlib.import_module(name.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {req} installed (version: {version})")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {req} NOT installed")
        except pkg_resources.VersionConflict as e:
            print(f"⚠️ {req} version conflict: {e}")
        except Exception as e:
            print(f"⚠️ Could not check {req}: {e}")

if __name__ == "__main__":
    check_requirements(requirements)
