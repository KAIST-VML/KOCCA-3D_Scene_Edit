import os
import torch
import trimesh
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelManager:
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
