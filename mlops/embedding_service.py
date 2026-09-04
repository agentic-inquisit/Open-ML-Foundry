import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
model_name = os.getenv("CLIP_MODEL_NAME", DEFAULT_CLIP_MODEL)


class VisualEmbeddingService:
    def __init__(self, model_name=model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_image_embedding(self, image_bytes):
        """
        Generates a normalized embedding vector for an image.
        """
        image = Image.open(io.BytesIO(image_bytes))
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize the embedding
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten().tolist()

    def get_text_embedding(self, text):
        """
        Generates a normalized embedding vector for text query.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten().tolist()
