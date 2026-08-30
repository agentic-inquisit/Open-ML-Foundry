import torch
import torch.nn.functional as F
from torchvision import models, transforms
from ts.torch_handler.base_handler import BaseHandler
import io
from PIL import Image
import json

class SentinelHandler(BaseHandler):
    def __init__(self):
        super(SentinelHandler, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def initialize(self, context):
        # Load model (ResNet50 as a placeholder)
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Load labels if available
        # self.mapping = json.load(open("index_to_name.json"))

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # Handle base64
                pass
            image = Image.open(io.BytesIO(image))
            image = self.transform(image)
            images.append(image)
        return torch.stack(images).to(self.device)

    def inference(self, data):
        with torch.no_grad():
            results = self.model(data)
        return results

    def postprocess(self, data):
        probs = F.softmax(data, dim=1)
        top_prob, top_class = probs.topk(1, dim=1)
        
        responses = []
        for i in range(len(data)):
            responses.append({
                "class_id": top_class[i].item(),
                "confidence": top_prob[i].item(),
                "status": "success"
            })
        return responses
