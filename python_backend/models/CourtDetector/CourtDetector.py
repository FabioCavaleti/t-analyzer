import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

class KeypointRegressor(nn.Module):
    def __init__(self, num_keypoints=14):
        super(KeypointRegressor, self).__init__()
        # base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # self.backbone = base_model.features  # features extraídas
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_keypoints * 2),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


class CourtDetector:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = KeypointRegressor(num_keypoints=14).to(device)
        self.model.load_state_dict(torch.load("/project/resources/models/court_detector.pth", map_location=device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def detect(self, image: np.ndarray) -> list:
        orig_h, orig_w = image.shape[:2]
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
        keypoints = output.squeeze().cpu().numpy()

        keypoints[::2] *= orig_w / 224
        keypoints[1::2] *= orig_h / 224

        return keypoints.tolist()
