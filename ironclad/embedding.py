import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

class Embedding:

    def __init__(self, pretrained='casia-webface', device='cpu'):
        # Initialize the FaceNet model and set it to evaluation mode
        self.device = torch.device(device)
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

    def encode(self, image):
        # Get the embedding for the image
        with torch.no_grad():
            embedding = self.model(image)

        return embedding.squeeze().cpu().numpy()