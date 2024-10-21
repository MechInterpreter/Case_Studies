import torch
from torchvision import transforms
from PIL import Image

class Preprocessing:

    def __init__(self, image_size=160):
        self.image_size = image_size # 224 depending on model

        # self.processing = transforms.Compose(
        #     [
        #         # Resize
        #         transforms.Resize((self.image_size, self.image_size)),
        #         # Scales and convert to tensor
        #         transforms.ToTensor(), 
        #     ]
        # )

        # Define a transform to preprocess images for the model
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def process(self, probe):
        probe = self.transform(probe).unsqueeze(0).to(self.device)
        return probe