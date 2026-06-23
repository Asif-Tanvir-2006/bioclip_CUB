import torch
from PIL import Image
import open_clip
import torch.nn as nn
class BioClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        # self.model = self.model.to(torch.float32)
        self.model.eval()

        for p in self.parameters():
            p.requires_grad = False
        self.eval()
    def forward(self, image_paths):
        image = torch.cat([self.preprocess(Image.open(image_path)).unsqueeze(0) for image_path in image_paths])
        image = torch.tensor(image, dtype=torch.half, device='cuda')
        with torch.no_grad(), torch.autocast("cuda"):
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.unsqueeze(1)
        return torch.tensor(image_features, dtype=torch.float32)




