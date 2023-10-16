import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CustomDataset(Dataset):
    def __init__(self, data, image_width, image_height):
        self.data = data
        self.images = self.data['images']
        self.masks = self.data['masks']
        self.transform = transforms.Compose([
            transforms.Resize(size=(image_width, image_height),
                              interpolation=transforms.InterpolationMode.BICUBIC),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(cv2.imread(self.data['images'][index], cv2.IMREAD_GRAYSCALE))
        mask = torch.from_numpy(cv2.imread(self.data['masks'][index], cv2.IMREAD_GRAYSCALE))
        image = self.transform(torch.unsqueeze(image, dim=0))
        mask = self.transform(torch.unsqueeze(mask, dim=0))
        return image, mask
