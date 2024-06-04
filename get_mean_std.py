from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd 
from tqdm import tqdm

class Mydataset(Dataset):
    def __init__(self, df, root):
        super(Mydataset, self).__init__()
        self.root = root
        image_paths_df = df["study_id"] + "/" + df["image_id"] +".png"
        self.image_paths = image_paths_df.tolist()
        label_df = df["breast_birads"]
        self.labels = label_df.tolist()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_paths[idx])
        img = self._read_image(img_path, (224,224))
        label = int(self.labels[idx][-1])
        return img, torch.tensor(label).to(torch.long)
    
    def _read_image(self, filepath, new_size):
        image_pil = Image.open(filepath)
        
        # Kiểm tra chế độ của ảnh
        if image_pil.mode != 'L':
            image_pil = image_pil.convert('L')  # Chuyển đổi sang chế độ 'L' (grayscale) nếu cần thiết
        
        # Tạo ảnh RGB từ ảnh đơn kênh bằng cách sao chép giá trị của kênh đó vào cả ba kênh
        image_pil = Image.merge('RGB', (image_pil, image_pil, image_pil))
        
        # Resize ảnh
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        resized_image = transform(image_pil)
        resized_image = resized_image.to(torch.float)
        
        return resized_image
        
if __name__ == "__main__":
    dataset = Mydataset(pd.read_csv("csv/split_data.csv"), "/media/jackson/Data/archive/Processed_Images")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    # Initialize accumulators
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    total_images = 0

    for images, _ in tqdm(dataloader):
        # images shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = images.shape
        # Flatten the images into (batch_size * height * width, channels)
        images = images.view(batch_size, channels, -1)
        # Accumulate the sum and sum of squares
        mean_sum += images.mean(dim=[0, 2])
        std_sum += images.std(dim=[0, 2])
        total_images += 1

    # Calculate mean and standard deviation
    mean = mean_sum / total_images
    std = std_sum / total_images

    print("Mean:", mean)
    print("Standard Deviation:", std)