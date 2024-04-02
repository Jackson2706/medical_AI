from torch.utils.data import Dataset
import pydicom
import os
import numpy as np
import random

class HealthDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_support_image=5, num_query_image=1):
        self.root_dir = root_dir
        self.transform = transform
        self.dicom_files = [f for f in os.listdir(root_dir) if f.endswith('.dicom')]
        self.num_support_image = num_support_image
        self.num_query_image = num_query_image

    def __len__(self):
        return len(self.dicom_files)
    
    def __getitem__(self, idx):
        dicom_path = os.path.join(self.root_dir, self.dicom_files[idx])
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(np.float32) / np.max(dicom.pixel_array)
        support_image_path_list = [path for path in self.dicom_files if path != self.dicom_files[idx]]
        selected_support_image_paths = random.sample(support_image_path_list, self.num_support_image)

        support_images = []
        for support_image_path in selected_support_image_paths:
            support_dicom_path = os.path.join(self.root_dir, support_image_path)
            support_dicom = pydicom.dcmread(support_dicom_path)
            support_images.append(support_dicom)
