# import os
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset

# class GTSRBDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         img_path = os.path.join(self.root_dir, row["Path"])
#         label = int(row["ClassId"])

#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)

#         return image, label


# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import pandas as pd
# import os

# class GTSRBDataset(Dataset):
#     """Custom dataset for GTSRB CSV annotations"""
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
#         image = Image.open(img_name).convert("RGB")
#         label = int(self.data.iloc[idx, 1])

#         if self.transform:
#             image = self.transform(image)

#         return image, label





import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, -1])
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx, 6])  # ClassId column
        if self.transform:
            image = self.transform(image)
        return image, label
