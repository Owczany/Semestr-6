import pandas as pd
import torch
from torch.utils.data import Dataset


class SignLanguageMNIST(Dataset):
    # Mapowanie oryginalnych labeli (0-24, bez 9) na ciągły zakres 0-23
    LABEL_MAP = {v: i for i, v in enumerate(sorted(set(range(25)) - {9}))}

    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        raw_labels = data.iloc[:, 0].values
        self.labels = [self.LABEL_MAP[l] for l in raw_labels]
        self.images = data.iloc[:, 1:].values.reshape(-1, 28, 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
