import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader


class CustomDataLoader(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform
        self.label_map = {"angry": 0, "happy": 1, "sad": 2}  # Можно расширить

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        label = self.label_map[label]

        filepath = os.path.join(self.audio_dir, filename)
        waveform, sample_rate = torchaudio.load(filepath)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


if __name__ == '__main__':

    dataset = CustomDataLoader(csv_path="data_csv/labels.csv", audio_dir="data_csv/audio/")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
