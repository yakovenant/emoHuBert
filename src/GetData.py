import torchaudio
from torchaudio.datasets import RAVDESS


if __name__ == '__main__':

    dataset = RAVDESS(root="data_csv", download=True)
    waveform, sample_rate, label, *_ = dataset[0]

    print(f"Sample rate: {sample_rate}, Label: {label}")
