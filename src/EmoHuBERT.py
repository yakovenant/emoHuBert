import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoConfig, HubertModel
from CustomLoss import TripletLoss, sample_triplets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

class EmoHuBERT(nn.Module):
    def __init__(self, hubert_model, config):
        super().__init__()
        self.hubert = hubert_model
        self.fc = nn.Linear(config.hidden_size, config.embedding_size)  # 1024 → размер выхода HuBERT large, 128 → размер эмбеддинга

    def forward(self, x):
        with torch.no_grad():  # Можно заморозить HuBERT
            features = self.hubert(x).last_hidden_state  # [B, T, 1024]
        embeddings = self.fc(features.mean(dim=1))  # Усредняем по времени
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # Нормализация L2
        return embeddings_norm


def predict_emotion(self, model, audio_tensor, emb_ref, labels):
    model.eval()
    with torch.no_grad():
        emb_cur = model(audio_tensor.unsqueeze(0))  # Получаем эмбеддинг
        similarities = F.cosine_similarity(emb_cur, emb_ref)
        predicted_label = labels[similarities.argmax().item()]

    return predicted_label


if __name__ == '__main__':

    num_epochs = 3
    embedding_size = 128  # 256
    model_id = "facebook/hubert-base-ls960-ft"
    hubert_model = HubertModel.from_pretrained(model_id)

    config = AutoConfig.from_pretrained(
        model_id,
        embedding_size
    )

    model = EmoHuBERT(hubert_model, config).to(device)
    criterion = TripletLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for anchor_idx, positive_idx, negative_idx in sample_triplets(train_dataset, train_labels):
            anchor, _ = train_dataset[anchor_idx]
            positive, _ = train_dataset[positive_idx]
            negative, _ = train_dataset[negative_idx]

            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
