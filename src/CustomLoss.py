import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


def sample_triplets(dataset, labels):
    """
    Выбираем триплеты: Anchor (A), Positive (P), Negative (N)
    """
    triplets = []
    label_dict = {label: [] for label in set(labels)}

    for idx, label in enumerate(labels):
        label_dict[label].append(idx)

    for label in label_dict:
        pos_indices = label_dict[label]
        neg_indices = [i for i in range(len(labels)) if labels[i] != label]

        for anchor_idx in pos_indices:
            positive_idx = random.choice(pos_indices)
            negative_idx = random.choice(neg_indices)
            triplets.append((anchor_idx, positive_idx, negative_idx))

    return triplets


if __name__ == '__main__':
    ...
