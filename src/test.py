import torch
import numpy as np
import json

from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm

from models import get_model_for_test, compute_metrics, plot_metrics
from data import load_data_for_test


def test_model(filepath, dirpath, model_dir, output):

    model_id = 'facebook/hubert-base-ls960'
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    model = get_model_for_test(model_dir)
    data = load_data_for_test(filepath, dirpath, feature_extractor)
    labels = np.array(data['test']['label'])

    with torch.no_grad():
        logits = []
        for row in tqdm(data['test'], ncols=100):
            cur_logits = model(torch.tensor([row['input_values']])).logits
            logits.append(cur_logits)

    metrics = compute_metrics((logits, labels))
    print(metrics)

    metrics['model'] = model_dir
    with open(output, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'Results are saved to {output} file.')

    predictions = np.argmax(logits, axis=-1)
    plot_metrics(labels, predictions)


if __name__ == 'main':
    print('This is the test script.')
