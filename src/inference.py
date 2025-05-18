import os
import torch
import glob

from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm

from models import get_model_for_test
from utils import get_input_for_model, get_audio_array


def predict_label(logits):
    predictions = torch.argmax(logits, dim=-1)
    return predictions.numpy()[0]


def make_predicts(dirpath, model_path):

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
    model = get_model_for_test(model_path)
    abs_path = os.path.abspath(dirpath)

    predictions = []
    for filepath in tqdm(glob.glob(f'{abs_path}/*.wav'), ncols=100):
        audio_array = get_audio_array(filepath)
        input_values = get_input_for_model(audio_array, feature_extractor)

        with torch.no_grad():
            logits = model(input_values).logits

        label = predict_label(logits)
        predictions.append({
            'file': os.path.basename(filepath),
            'label': label
        })
    return predictions


if __name__ == 'main':
    print('This is the inference script.')
