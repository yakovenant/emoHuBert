import torch
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Union, Optional


def get_input_len(x):
    return len(x['input_values'])


def batch_padding(input_vals: List[Dict[str, List[float]]], max_len: int) -> torch.Tensor:

    padded_input_vals = []
    for data in input_vals:
        padded_input = data['input_values'].copy()
        padded_len = len(padded_input)
        for _ in range(1, max_len // padded_len):
            padded_input.extend(data['input_values'])
        padded_input.extend(data['input_values'][:(max_len % padded_len)])
        padded_input_vals.append(padded_input)
    return torch.tensor(padded_input_vals)


@dataclass
class DataCollatorForClassification:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: bool = False
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{'input_values': feature['input_values']} for feature in features]
        labels = [feature['labels'] for feature in features]

        max_len = max(map(get_input_len, input_features))
        batch = batch_padding(input_features, max_len)

        return {
            'input_values': batch,
            'labels': torch.tensor(labels)
        }


@dataclass
class DataCollatorForTriplets:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: bool = False
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_anchor = [{'input_values': feature['anchor_input_values']} for feature in features]
        input_positive = [{'input_values': feature['positive_input_values']} for feature in features]
        input_negative = [{'input_values': feature['negative_input_values']} for feature in features]

        max_anchor_len = max(map(get_input_len, input_anchor))
        max_positive_len = max(map(get_input_len, input_positive))
        max_negative_len = max(map(get_input_len, input_negative))

        max_len = max(max(max_anchor_len, max_positive_len), max_negative_len)

        anchor_vals = batch_padding(input_anchor, max_len)
        positive_vals = batch_padding(input_positive, max_len)
        negative_vals = batch_padding(input_negative, max_len)

        return {
            'anchor_input_values': anchor_vals,
            'positive_input_values': positive_vals,
            'negative_input_values': negative_vals
        }
