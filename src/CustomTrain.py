from datasets import load_dataset, DatasetDict
from transformers import (
    HubertModel,
    HubertPreTrainedModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
from transformers.utils.generic import ModelOutput

#import librosa
import numpy as np

import torch
import torch.nn as nn

from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


class HubertTripletModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class HubertTripletModel(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # self.hubert = CustomHubertModel(config)
        self.hubert = HubertModel(config)
        self.fc1 = nn.Linear(config.hidden_size, config.embedding_size)
        # self.fc2 = nn.Linear(256, 4)

        self.post_init()

    def forward(
        self,
        anchor_input_values,
        positive_input_values,
        negative_input_values,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        return_loss=True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        anch_outputs = self.hubert(
            anchor_input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pos_outputs = self.hubert(
            positive_input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        neg_outputs = self.hubert(
            negative_input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        relu = nn.ReLU(inplace=True)

        hidden_state = anch_outputs.last_hidden_state
        hidden_state = self.fc1(hidden_state.mean(dim=1))
        hidden_state = relu(hidden_state)
        # anchor_embeddings = self.fc2(hidden_state)
        anchor_embeddings = hidden_state

        hidden_state = pos_outputs.last_hidden_state
        hidden_state = self.fc1(hidden_state.mean(dim=1))
        hidden_state = relu(hidden_state)
        # positive_embeddings = self.fc2(hidden_state)
        positive_embeddings = hidden_state

        hidden_state = neg_outputs.last_hidden_state
        hidden_state = self.fc1(hidden_state.mean(dim=1))
        hidden_state = relu(hidden_state)
        # negative_embeddings = self.fc2(hidden_state)
        negative_embeddings = hidden_state

        # pooled_output = pos_outputs.last_hidden_state.mean(dim=1)
        # positive_output = self.projector(pooled_output)
        # positive_output = relu(positive_output)

        # pooled_output = neg_outputs.last_hidden_state.mean(dim=1)
        # negative_output = self.projector(pooled_output)
        # negative_output = relu(negative_output)

        loss_fn = TripletMarginLoss(margin=1.0)
        # loss_fn = TripletMarginWithDistanceLoss(margin=0.5, distance_function=nn.CosineSimilarity())
        # loss = loss_fn(anch_outputs.last_hidden_state.mean(dim=1), pos_outputs.last_hidden_state.mean(dim=1), neg_outputs.last_hidden_state.mean(dim=1))
        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

        return HubertTripletModelOutput(
            loss=loss,
            embeddings=(anchor_embeddings, positive_embeddings, negative_embeddings),
            attentions=None,
        )


class DataCollatorTripletsWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: bool = False
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def pad(self, input_values: List[Dict[str, List[float]]], max_len: int) -> torch.Tensor:
        padded_input_values = []
        for in_values in input_values:
            padded_input = in_values["input_values"].copy()

            arr_len = len(padded_input)

            for _ in range(1, max_len // arr_len):
                padded_input.extend(in_values["input_values"])

            padded_input.extend(in_values["input_values"][:(max_len % arr_len)])
            padded_input_values.append(padded_input)

        return torch.tensor(padded_input_values)

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        anchor_input = [ {"input_values": feature["anchor_input_values"]} for feature in features ]
        positive_input = [ {"input_values": feature["positive_input_values"]} for feature in features ]
        negative_input = [ {"input_values": feature["negative_input_values"]} for feature in features ]

        get_len = lambda x: len(x["input_values"])

        max_anchor_len = max(map(get_len, anchor_input))
        max_positive_len = max(map(get_len, positive_input))
        max_negative_len = max(map(get_len, negative_input))

        max_len = max(max(max_anchor_len, max_positive_len), max_negative_len)

        anchor = self.pad(anchor_input, max_len)
        positive = self.pad(positive_input, max_len)
        negative = self.pad(negative_input, max_len)

        return {
            'anchor_input_values': anchor,
            'positive_input_values': positive,
            'negative_input_values': negative,
        }


# Для датасета с одним аудио в каждой строке
def speech_file_to_array(batch):
    batch["array"] = librosa.load(f'{data_root}/{batch["audio_path"]}', sr=16000, mono=False)[0]
    return batch

def get_input_values(batch, feature_extractor):
    array = batch["array"]
    input = feature_extractor(
        array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    batch["input_values"] = input.input_values[0]

    return batch

# Для датасета с триплетами
def speech_file_triplets_to_arrays(batch):
    batch["anchor_input_values"] = librosa.load(f'{data_root}/{batch["anchor"]}', sr=16000, mono=False)[0]
    batch["positive_input_values"] = librosa.load(f'{data_root}/{batch["positive"]}', sr=16000, mono=False)[0]
    batch["negative_input_values"] = librosa.load(f'{data_root}/{batch["negative"]}', sr=16000, mono=False)[0]
    return batch

def get_triplets_input_values(batch, feature_extractor):
    keys = ["anchor_input_values", "positive_input_values", "negative_input_values"]
    for key in keys:
        array = batch[key]
        input = feature_extractor(
            array,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )

        batch[key] = input.input_values[0]

    return batch


if __name__ == '__main__':

    data_root = '/media/ssd/Dusha'
    data_files = {"train": f"{data_root}/triplets_480.csv"}
    ds = load_dataset("csv", data_files=data_files)

    NUM_LABELS = 4
    labels_names = ["neutral", "angry", "positive", "sad"]
    model_id = "facebook/hubert-base-ls960"
    embedding_size = 128  # 256

    config = AutoConfig.from_pretrained(model_id)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_id,
        config=config
    )

    ds = ds.map(speech_file_triplets_to_arrays, num_proc=5)
    ds = ds.map(
        get_triplets_input_values,
        fn_kwargs={"feature_extractor": feature_extractor}
    )

    train_val = ds["train"].train_test_split(shuffle=True, test_size=0.1)

    ds = DatasetDict({
        'train': train_val['train'],
        'val': train_val['test']
    })

    model = HubertTripletModel.from_pretrained(
        model_id
    )

    for param in model.parameters():
        param.requires_grad = False

    layers_freeze_num = 10
    additional_layers_num = 1
    n_layers = layers_freeze_num * 16 + additional_layers_num * 2

    for name, param in list(model.named_parameters())[-n_layers:]:
        param.requires_grad = True
        print(name)

    data_collator = DataCollatorTripletsWithPadding(
        processor=feature_extractor,
        padding=True,
    )

    trainer_config = {
        "OUTPUT_DIR": f"{data_root}/train_results/custom-hubert-triplets-m1-480tr-1al",
        "MODEL_DIR": f"{data_root}/custom-hubert-base-dusha-ft-triplets-m1-480tr-1al",
        "EPOCHS": 20,
        "TRAIN_BATCH_SIZE": 4,
        "EVAL_BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,
        "WARMUP_STEPS": 500,
        "DECAY": 0.01,
        "LOGGING_STEPS": 20,
        "SAVE_STEPS": 200,
        "LR": 5e-5,
        "FP16": False,
    }

    training_args = TrainingArguments(
        output_dir=trainer_config["OUTPUT_DIR"],  # output directory
        gradient_accumulation_steps=trainer_config["GRADIENT_ACCUMULATION_STEPS"],
        # accumulate the gradients before running optimization step
        num_train_epochs=trainer_config["EPOCHS"],  # total number of training epochs
        per_device_train_batch_size=trainer_config["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=trainer_config["EVAL_BATCH_SIZE"],  # batch size for evaluation
        # warmup_steps=trainer_config["WARMUP_STEPS"],  # number of warmup steps for learning rate scheduler
        save_steps=trainer_config["SAVE_STEPS"],  # save checkpoint every 100 steps
        # weight_decay=trainer_config["DECAY"],  # strength of weight decay
        logging_steps=trainer_config["LOGGING_STEPS"],
        logging_strategy='epoch',
        eval_strategy="epoch",  # report metric at end of each epoch
        learning_rate=trainer_config["LR"],  # 5e-5 by default,
        report_to="none",
        fp16=trainer_config["FP16"],
        gradient_checkpointing=True,
        do_eval=True,
    )

    # triplets_480; 10 encoder layers; 1 add layer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
    )

    trainer.train()

    trainer.save_model(trainer_config["MODEL_DIR"])

    ds = ds.remove_columns(["anchor_input_values", "positive_input_values", "negative_input_values"])
    ds["train"].to_csv(f'{data_root}/triplets_480_train.csv')
    ds["val"].to_csv(f'{data_root}/triplets_480_val.csv')
