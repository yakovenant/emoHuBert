import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, TripletMarginLoss
from transformers import HubertModel, HubertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HubertTripletModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class HubertForTripletTrain(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hubert = HubertModel(config)
        self.fc1 = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.post_init()

    def get_embeddings(self, input_vals, attention_mask=None, output_attentions=None,
                       output_hidden_states=None, return_dict=None):

        outputs = self.hubert(input_vals, attention_mask, output_attentions, output_hidden_states, return_dict)
        activation = nn.ReLU(inplace=True)

        hidden_state = outputs.last_hidden_state
        hidden_state = self.fc1(hidden_state.mean(dim=1))
        hidden_state = activation(hidden_state)

        return hidden_state

    def forward(self, anchor_input_vals, positive_input_vals, negative_input_vals, attention_mask=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, return_loss=True):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        anchor_embeddings = self.get_embeddings(anchor_input_vals)
        positive_embeddings = self.get_embeddings(positive_input_vals)
        negative_embeddings = self.get_embeddings(negative_input_vals)

        loss_fn = TripletMarginLoss(margin=1.0)
        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        model_output = HubertTripletModelOutput(
            loss=loss, embeddings=(anchor_embeddings, positive_embeddings, negative_embeddings), attentions=None)
        return model_output


class HubertTripletClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.hubert = HubertModel(config)
        self.fc1 = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.post_init()

    def forward(self, input_values, labels=None, attention_mask=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, return_loss=True):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.hubert(input_values, attention_mask, output_attentions, output_hidden_states, return_dict)
        activation = nn.ReLU(inplace=True)

        hidden_state = outputs.last_hidden_state
        hidden_state = self.fc1(hidden_state.mean(dim=1))
        hidden_state = activation(hidden_state)

        logits = self.classifier(hidden_state)

        loss = None
        if labels is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        model_output = SequenceClassifierOutput(loss, logits,
                                                hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        return model_output
