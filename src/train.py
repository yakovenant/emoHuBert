import torch.cuda
import warnings
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor, AutoConfig

from models import (HubertForTripletTrain, HubertTripletClassification,
                    get_model_for_clf_train, compute_metrics)
from data import (DataCollatorForClassification, DataCollatorForTriplets,
                  load_data_for_clf_train, load_data_for_triplet_train)


def check_device(device):
    if not torch.cuda.is_available() and device == 'gpu':
        warnings.warn('CUDA device is not defined, use CPU instead.')
        device = 'cpu'
    return device


def unfreeze_model_layers(model, n_layers):

    for param in model.parameters():
        param.requires_grad = False

    for name, param in list(model.named_parameters())[-n_layers:]:
        param.requires_grad = True


def triplet_train(filepath, dirpath, output_dir, model_dir, n_epochs, batch_size, learning_rate,
                  grad_accum_steps, device):

    model_id = 'facebook/hubert-base-ls960' if model_dir =='' else model_dir
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    device = check_device(device)

    config = AutoConfig.from_pretrained(model_id)
    model = HubertForTripletTrain.from_pretrained(model_id, config)

    unfreeze_model_layers(model, 2 + config.num_hidden_layers * 16)  # todo

    training_args = TrainingArguments(
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,  # todo
        gradient_accumulation_steps=grad_accum_steps,
        save_strategy='no',
        logging_strategy='epoch',
        report_to='none',
        fp16=False,
        use_cpu=True if device == 'cpu' else False)

    print('Prepare training data')
    data = load_data_for_triplet_train(filepath, dirpath, feature_extractor)
    data_collator = DataCollatorForTriplets(processor=feature_extractor)

    print('Use "Triplet loss"')
    train(model, data, data_collator, training_args, output_dir)


def classification_train(filepath, dirpath, output_dir, model_dir, n_labels, n_epochs, batch_size, learning_rate,
                         grad_accum_steps, device):

    model_id = 'facebook/hubert-base-ls960' if model_dir =='' else model_dir
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    device = check_device(device)

    model = get_model_for_clf_train(model_dir, n_labels)

    if isinstance(model, HubertTripletClassification):
        unfreeze_model_layers(model, 2)
    else:
        unfreeze_model_layers(model, 4 + model.config.num_hidden_layers * 16)  # todo

    training_args = TrainingArguments(
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,  # todo
        gradient_accumulation_steps=grad_accum_steps,
        save_strategy='no',
        eval_strategy='epoch',
        logging_strategy='epoch',
        report_to='none',
        fp16=False,
        use_cpu=True if device == 'cpu' else False)

    print('Prepare training data')
    data = load_data_for_clf_train(filepath, dirpath, feature_extractor)
    data_collator = DataCollatorForClassification(processor=feature_extractor)

    train(model, data, data_collator, training_args, output_dir, compute_metrics)


def train(model, data, data_collator, training_args, output_dir, compute_metrics_func=None):

    print('Start training process...')
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data['train'],
        eval_dataset=data['val'],
        compute_metrics=compute_metrics_func)

    trainer.train()
    trainer.save_model(output_dir)

    print(f'Trained model is saved to "{output_dir}"')


if __name__ == 'main':
    print('This is the training script.')
