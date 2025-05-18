from datasets import DatasetDict, load_dataset
from ..utils import audio


def read_audio_to_array(batch, dirpath):
    batch['array'] = audio.get_audio_array(f'{dirpath}/{batch["audio"]}')
    return batch


def get_input_values(batch, feature_extractor):
    arr = batch['array']
    batch['input_values'] = audio.get_input_for_model(arr, feature_extractor)[0]
    return batch


def read_triplets_to_arrays(batch, dirpath):
    batch['anchor_input_values'] = audio.get_audio_array(f'{dirpath}/{batch["anchor"]}')
    batch['positive_input_values'] = audio.get_audio_array(f'{dirpath}/{batch["positive"]}')
    batch['negative_input_values'] = audio.get_audio_array(f'{dirpath}/{batch["negative"]}')
    return batch


def get_input_values_for_triplets(batch, feature_extractor):
    keys = ['anchor_input_values', 'positive_input_values', 'negative_input_values']
    for k in keys:
        arr = batch[k]
        batch[k] = audio.get_input_for_model(arr, feature_extractor)[0]
    return batch


def load_data(data_files, dirpath, feature_extractor, read_audio_func, get_input_values_func):
    ds = load_dataset('csv', data_files)
    ds = ds.map(read_audio_func, fn_kwargs={'dirpath': dirpath})
    ds = ds.map(get_input_values_func, fn_kwargs={'feature_extractor': feature_extractor})
    return ds


def load_data_for_test(filepath, dirpath, feature_extractor):
    data_files = {'test': str(filepath)}
    ds = load_data(data_files, dirpath, feature_extractor, read_audio_to_array, get_input_values)
    ds = ds.remove_columns('array')
    return ds


def load_data_for_clf_train(filepath, dirpath, feature_extractor):
    data_files = {'train': str(filepath)}
    ds = load_data(data_files, dirpath, feature_extractor, read_audio_to_array, get_input_values)
    ds = ds.rename_column('label', 'labels')
    ds = ds.remove_columns('array')
    train_data = ds['train'].train_test_split(shuffle=True, test_size=0.1)
    return DatasetDict({
        'train': train_data['train'],
        'val': train_data['test']
    })

def load_data_for_triplet_train(filepath, dirpath, feature_extractor):
    data_files = {'train': str(filepath)}
    ds = load_data(data_files, dirpath, feature_extractor, read_triplets_to_arrays, get_input_values_for_triplets)
    train_data = ds['train'].train_test_split(shuffle=True, test_size=0.1)
    return DatasetDict({
        'train': train_data['train'],
        'val': train_data['test']
    })
