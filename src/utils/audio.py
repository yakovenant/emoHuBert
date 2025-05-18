import librosa

def get_audio_array(filepath, sampling_rate=16000):
    arr = librosa.load(filepath, sr=sampling_rate, mono=True)[0]
    return arr

def get_input_for_model(audio_array, feature_extractor):
    input_vals = feature_extractor(audio_array,
                                   sampling_rate=feature_extractor.sampling_rate, return_tensors='pt').input_values
    return input_vals
