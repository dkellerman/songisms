
def transcribe_audio_wav2vec(f, offset=None, duration=None):
    import librosa
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

    audio, rate = librosa.load(f, sr=16000, offset=offset, duration=duration)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    input_values = tokenizer(audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(prediction)[0]

    return transcription
