import whisper 

def speech_recognition(speech_file):
    # whisper
    model = whisper.load_model("base")
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(speech_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    speech_language = max(probs, key=probs.get)

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    speech_text = result.text

    #print the recognized text and language
    print(f"speech_text: {speech_text}")
    print(f"speech_language: {speech_language}")

    return speech_text, speech_language