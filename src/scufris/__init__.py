import argparse
import ollama
from faster_whisper import WhisperModel


def main() -> None:
    model_size = "distil-large-v3"

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe("audio.mp3", beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
