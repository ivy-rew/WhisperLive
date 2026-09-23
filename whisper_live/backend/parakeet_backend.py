import json
import logging
import threading
from dataclasses import dataclass

import onnx_asr

from whisper_live.backend.base import ServeClientBase


@dataclass
class Segment:
    start: float
    end: float
    text: str


def has_words(text):
    return any(c.isalnum() for c in text)


def split_sentences(result):
    segments = []
    start = None
    text = ""
    for token, timestamp in zip(result.tokens, result.timestamps):
        if token.startswith(" ") and text.rstrip().endswith((".", "?", "!")):
            if has_words(text):
                segments.append(Segment(start, timestamp, text.strip()))
            start = None
            text = ""
        if not has_words(token) and not has_words(text):
            continue
        if start is None:
            start = timestamp
        text += token
    if has_words(text):
        segments.append(Segment(start, result.timestamps[-1], text.strip()))
    return segments


class ServeClientParakeet(ServeClientBase):
    MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
    MODEL = None
    MODEL_LOCK = threading.Lock()
    SILENCE_DROP_S = 3
    SILENCE_KEEP_S = 1

    def __init__(
        self,
        websocket,
        language=None,
        client_uid=None,
        send_last_n_segments=10,
        clip_audio=False,
        same_output_threshold=10,
        translation_queue=None,
    ):
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            clip_audio=clip_audio,
            same_output_threshold=same_output_threshold,
            translation_queue=translation_queue,
        )
        self.language = language or "auto"

        with ServeClientParakeet.MODEL_LOCK:
            if ServeClientParakeet.MODEL is None:
                logging.info(f"Loading {self.MODEL_NAME}")
                ServeClientParakeet.MODEL = onnx_asr.load_model(self.MODEL_NAME, quantization="int8").with_timestamps()

        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()

        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": "parakeet"
        }))

    def transcribe_audio(self, input_sample):
        with ServeClientParakeet.MODEL_LOCK:
            result = ServeClientParakeet.MODEL.recognize(input_sample)
        if has_words(result.text):
            return split_sentences(result)
        # short windows often decode to nothing, keep them until there is clearly no speech
        duration = len(input_sample) / self.RATE
        if duration > self.SILENCE_DROP_S:
            with self.lock:
                self.timestamp_offset += duration - self.SILENCE_KEEP_S
        return []

    def handle_transcription_output(self, result, duration):
        if not result:
            return
        last_segment = self.update_segments(result, duration)
        segments = self.prepare_segments(last_segment)
        if segments:
            self.send_transcription_to_client(segments)
