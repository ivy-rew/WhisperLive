import json
import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np
import onnx_asr

from whisper_live.backend.base import ServeClientBase


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def has_words(text):
    return any(c.isalnum() for c in text)


def drop_context(tokens, timestamps, context_s, tolerance_s=0.08, logprobs=None):
    for i, (token, timestamp) in enumerate(zip(tokens, timestamps)):
        if timestamp >= context_s - tolerance_s and token.startswith(" "):
            filtered_logprobs = logprobs[i:] if logprobs is not None else None
            return tokens[i:], [max(0.0, t - context_s) for t in timestamps[i:]], filtered_logprobs
    return [], [], [] if logprobs is not None else None


def _make_segment(tokens, timestamps, logprobs, end=None):
    text = " " + "".join(tokens).strip()
    segment_start = timestamps[0]
    segment_end = timestamps[-1] if end is None else end
    token_groups = []
    group_tokens = []
    group_timestamps = []
    group_logprobs = []

    for index, token in enumerate(tokens):
        if token.startswith(" ") and group_tokens and has_words("".join(group_tokens)):
            token_groups.append((group_tokens, group_timestamps, group_logprobs))
            group_tokens, group_timestamps, group_logprobs = [], [], []
        group_tokens.append(token)
        group_timestamps.append(timestamps[index])
        if logprobs is not None and index < len(logprobs):
            group_logprobs.append(logprobs[index])

    if group_tokens:
        token_groups.append((group_tokens, group_timestamps, group_logprobs))

    words = []
    for index, (word_tokens, word_timestamps, word_logprobs) in enumerate(token_groups):
        word_text = "".join(word_tokens)
        if not has_words(word_text) or not word_logprobs:
            continue
        next_start = token_groups[index + 1][1][0] if index + 1 < len(token_groups) else segment_end
        probability = math.exp(sum(word_logprobs) / len(word_logprobs))
        words.append(Word(
            word=word_text,
            start=word_timestamps[0],
            end=max(word_timestamps[0], next_start),
            probability=min(1.0, probability),
        ))

    return Segment(segment_start, segment_end, text, words)


def split_sentences(tokens, timestamps, logprobs=None):
    segments = []
    sentence_tokens = []
    sentence_timestamps = []
    sentence_logprobs = [] if logprobs is not None else None

    for index, (token, timestamp) in enumerate(zip(tokens, timestamps)):
        sentence_text = "".join(sentence_tokens)
        if token.startswith(" ") and sentence_text.rstrip().endswith((".", "?", "!")):
            if has_words(sentence_text):
                segments.append(_make_segment(
                    sentence_tokens,
                    sentence_timestamps,
                    sentence_logprobs,
                    end=timestamp,
                ))
            sentence_tokens = []
            sentence_timestamps = []
            sentence_logprobs = [] if logprobs is not None else None
        if not has_words(token) and not has_words("".join(sentence_tokens)):
            continue
        sentence_tokens.append(token)
        sentence_timestamps.append(timestamp)
        if sentence_logprobs is not None and index < len(logprobs):
            sentence_logprobs.append(logprobs[index])
    if sentence_tokens and has_words("".join(sentence_tokens)):
        segments.append(_make_segment(sentence_tokens, sentence_timestamps, sentence_logprobs))
    return segments


class ServeClientParakeet(ServeClientBase):
    MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
    MODEL = None
    MODEL_LOCK = threading.Lock()
    SILENCE_DROP_S = 3
    SILENCE_KEEP_S = 1
    CONTEXT_S = 3

    def __init__(
        self,
        websocket,
        language=None,
        client_uid=None,
        send_last_n_segments=10,
        clip_audio=False,
        same_output_threshold=10,
        translation_queue=None,
        word_timestamps=False,
    ):
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            clip_audio=clip_audio,
            same_output_threshold=same_output_threshold,
            translation_queue=translation_queue,
            word_timestamps=word_timestamps,
        )
        self.language = language or "auto"
        # recent committed speech is decoded again in front of each window so the model keeps its language
        self.context = np.zeros(0, dtype=np.float32)
        self.window = self.context

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
        self.window = input_sample
        with ServeClientParakeet.MODEL_LOCK:
            result = ServeClientParakeet.MODEL.recognize(np.concatenate((self.context, input_sample)))
        tokens, timestamps, logprobs = drop_context(
            result.tokens,
            result.timestamps,
            len(self.context) / self.RATE,
            logprobs=result.logprobs,
        )
        if has_words("".join(tokens)):
            return split_sentences(tokens, timestamps, logprobs)
        # short windows often decode to nothing, keep them until there is clearly no speech
        duration = len(input_sample) / self.RATE
        if duration > self.SILENCE_DROP_S:
            with self.lock:
                self.timestamp_offset += duration - self.SILENCE_KEEP_S
        return []

    def handle_transcription_output(self, result, duration):
        if not result:
            return
        offset = self.timestamp_offset
        last_segment = self.update_segments(result, duration)
        committed = int((self.timestamp_offset - offset) * self.RATE)
        if committed > 0:
            self.context = np.concatenate((self.context, self.window[:committed]))[-self.CONTEXT_S * self.RATE:]
        segments = self.prepare_segments(last_segment)
        if segments:
            self.send_transcription_to_client(segments)
