import os
import shutil
import textwrap
import scipy
import numpy as np
import av
from pathlib import Path


def clear_screen():
    """Clears the console screen and its scrollback buffer."""
    print("\033[H\033[2J\033[3J", end="", flush=True)


def _wrap_with_highlights(text, wrapper, highlights):
    wrapped_lines = []
    source_cursor = 0

    for line_index, line in enumerate(wrapper.wrap(text)):
        indent = wrapper.initial_indent if line_index == 0 else wrapper.subsequent_indent
        content = line[len(indent):] if line.startswith(indent) else line
        source_start = text.find(content, source_cursor)
        if source_start < 0:
            wrapped_lines.append((line, []))
            continue

        source_end = source_start + len(content)
        line_highlights = []
        for start, end, color in highlights:
            highlight_start = max(0, start - source_start)
            highlight_end = min(len(content), end - source_start)
            if highlight_start < highlight_end:
                line_highlights.append((highlight_start, highlight_end, color))
        wrapped_lines.append((indent + content, line_highlights))
        source_cursor = source_end

    return wrapped_lines


def print_transcript(
    text,
    translated=False,
    timestamps=False,
    max_lines=3,
    ongoing_index=None,
    completed_indices=(),
):
    """Prints the last `max_lines` wrapped lines of transcript text in a subtitle-like block."""
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    wrap_width = max(10, min(80, terminal_width - 8))
    completed_indices = set(completed_indices)

    if timestamps:
        lines = []
        for index, t in enumerate(text):
            prefix = f'[{t["start"]} -> {t["end"]}] '
            wrapper = textwrap.TextWrapper(
                width=wrap_width,
                subsequent_indent=" " * len(prefix),
            )
            segment_text = f'{prefix}{t["text"]}'
            if index == ongoing_index:
                highlights = [(len(prefix), len(segment_text), "\033[30;43m")]
            elif index in completed_indices:
                highlights = [(len(prefix), len(segment_text), "\033[30;42m")]
            else:
                highlights = []
            lines.extend(_wrap_with_highlights(segment_text, wrapper, highlights))
    else:
        wrapper = textwrap.TextWrapper(width=wrap_width)
        transcript = " ".join(text) if translated else "".join(text)
        separator = " " if translated else ""
        highlights = []
        source_start = 0
        for index, part in enumerate(text):
            if index == ongoing_index:
                color = "\033[30;43m"
            elif index in completed_indices:
                color = "\033[30;42m"
            else:
                color = None
            if color and part:
                highlights.append((source_start, source_start + len(part), color))
            source_start += len(part) + (len(separator) if index < len(text) - 1 else 0)
        lines = _wrap_with_highlights(transcript, wrapper, highlights)

    for line, line_highlights in lines[-max_lines:]:
        centered = line.center(terminal_width)
        left_padding = (terminal_width - len(line)) // 2
        for start, end, color in reversed(line_highlights):
            start += left_padding
            end += left_padding
            centered = f'{centered[:end]}\033[0m{centered[end:]}'
            centered = f'{centered[:start]}{color}{centered[start:]}'
        print(centered)


def format_time(s):
    """Convert seconds (float) to SRT time format."""
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    seconds = int(s % 60)
    milliseconds = int((s - int(s)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def create_srt_file(segments, resampled_file):
    with open(resampled_file, 'w', encoding='utf-8') as srt_file:
        segment_number = 1
        for segment in segments:
            start_time = format_time(float(segment['start']))
            end_time = format_time(float(segment['end']))
            text = segment['text']

            srt_file.write(f"{segment_number}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

            segment_number += 1


def resample(file: str, sr: int = 16000):
    """
    Resample the audio file to 16kHz.

    Args:
        file (str): The audio file to open
        sr (int): The sample rate to resample the audio if necessary

    Returns:
        resampled_file (str): The resampled audio file
    """
    container = av.open(file)
    stream = next(s for s in container.streams if s.type == 'audio')

    resampler = av.AudioResampler(
        format='s16',
        layout='mono',
        rate=sr,
    )

    resampled_file = Path(file).stem + "_resampled.wav"
    output_container = av.open(resampled_file, mode='w')
    output_stream = output_container.add_stream('pcm_s16le', rate=sr)
    output_stream.layout = 'mono'

    for frame in container.decode(audio=0):
        frame.pts = None
        resampled_frames = resampler.resample(frame)
        if resampled_frames is not None:
            for resampled_frame in resampled_frames:
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)

    for packet in output_stream.encode(None):
        output_container.mux(packet)

    output_container.close()
    return resampled_file
