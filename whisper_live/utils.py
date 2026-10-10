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
        for start, end, style in highlights:
            highlight_start = max(0, start - source_start)
            highlight_end = min(len(content), end - source_start)
            if highlight_start < highlight_end:
                line_highlights.append((highlight_start, highlight_end, style))
        wrapped_lines.append((indent + content, line_highlights))
        source_cursor = source_end

    return wrapped_lines


def _word_probability_highlights(text, words, offset=0):
    highlights = []
    cursor = 0
    for word in words or []:
        token = word.get("word", "").strip()
        if not token:
            continue
        start = text.find(token, cursor)
        if start < 0:
            start = text.lower().find(token.lower(), cursor)
        if start < 0:
            continue
        end = start + len(token)
        cursor = end
        probability = word.get("probability")
        if probability is None:
            continue
        probability = float(probability)
        if probability < 0.3:
            highlights.append((offset + start, offset + end, "confidence_low"))
        elif probability < 0.6:
            highlights.append((offset + start, offset + end, "confidence_medium"))
    return highlights


def print_transcript(
    text,
    translated=False,
    timestamps=False,
    max_lines=3,
    ongoing_index=None,
    completed_indices=(),
    words_by_segment=None,
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
                highlights = [(len(prefix), len(segment_text), "ongoing")]
            elif index in completed_indices:
                highlights = [(len(prefix), len(segment_text), "completed")]
            else:
                highlights = []
            highlights.extend(_word_probability_highlights(
                t["text"], t.get("words"), offset=len(prefix)
            ))
            lines.extend(_wrap_with_highlights(segment_text, wrapper, highlights))
    else:
        wrapper = textwrap.TextWrapper(width=wrap_width)
        transcript = " ".join(text) if translated else "".join(text)
        separator = " " if translated else ""
        highlights = []
        source_start = 0
        for index, part in enumerate(text):
            if index == ongoing_index:
                style = "ongoing"
            elif index in completed_indices:
                style = "completed"
            else:
                style = None
            if style and part:
                highlights.append((source_start, source_start + len(part), style))
            segment_words = words_by_segment[index] if words_by_segment and index < len(words_by_segment) else None
            highlights.extend(_word_probability_highlights(part, segment_words, offset=source_start))
            source_start += len(part) + (len(separator) if index < len(text) - 1 else 0)
        lines = _wrap_with_highlights(transcript, wrapper, highlights)

    for line, line_highlights in lines[-max_lines:]:
        left_padding = max(0, (terminal_width - len(line)) // 2)
        right_padding = max(0, terminal_width - len(line) - left_padding)
        rendered = []
        position = 0
        while position < len(line):
            background = None
            confidence = None
            for start, end, style in line_highlights:
                if start <= position < end:
                    if style in ("ongoing", "completed"):
                        background = style
                    elif style.startswith("confidence_"):
                        confidence = style

            end = position + 1
            while end < len(line):
                next_background = None
                next_confidence = None
                for start, stop, style in line_highlights:
                    if start <= end < stop:
                        if style in ("ongoing", "completed"):
                            next_background = style
                        elif style.startswith("confidence_"):
                            next_confidence = style
                if (next_background, next_confidence) != (background, confidence):
                    break
                end += 1

            codes = []
            if confidence == "confidence_low":
                codes.extend(("1", "4", "58;5;1"))
            elif confidence == "confidence_medium":
                codes.extend(("4", "58;5;9"))
            if background and confidence:
                codes.append("30")
                codes.append("43" if background == "ongoing" else "42")
            elif background == "ongoing":
                codes.extend(("30", "43"))
            elif background == "completed":
                codes.extend(("30", "42"))
            content = line[position:end]
            if codes:
                rendered.append(f'\033[{";".join(codes)}m{content}\033[0m')
            else:
                rendered.append(content)
            position = end

        print(f'{" " * left_padding}{"".join(rendered)}{" " * right_padding}')


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
