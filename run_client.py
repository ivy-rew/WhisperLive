from pathlib import Path
import sys
from whisper_live.client import TranscriptionClient
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p',
                          type=int,
                          default=9090,
                          help="Websocket port to run the server on.")
    parser.add_argument('--server', '-s',
                          type=str,
                          default='localhost',
                          help='hostname or ip address of server')
    parser.add_argument('--files', '-f',
                          type=str,
                          nargs='+',
                          help='Files to transcribe, separated by spaces. '
                              'If not provided, will use microphone input.')
    parser.add_argument('--output_file', '-o',
                          type=str,
                          default='./output_recording.wav',
                          help='output recording filename, only used for microphone input.')
    parser.add_argument('--model', '-m',
                          type=str,
                          default='small',
                          help='Model to use for transcription, e.g., "tiny, small.en, large-v3".')
    parser.add_argument('--lang', '-l',
                          type=str,
                          default='en',
                          help='Language code for transcription, e.g., "en" for English.')
    parser.add_argument('--prompt_file',
                          type=Path,
                          default=None,
                          help='UTF-8 text or Markdown file containing an initial prompt for Whisper.')
    parser.add_argument('--translate', '-t',
                          action='store_true',
                          help='Use Whisper built-in translation to English (sets task=translate). '
                              'For any-to-any translation, use --enable_translation instead.')
    parser.add_argument('--mute_audio_playback', '-a',
                          action='store_true',
                          help='Mute audio playback during transcription.') 
    parser.add_argument('--save_output_recording', '-r',
                          action='store_true',
                          help='Save the output recording, only used for microphone input.')
    parser.add_argument('--enable_translation',
                          action='store_true',
                          help='Enable translation via an OPUS-MT model (separate from Whisper --translate).')
    parser.add_argument('--target_language', '-tl',
                          type=str,
                          default='fr',
                          help='Target language for translation, e.g., "fr" for French.')
    parser.add_argument('--enable_timestamps',
                          action='store_true',
                          help='Show transcription with timestamps')
    parser.add_argument('--word_timestamps',
                          action='store_true',
                          help='Show word-level timestamps and underline low-confidence transcription words')
    parser.add_argument('--n_display_segments',
                          type=int,
                          default=10,
                          help='Number of transcript segments to display in terminal (default: 10).')

    args = parser.parse_args()

    if args.translate and args.enable_translation:
        print("[WARN]: Both --translate and --enable_translation are set. "
              "--translate uses Whisper's built-in to-English translation, "
              "while --enable_translation uses OPUS-MT. "
              "Both will be active.")

    initial_prompt = None
    if args.prompt_file is not None:
        try:
            initial_prompt = args.prompt_file.read_text(encoding='utf-8')
        except OSError as error:
            parser.error(f"Could not read prompt file '{args.prompt_file}': {error}")

    client = TranscriptionClient(
        args.server,
        args.port,
        lang=args.lang,
        translate=args.translate,
        model=args.model,                                  # also support hf_model => `Systran/faster-whisper-small`
        use_vad=True,
        save_output_recording=args.save_output_recording,  # Only used for microphone input, False by Default
        output_recording_filename=args.output_file,        # Only used for microphone input
        mute_audio_playback=args.mute_audio_playback,      # Only used for file input, False by Default
        enable_translation=args.enable_translation,        # Enable translation of the transcription output
        target_language=args.target_language,              # Target language for translation, e.g., "fr
        enable_timestamps=args.enable_timestamps,
        word_timestamps=args.word_timestamps,
        display_segments=args.n_display_segments,
        initial_prompt=initial_prompt,
    )

    if args.files is None:
        client()
        sys.exit(0)

    # Validate audio files
    valid_files = []
    for file_path in args.files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            valid_files.append(str(path))
        else:
            print(f"Warning: File not found: {file_path}")

    if not valid_files:
        print("Error: No valid audio files found!")
        sys.exit(1)

    print(f"Found {len(valid_files)} audio file(s) to stream:")
    for file_path in valid_files:
        print(f"  - {file_path}")

    for f in valid_files:
        client(f)
