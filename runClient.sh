#!/bin/sh
exec uv run run_client.py --lang de --model large-v3-turbo --enable_translation --target_language en "$@"
