#!/bin/sh
exec uv run run_client.py --lang de --enable_translation --target_language en "$@"
