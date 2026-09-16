#!/bin/bash

source .venv/bin/activate
python run_client.py --lang de --model large-v3-turbo --enable_translation --target_language en

