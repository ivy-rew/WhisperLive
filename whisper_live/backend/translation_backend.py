import json
import logging
import threading
import time
import queue
from typing import Dict, Any, Optional
import torch
import threading
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from whisper_live.backend.base import ServeClientBase


class ServeClientTranslation(ServeClientBase):
    PARTIAL_INTERVAL_S = 1.5

    """
    Handles translation of completed transcription segments in a separate thread.
    Reads from a queue populated by the transcription backend and sends translated
    segments back to the client via WebSocket.
    """
    
    def __init__(
        self,
        client_uid,
        websocket,
        translation_queue,
        target_language="fr",
        send_last_n_segments=10,
        source_language=None,
    ):
        """
        Initialize the translation client.
        
        Args:
            client_uid (str): Unique identifier for the client
            websocket: WebSocket connection to the client
            translation_queue (queue.Queue): Queue containing completed segments to translate
            target_language (str): Target language code (default: "fr" for French)
            send_last_n_segments (int): Number of recent translated segments to send
            source_language (str): Language of the transcript, None for multilingual input
        """
        super().__init__(client_uid, websocket, send_last_n_segments)
        self.translation_queue = translation_queue
        self.target_language = target_language
        self.model_name = f"Helsinki-NLP/opus-mt-{source_language or 'mul'}-{target_language}"
        self.translated_segments = []
        self.partial_segment = None
        self.translation_model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self.load_translation_model()
        
    def load_translation_model(self):
        """Load the translation model and tokenizer."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Loading translation model on device: {self.device}")
            
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model_loaded = True
            logging.info(f"Translation model {self.model_name} loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load translation model: {e}")
            self.translation_model = None
            self.tokenizer = None
            self.model_loaded = False
    
    def translate_text(self, text: str, num_beams=None) -> str:
        """
        Translate a single text segment.
        
        Args:
            text (str): Text to translate
            
        Returns:
            str: Translated text or original text if translation fails
        """
        if not self.model_loaded or not text.strip():
            return text
            
        try:
            # Encode input and move to device
            encoded_input = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.translation_model.generate(**encoded_input, num_beams=num_beams)
            
            # Decode output
            output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return output[0] if output else text
            
        except Exception as e:
            logging.error(f"Translation failed for text '{text}': {e}")
            return text
    
    def process_translation_queue(self):
        """
        Process segments from the translation queue.
        Completed segments are translated in order. The newest incomplete segment is
        translated at most every PARTIAL_INTERVAL_S so the client sees a live preview.
        """
        logging.info(f"Starting translation processing for client {self.client_uid}")
        pending = None
        last_partial_time = 0.0
        last_partial_text = None

        while not self.exit:
            try:
                segment = self.translation_queue.get(timeout=0.2)
                if segment is None:
                    logging.info(f"Received exit signal for translation client {self.client_uid}")
                    break
                self.translation_queue.task_done()

                if segment.get("completed", False):
                    pending = None
                    self.partial_segment = None
                    self.translated_segments.append(self.translate_segment(segment))
                    self.send_translation_to_client(self.prepare_translated_segments())
                    continue
                pending = segment
            except queue.Empty:
                pass

            try:
                if pending and pending["text"] == last_partial_text:
                    pending = None
                if pending and time.time() - last_partial_time >= self.PARTIAL_INTERVAL_S:
                    self.partial_segment = self.translate_segment(pending, num_beams=1)
                    last_partial_time = time.time()
                    last_partial_text = pending["text"]
                    pending = None
                    self.send_translation_to_client(self.prepare_translated_segments())
            except Exception as e:
                logging.error(f"Error processing translation queue: {e}")

        logging.info(f"Translation processing ended for client {self.client_uid}")

    def translate_segment(self, segment, num_beams=None):
        return {
            "start": segment["start"],
            "end": segment["end"],
            "text": self.translate_text(segment.get("text", ""), num_beams=num_beams),
            "completed": segment.get("completed", False),
            "target_language": self.target_language
        }

    def prepare_translated_segments(self):
        """
        Prepare the last n translated segments to send to client.
        
        Returns:
            list: List of recent translated segments
        """
        segments = self.translated_segments[-self.send_last_n_segments:]
        if self.partial_segment:
            segments.append(self.partial_segment)
        return segments
    
    def send_translation_to_client(self, translated_segments):
        """
        Send translated segments to the client via WebSocket.
        
        Args:
            translated_segments (list): List of translated segments to send
        """
        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "translated_segments": translated_segments,
                })
            )
        except Exception as e:
            logging.error(f"[ERROR]: Sending translation data to client: {e}")
    
    def speech_to_text(self):
        """
        Override parent method to handle translation processing.
        This method will be called when the translation thread starts.
        """
        self.process_translation_queue()
    
    def cleanup(self):
        """Clean up translation resources."""
        logging.info(f"Cleaning up translation resources for client {self.client_uid}")
        self.exit = True
        
        try:
            self.translation_queue.put(None, timeout=1.0)
        except:
            pass
        
        self.translated_segments.clear()
        
        if self.translation_model:
            del self.translation_model
            self.translation_model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if self.device and self.device.type == 'cuda':
            torch.cuda.empty_cache()
