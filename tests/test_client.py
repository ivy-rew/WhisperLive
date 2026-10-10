import json
import os
import scipy
import websocket
import copy
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock
from whisper_live.client import Client, TranscriptionClient, TranscriptionTeeClient
from whisper_live.utils import print_transcript, resample
from pathlib import Path


class BaseTestCase(unittest.TestCase):
    @patch('whisper_live.client.websocket.WebSocketApp')
    @patch('whisper_live.client.pyaudio.PyAudio')
    def setUp(self, mock_pyaudio, mock_websocket):
        self.mock_pyaudio_instance = MagicMock()
        mock_pyaudio.return_value = self.mock_pyaudio_instance
        self.mock_stream = MagicMock()
        self.mock_pyaudio_instance.open.return_value = self.mock_stream

        self.mock_ws_app = mock_websocket.return_value
        self.mock_ws_app.send = MagicMock()

        self.client = TranscriptionClient(host='localhost', port=9090, lang="en").client

        self.mock_pyaudio = mock_pyaudio
        self.mock_websocket = mock_websocket
        self.mock_audio_packet = b'\x00\x01\x02\x03'

    def tearDown(self):
        self.client.close_websocket()
        self.mock_pyaudio.stop()
        self.mock_websocket.stop()
        del self.client

class TestClientWebSocketCommunication(BaseTestCase):
    def test_websocket_communication(self):
        expected_url = 'ws://localhost:9090'
        self.mock_websocket.assert_called()
        self.assertEqual(self.mock_websocket.call_args[0][0], expected_url)


class TestTranscriptHighlight(unittest.TestCase):
    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_highlights_only_ongoing_plain_text(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((80, 20))

        print_transcript(["committed ", "still changing"], ongoing_index=1)

        output = mock_stdout.getvalue()
        self.assertIn("\033[30;43mstill changing\033[0m", output)
        self.assertNotIn("\033[30;43mcommitted", output)

    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_highlights_ongoing_text_not_timestamp(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((80, 20))

        print_transcript(
            [
                {"start": "1.000", "end": "2.000", "text": "earlier"},
                {"start": "2.000", "end": "3.000", "text": "just completed"},
                {"start": "3.000", "end": "4.000", "text": "still changing"},
            ],
            timestamps=True,
            ongoing_index=2,
            completed_indices=[1],
        )

        output = mock_stdout.getvalue()
        self.assertIn("[2.000 -> 3.000] \033[30;42mjust completed\033[0m", output)
        self.assertIn("[3.000 -> 4.000] \033[30;43mstill changing\033[0m", output)

    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_underlines_words_by_probability_band(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((80, 20))

        print_transcript(
            ["high medium low boundary"],
            words_by_segment=[[
                {"word": "high", "probability": 0.9},
                {"word": "medium", "probability": 0.5},
                {"word": "low", "probability": 0.1},
                {"word": "boundary", "probability": 0.6},
            ]],
        )

        output = mock_stdout.getvalue()
        self.assertIn("\033[1;4;58;5;1mlow\033[0m", output)
        self.assertIn("\033[4;58;5;9mmedium\033[0m", output)
        self.assertNotIn("\033[4;58;5;9mhigh", output)
        self.assertNotIn("\033[4;58;5;9mboundary", output)

    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_low_probability_underline_preserves_ongoing_background(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((80, 20))

        print_transcript(
            ["uncertain"],
            ongoing_index=0,
            words_by_segment=[[{"word": "uncertain", "probability": 0.2}]],
        )

        self.assertIn("\033[1;4;58;5;1;30;43muncertain\033[0m", mock_stdout.getvalue())


class TestLiveSegmentDisplay(BaseTestCase):
    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_highlights_incomplete_transcript_and_translation(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((80, 20))
        self.client.server_backend = "faster_whisper"
        self.client.enable_translation = True
        self.client.word_timestamps = True

        self.client.process_segments([
            {"start": "0.000", "end": "1.000", "text": "old transcript", "completed": True},
            {"start": "1.000", "end": "2.000", "text": "recent transcript one", "completed": True},
            {"start": "2.000", "end": "3.000", "text": "recent transcript two", "completed": True},
            {
                "start": "3.000",
                "end": "4.000",
                "text": "live transcript",
                "completed": False,
                "words": [{"word": "transcript", "probability": 0.2}],
            },
        ])
        self.client.process_segments([
            {"start": "0.000", "end": "1.000", "text": "ancienne traduction", "completed": True},
            {"start": "1.000", "end": "2.000", "text": "traduction recente un", "completed": True},
            {"start": "2.000", "end": "3.000", "text": "traduction recente deux", "completed": True},
            {"start": "3.000", "end": "4.000", "text": "traduction en cours", "completed": False},
        ], translated=True)

        output = mock_stdout.getvalue()
        self.assertIn("\033[30;42mrecent transcript onerecent transcript two\033[0m", output)
        self.assertNotIn("\033[30;42mold transcript\033[0m", output)
        self.assertIn("\033[30;43mlive \033[0m\033[1;4;58;5;1;30;43mtranscript\033[0m", output)
        self.assertIn("\033[30;42mtraduction recente un\033[0m", output)
        self.assertIn("\033[30;42mtraduction recente deux\033[0m", output)
        self.assertNotIn("\033[30;42mancienne traduction\033[0m", output)
        self.assertIn("\033[30;43mtraduction en cours\033[0m", output)


class TestClientCallbacks(BaseTestCase):
    def test_on_open(self):

        self.client.on_open(self.mock_ws_app)
        self.mock_ws_app.send.assert_called_once()
        sent_message = json.loads(self.mock_ws_app.send.call_args[0][0])
        self.assertEqual(sent_message["uid"], self.client.uid)
        self.assertEqual(sent_message["language"], self.client.language)
        self.assertEqual(sent_message["task"], self.client.task)
        self.assertEqual(sent_message["model"], self.client.model)
        self.assertTrue(sent_message["use_vad"])
        self.assertEqual(sent_message["send_last_n_segments"], 10)
        self.assertAlmostEqual(sent_message["no_speech_thresh"], 0.45)
        self.assertFalse(sent_message["clip_audio"])
        self.assertEqual(sent_message["same_output_threshold"], 10)
        self.assertFalse(sent_message["enable_translation"])
        self.assertEqual(sent_message["target_language"], "fr")
        self.assertIsNone(sent_message["hotwords"])
        self.assertFalse(sent_message["enable_diarization"])
        self.assertEqual(sent_message["max_speakers"], 10)
        self.assertFalse(sent_message["word_timestamps"])

    def test_on_message(self):
        message = json.dumps(
            {
                "uid": self.client.uid,
                "message": "SERVER_READY",
                "backend": "faster_whisper"
            }
        )
        self.client.on_message(self.mock_ws_app, message)

        message = json.dumps({
            "uid": self.client.uid,
            "segments": [
                {"start": 0, "end": 1, "text": "Test transcript", "completed": True},
                {"start": 1, "end": 2, "text": "Test transcript 2", "completed": True},
                {"start": 2, "end": 3, "text": "Test transcript 3", "completed": True}
            ]
        })
        self.client.on_message(self.mock_ws_app, message)

        # Assert that the transcript was updated correctly
        self.assertEqual(len(self.client.transcript), 3)
        self.assertEqual(self.client.transcript[1]['text'], "Test transcript 2")

    def test_on_close(self):
        close_status_code = 1000
        close_msg = "Normal closure"
        self.client.on_close(self.mock_ws_app, close_status_code, close_msg)

        self.assertFalse(self.client.recording)
        self.assertFalse(self.client.server_error)
        self.assertFalse(self.client.waiting)

    def test_on_error(self):
        error_message = "Test Error"
        self.client.on_error(self.mock_ws_app, error_message)

        self.assertTrue(self.client.server_error)
        self.assertEqual(self.client.error_message, error_message)


class TestAudioResampling(unittest.TestCase):
    def test_resample_audio(self):
        original_audio = "assets/jfk.flac"
        expected_sr = 16000
        resampled_audio = resample(original_audio, expected_sr)

        sr, _ = scipy.io.wavfile.read(resampled_audio)
        self.assertEqual(sr, expected_sr)

        os.remove(resampled_audio)


class TestPrintTranscript(unittest.TestCase):
    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_transcript_respects_narrow_terminals(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((20, 20))

        print_transcript(["This transcript should still wrap cleanly on a narrow terminal."])

        output_lines = [line for line in mock_stdout.getvalue().splitlines() if line.strip()]
        self.assertGreater(len(output_lines), 1)
        self.assertTrue(all(len(line) <= 20 for line in output_lines))

    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_transcript_indents_timestamp_continuations(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((32, 20))

        print_transcript(
            [{"start": "00:00", "end": "00:05", "text": "This line should wrap and keep its timestamp indentation."}],
            timestamps=True,
        )

        output_lines = [line.rstrip() for line in mock_stdout.getvalue().splitlines() if line.strip()]
        self.assertGreater(len(output_lines), 1)
        self.assertTrue(output_lines[1].startswith(" " * len("[00:00 -> 00:05] ")))

    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_transcript_honors_max_lines(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((80, 20))

        print_transcript([f"segment {i} of the running transcript. " for i in range(10)], max_lines=4)

        output_lines = [line for line in mock_stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(output_lines), 4)
        self.assertNotIn("segment 0", "".join(output_lines))

    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_transcript_defaults_to_three_lines(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((80, 20))

        print_transcript([f"segment {i} of the running transcript. " for i in range(10)])

        output_lines = [line for line in mock_stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(output_lines), 3)


class TestTranscriptDisplay(BaseTestCase):
    @patch("whisper_live.utils.shutil.get_terminal_size")
    @patch("sys.stdout", new_callable=StringIO)
    def test_display_segments_controls_printed_lines(self, mock_stdout, mock_terminal_size):
        mock_terminal_size.return_value = os.terminal_size((80, 20))
        self.client.display_segments = 8
        self.client.transcript = [
            {"start": str(i), "end": str(i + 1), "text": f"segment {i} of the running transcript. "}
            for i in range(8)
        ]

        self.client.process_segments([{"start": "8", "end": "9", "text": "tail", "completed": False}])

        output = mock_stdout.getvalue()
        output_lines = [line for line in output.splitlines() if line.strip()]
        self.assertGreater(len(output_lines), 3)
        self.assertIn("segment 0", output)


class TestSendingAudioPacket(BaseTestCase):
    def test_send_packet(self):
        self.client.send_packet_to_server(self.mock_audio_packet)
        self.client.client_socket.send.assert_called_with(self.mock_audio_packet, websocket.ABNF.OPCODE_BINARY)

    def test_send_packet_propagates_websocket_error(self):
        error = websocket.WebSocketConnectionClosedException("connection closed")
        self.client.client_socket.send.side_effect = error

        with self.assertRaises(websocket.WebSocketConnectionClosedException):
            self.client.send_packet_to_server(self.mock_audio_packet)

class TestTee(BaseTestCase):
    @patch('whisper_live.client.websocket.WebSocketApp')
    @patch('whisper_live.client.pyaudio.PyAudio')
    def setUp(self, mock_audio, mock_websocket):
        super().setUp()
        self.client2 = Client(host='localhost', port=9090, lang="es", translate=False, srt_file_path="transcript.srt")
        self.client3 = Client(host='localhost', port=9090, lang="es", translate=True, srt_file_path="translation.srt")
        # need a separate mock for each websocket
        self.client3.client_socket = copy.deepcopy(self.client3.client_socket)
        self.tee = TranscriptionTeeClient([self.client2, self.client3])

    def tearDown(self):
        self.tee.close_all_clients()
        del self.tee
        super().tearDown()

    def test_invalid_constructor(self):
        with self.assertRaises(Exception) as context:
            TranscriptionTeeClient([])

    def test_multicast_unconditional(self):
        self.tee.multicast_packet(self.mock_audio_packet, True)
        for client in self.tee.clients:
            client.client_socket.send.assert_called_with(self.mock_audio_packet, websocket.ABNF.OPCODE_BINARY)

    def test_multicast_conditional(self):
        self.client2.recording = False
        self.client3.recording = True
        self.tee.multicast_packet(self.mock_audio_packet, False)
        self.client2.client_socket.send.assert_not_called()
        self.client3.client_socket.send.assert_called_with(self.mock_audio_packet, websocket.ABNF.OPCODE_BINARY)

    def test_close_all(self):
        self.tee.close_all_clients()
        for client in self.tee.clients:
            client.client_socket.close.assert_called()

    def test_write_all_srt(self):
        for client in self.tee.clients:
            client.server_backend = "faster_whisper"
        self.tee.write_all_clients_srt()
        self.assertTrue(Path("transcript.srt").is_file())
        self.assertTrue(Path("translation.srt").is_file())
