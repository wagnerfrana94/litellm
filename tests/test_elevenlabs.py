import os
import sys
import unittest
import asyncio
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import litellm
from litellm.llms.elevenlabs.text_to_speech.handler import text_to_speech


class TestElevenLabs(unittest.TestCase):
    @patch("litellm.llms.elevenlabs.text_to_speech.handler.httpx.AsyncClient")
    async def test_elevenlabs_text_to_speech(self, mock_client):
        # Mock the response from ElevenLabs API
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Create a mock audio content with a specific size (160KB for approximately 10 seconds of audio at 128kbps)
        mock_audio_content = b"0" * 160 * 1024
        mock_response.content = mock_audio_content

        # Set up the mock client to return the mock response
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        # Call the text_to_speech function with test parameters
        response = await text_to_speech(
            model="eleven_multilingual_v2",
            input="Hello, this is a test of the ElevenLabs text-to-speech integration.",
            voice="Rachel",
            api_key="test_api_key",
        )

        # Assert that the response contains the expected content
        self.assertEqual(response.content, mock_audio_content)

        # Assert that the client was called with the expected parameters
        mock_client_instance.post.assert_called_once()
        call_args = mock_client_instance.post.call_args
        self.assertIn("url", call_args[1])
        self.assertIn("headers", call_args[1])
        self.assertIn("json", call_args[1])

        # Assert that the URL contains the voice ID
        self.assertIn("text-to-speech/Rachel", call_args[1]["url"])

        # Assert that the headers contain the API key
        self.assertEqual(call_args[1]["headers"]["xi-api-key"], "test_api_key")

        # Assert that the JSON payload contains the text to convert
        self.assertEqual(call_args[1]["json"]["text"], "Hello, this is a test of the ElevenLabs text-to-speech integration.")

        # Assert that the response includes the expected pricing information
        self.assertTrue(hasattr(response, '_hidden_params'))
        self.assertIn("audio_duration_seconds", response._hidden_params)
        self.assertIn("response_cost", response._hidden_params)
        self.assertIn("cost_per_second", response._hidden_params)

        # Assert that the duration is approximately 10 seconds (160KB / 16KB per second)
        self.assertAlmostEqual(response._hidden_params["audio_duration_seconds"], 10.0, delta=0.1)

        # Assert that the cost is calculated correctly (10 seconds * $0.00045 per second)
        expected_cost = 10.0 * 0.00045
        self.assertAlmostEqual(response._hidden_params["response_cost"], expected_cost, delta=0.0001)

        # Assert that the cost per second matches the value in model_prices_and_context_window_backup.json
        self.assertEqual(response._hidden_params["cost_per_second"], 0.00045)

    @patch("litellm.llms.elevenlabs.text_to_speech.handler.httpx.AsyncClient")
    async def test_elevenlabs_text_to_speech_via_litellm_speech(self, mock_client):
        """Test ElevenLabs TTS integration via litellm.speech function"""
        # Mock the response from ElevenLabs API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_audio_content = b"test_audio_content"
        mock_response.content = mock_audio_content

        # Set up the mock client to return the mock response
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        # Call the litellm.speech function with ElevenLabs model
        response = litellm.speech(
            model="elevenlabs/eleven_multilingual_v2",
            input="Hello from LiteLLM ElevenLabs integration!",
            voice="Rachel",
            api_key="test_api_key",
        )

        # Assert that the response contains the expected content
        self.assertEqual(response.content, mock_audio_content)

        # Assert that the client was called
        mock_client_instance.post.assert_called_once()
        call_args = mock_client_instance.post.call_args
        
        # Assert that the URL contains the voice ID
        self.assertIn("text-to-speech/Rachel", call_args[1]["url"])
        
        # Assert that the headers contain the API key
        self.assertEqual(call_args[1]["headers"]["xi-api-key"], "test_api_key")
        
        # Assert that the JSON payload contains the text to convert
        self.assertEqual(call_args[1]["json"]["text"], "Hello from LiteLLM ElevenLabs integration!")

    async def test_elevenlabs_text_to_speech_via_litellm_aspeech(self):
        """Test ElevenLabs TTS async integration via litellm.aspeech function"""
        with patch("litellm.llms.elevenlabs.text_to_speech.handler.httpx.AsyncClient") as mock_client:
            # Mock the response from ElevenLabs API
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_audio_content = b"test_audio_content_async"
            mock_response.content = mock_audio_content

            # Set up the mock client to return the mock response
            mock_client_instance = MagicMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value = mock_client_instance

            # Call the litellm.aspeech function with ElevenLabs model
            response = await litellm.aspeech(
                model="elevenlabs/eleven_turbo_v2",
                input="Async hello from LiteLLM ElevenLabs integration!",
                voice="Adam",
                api_key="test_api_key",
            )

            # Assert that the response contains the expected content
            self.assertEqual(response.content, mock_audio_content)

            # Assert that the client was called
            mock_client_instance.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
