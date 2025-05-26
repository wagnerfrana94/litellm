import base64
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestElevenLabsVoiceManagement(unittest.TestCase):
    
    @patch("litellm.llms.elevenlabs.voice_management.handler.httpx.AsyncClient")
    async def test_create_voice(self, mock_client):
        """Test creating a new voice clone"""
        from litellm.llms.elevenlabs.voice_management.handler import create_voice
        
        # Mock the response from ElevenLabs API
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "voice_id": "test-voice-id-123",
            "name": "Test Voice",
            "description": "A test voice",
            "status": "ready"
        }
        
        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        # Test audio file content
        test_audio = b"fake_audio_content"
        
        # Call the create_voice function
        result = await create_voice(
            name="Test Voice",
            files=[test_audio],
            description="A test voice",
            api_key="test_api_key",
        )
        
        # Assert that the response contains expected data
        self.assertEqual(result["voice_id"], "test-voice-id-123")
        self.assertEqual(result["name"], "Test Voice")
        self.assertEqual(result["status"], "ready")
        
        # Assert that the client was called correctly
        mock_client_instance.post.assert_called_once()
        call_args = mock_client_instance.post.call_args
        
        # Check URL
        self.assertEqual(call_args[1]["url"], "https://api.elevenlabs.io/v1/voices/add")
        
        # Check headers
        self.assertEqual(call_args[1]["headers"]["xi-api-key"], "test_api_key")
        
        # Check form data
        self.assertEqual(call_args[1]["data"]["name"], "Test Voice")
        self.assertEqual(call_args[1]["data"]["description"], "A test voice")

    @patch("litellm.llms.elevenlabs.voice_management.handler.httpx.AsyncClient")
    async def test_delete_voice(self, mock_client):
        """Test deleting a voice by ID"""
        from litellm.llms.elevenlabs.voice_management.handler import delete_voice
        
        # Mock the response from ElevenLabs API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        
        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.delete.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        voice_id = "test-voice-id-123"
        
        # Call the delete_voice function
        result = await delete_voice(
            voice_id=voice_id,
            api_key="test_api_key",
        )
        
        # Assert that the response contains expected data
        self.assertEqual(result["status"], "ok")
        
        # Assert that the client was called correctly
        mock_client_instance.delete.assert_called_once()
        call_args = mock_client_instance.delete.call_args
        
        # Check URL
        self.assertEqual(call_args[1]["url"], f"https://api.elevenlabs.io/v1/voices/{voice_id}")
        
        # Check headers
        self.assertEqual(call_args[1]["headers"]["xi-api-key"], "test_api_key")

    @patch("litellm.llms.elevenlabs.voice_management.handler.httpx.AsyncClient")
    async def test_get_voice(self, mock_client):
        """Test getting voice information by ID"""
        from litellm.llms.elevenlabs.voice_management.handler import get_voice
        
        # Mock the response from ElevenLabs API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "voice_id": "test-voice-id-123",
            "name": "Test Voice",
            "description": "A test voice",
            "samples": [
                {"sample_id": "sample1", "file_name": "sample1.mp3"},
                {"sample_id": "sample2", "file_name": "sample2.mp3"}
            ]
        }
        
        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        voice_id = "test-voice-id-123"
        
        # Call the get_voice function
        result = await get_voice(
            voice_id=voice_id,
            api_key="test_api_key",
        )
        
        # Assert that the response contains expected data
        self.assertEqual(result["voice_id"], "test-voice-id-123")
        self.assertEqual(result["name"], "Test Voice")
        self.assertEqual(len(result["samples"]), 2)
        
        # Assert that the client was called correctly
        mock_client_instance.get.assert_called_once()
        call_args = mock_client_instance.get.call_args
        
        # Check URL
        self.assertEqual(call_args[1]["url"], f"https://api.elevenlabs.io/v1/voices/{voice_id}")
        
        # Check headers
        self.assertEqual(call_args[1]["headers"]["xi-api-key"], "test_api_key")

    @patch("litellm.llms.elevenlabs.voice_management.handler.httpx.AsyncClient")
    async def test_list_voices(self, mock_client):
        """Test listing all available voices"""
        from litellm.llms.elevenlabs.voice_management.handler import list_voices
        
        # Mock the response from ElevenLabs API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "voices": [
                {
                    "voice_id": "voice1",
                    "name": "Voice One",
                    "category": "premade"
                },
                {
                    "voice_id": "voice2", 
                    "name": "Voice Two",
                    "category": "cloned"
                }
            ]
        }
        
        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        # Call the list_voices function
        result = await list_voices(api_key="test_api_key")
        
        # Assert that the response contains expected data
        self.assertEqual(len(result["voices"]), 2)
        self.assertEqual(result["voices"][0]["voice_id"], "voice1")
        self.assertEqual(result["voices"][1]["voice_id"], "voice2")
        
        # Assert that the client was called correctly
        mock_client_instance.get.assert_called_once()
        call_args = mock_client_instance.get.call_args
        
        # Check URL
        self.assertEqual(call_args[1]["url"], "https://api.elevenlabs.io/v1/voices")
        
        # Check headers
        self.assertEqual(call_args[1]["headers"]["xi-api-key"], "test_api_key")

    def test_create_voice_http_endpoint(self):
        """Test the HTTP endpoint for creating voices"""
        # Test the request format for the HTTP endpoint
        test_audio_b64 = base64.b64encode(b"fake_audio_content").decode('utf-8')
        
        expected_request = {
            "name": "Test Voice",
            "description": "A test voice",
            "files": [test_audio_b64]
        }
        
        # Test that base64 encoding/decoding works correctly
        decoded_audio = base64.b64decode(test_audio_b64)
        self.assertEqual(decoded_audio, b"fake_audio_content")
        
        # Test request validation
        self.assertIn("name", expected_request)
        self.assertIn("files", expected_request)
        self.assertTrue(len(expected_request["files"]) > 0)

    def test_voice_management_error_handling(self):
        """Test error handling for voice management operations"""
        from litellm.llms.elevenlabs.voice_management.handler import ElevenLabsVoiceManager
        
        # Test missing API key handling
        manager = ElevenLabsVoiceManager()
        
        with patch('litellm.secret_managers.main.get_secret_str', return_value=None):
            with self.assertRaises(Exception) as context:
                manager._get_headers()
            
            self.assertIn("Missing ElevenLabs API key", str(context.exception))


if __name__ == "__main__":
    unittest.main()