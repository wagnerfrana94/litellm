"""
ElevenLabs Voice Management Handler
Handles voice creation, deletion, and listing operations
"""
from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.utils import print_verbose

from ..common_utils import ElevenLabsException


class ElevenLabsVoiceManager:
    def __init__(self):
        self.api_base = "https://api.elevenlabs.io/v1"

    def _get_headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        """Get headers for ElevenLabs API requests"""
        # Try multiple sources for the API key
        api_key = (
            api_key or 
            get_secret_str("ELEVENLABS_API_KEY") or 
            get_secret_str("ELEVEN_LABS_API_KEY") or  # Alternative naming
            None
        )
        
        if not api_key:
            raise ElevenLabsException(
                message="Missing ElevenLabs API key. Set ELEVENLABS_API_KEY environment variable or pass api_key parameter.",
                status_code=401,
                headers={},
            )
        
        return {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }

    async def create_voice(
        self,
        name: str,
        files: List[bytes],  # Audio files content
        description: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new voice clone from audio files
        
        Args:
            name: Name for the voice
            files: List of audio file contents (bytes)
            description: Optional description for the voice
            api_key: Optional API key
            timeout: Request timeout
            
        Returns:
            Dict containing voice creation response
        """
        headers = self._get_headers(api_key)
        url = f"{self.api_base}/voices/add"
        
        # Prepare multipart form data
        form_data = {
            "name": name,
        }
        
        if description:
            form_data["description"] = description
        
        # Add files to form data
        files_data = []
        for i, file_content in enumerate(files):
            files_data.append(
                ("files", (f"audio_{i}.mp3", file_content, "audio/mpeg"))
            )
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url=url,
                headers=headers,
                data=form_data,
                files=files_data,
            )
            
            if response.status_code not in [200, 201]:
                error_message = f"ElevenLabs Voice Creation Error: {response.status_code} - {response.text}"
                print_verbose(error_message)
                raise ElevenLabsException(
                    message=error_message,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            
            return response.json()

    async def delete_voice(
        self,
        voice_id: str,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Dict[str, Any]:
        """
        Delete a voice by ID
        
        Args:
            voice_id: ID of the voice to delete
            api_key: Optional API key
            timeout: Request timeout
            
        Returns:
            Dict containing deletion status
        """
        headers = self._get_headers(api_key)
        url = f"{self.api_base}/voices/{voice_id}"
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.delete(
                url=url,
                headers=headers,
            )
            
            if response.status_code != 200:
                error_message = f"ElevenLabs Voice Deletion Error: {response.status_code} - {response.text}"
                print_verbose(error_message)
                raise ElevenLabsException(
                    message=error_message,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            
            return response.json()

    async def get_voice(
        self,
        voice_id: str,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Dict[str, Any]:
        """
        Get voice information by ID
        
        Args:
            voice_id: ID of the voice to retrieve
            api_key: Optional API key
            timeout: Request timeout
            
        Returns:
            Dict containing voice information
        """
        headers = self._get_headers(api_key)
        url = f"{self.api_base}/voices/{voice_id}"
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url=url,
                headers=headers,
            )
            
            if response.status_code != 200:
                error_message = f"ElevenLabs Voice Retrieval Error: {response.status_code} - {response.text}"
                print_verbose(error_message)
                raise ElevenLabsException(
                    message=error_message,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            
            return response.json()

    async def list_voices(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Dict[str, Any]:
        """
        List all available voices
        
        Args:
            api_key: Optional API key
            timeout: Request timeout
            
        Returns:
            Dict containing list of voices
        """
        headers = self._get_headers(api_key)
        url = f"{self.api_base}/voices"
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url=url,
                headers=headers,
            )
            
            if response.status_code != 200:
                error_message = f"ElevenLabs Voice List Error: {response.status_code} - {response.text}"
                print_verbose(error_message)
                raise ElevenLabsException(
                    message=error_message,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            
            return response.json()


# Create global instance
elevenlabs_voice_manager = ElevenLabsVoiceManager()


# Public functions to be used by the routing system
async def create_voice(
    name: str,
    files: List[bytes],
    description: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> Dict[str, Any]:
    """Create a new voice clone"""
    return await elevenlabs_voice_manager.create_voice(
        name=name,
        files=files,
        description=description,
        api_key=api_key,
        timeout=timeout,
    )


async def delete_voice(
    voice_id: str,
    api_key: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> Dict[str, Any]:
    """Delete a voice by ID"""
    return await elevenlabs_voice_manager.delete_voice(
        voice_id=voice_id,
        api_key=api_key,
        timeout=timeout,
    )


async def get_voice(
    voice_id: str,
    api_key: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> Dict[str, Any]:
    """Get voice information by ID"""
    return await elevenlabs_voice_manager.get_voice(
        voice_id=voice_id,
        api_key=api_key,
        timeout=timeout,
    )


async def list_voices(
    api_key: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> Dict[str, Any]:
    """List all available voices"""
    return await elevenlabs_voice_manager.list_voices(
        api_key=api_key,
        timeout=timeout,
    )