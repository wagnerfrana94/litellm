"""
Handler for ElevenLabs text-to-speech API
"""
from typing import Any, Dict, Optional, Union

import httpx

from litellm.llms.base_llm.text_to_speech.transformation import LiteLLMLoggingObj
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.utils import print_verbose

from .transformation import ElevenLabsTextToSpeechConfig


class ElevenLabsTextToSpeech:
    def __init__(self):
        self.config = ElevenLabsTextToSpeechConfig()

    async def text_to_speech(
        self,
        model: str,
        input: str,
        voice: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        logging_obj: Optional[LiteLLMLoggingObj] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        optional_params: Optional[Dict[str, Any]] = None,
        client: Optional[httpx.Client] = None,
    ) -> HttpxBinaryResponseContent:
        """
        Converts text to speech using ElevenLabs API
        """
        if litellm_params is None:
            litellm_params = {}
        if optional_params is None:
            optional_params = {}

        # Get headers for the request
        headers = self.config.validate_environment(
            headers={},
            model=model,
            messages=[],  # Not used for text-to-speech
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

        # Transform the request to ElevenLabs format
        elevenlabs_params = self.config.transform_text_to_speech_request(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Get the complete URL for the request
        url = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=elevenlabs_params,
            litellm_params=litellm_params,
        )

        # Make the request to ElevenLabs API
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url=url,
                headers=headers,
                json=elevenlabs_params,
            )

            if response.status_code != 200:
                error_message = f"ElevenLabs API Error: {response.status_code} - {response.text}"
                print_verbose(error_message)
                raise self.config.get_error_class(
                    error_message=error_message,
                    status_code=response.status_code,
                    headers=response.headers,
                )

            # ElevenLabs returns audio bytes directly in response.content
            audio_bytes = response.content
            
            # Create httpx.Response object with the bytes
            mock_response = httpx.Response(
                status_code=200,
                content=audio_bytes,
                headers={"Content-Type": "audio/mpeg"}
            )
            
            # Create HttpxBinaryResponseContent object
            http_binary_response = HttpxBinaryResponseContent(mock_response)

            # Transform the response to include pricing information
            transformed_response = self.config.transform_text_to_speech_response(
                model=model,
                raw_response=response,
                model_response=http_binary_response,
                logging_obj=logging_obj,
                request_data={"input": input, "voice": voice},
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=api_key,
            )

            return transformed_response


# Create an instance of the handler
elevenlabs_text_to_speech = ElevenLabsTextToSpeech()


# Function to be called from litellm.speech
async def text_to_speech(
    model: str,
    input: str,
    voice: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    logging_obj: Optional[LiteLLMLoggingObj] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    litellm_params: Optional[Dict[str, Any]] = None,
    optional_params: Optional[Dict[str, Any]] = None,
    client: Optional[httpx.Client] = None,
) -> HttpxBinaryResponseContent:
    """
    Converts text to speech using ElevenLabs API
    """
    return await elevenlabs_text_to_speech.text_to_speech(
        model=model,
        input=input,
        voice=voice,
        api_key=api_key,
        api_base=api_base,
        logging_obj=logging_obj,
        timeout=timeout,
        litellm_params=litellm_params,
        optional_params=optional_params,
        client=client,
    )
