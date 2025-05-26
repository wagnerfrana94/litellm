"""
Translates from OpenAI's `/v1/audio/speech` to ElevenLabs' text-to-speech API
"""

from typing import List, Optional, Union

from httpx import Headers, Response

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    HttpxBinaryResponseContent,
    OpenAIAudioSpeechOptionalParams,
)

from ...base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    LiteLLMLoggingObj,
)
from ..common_utils import ElevenLabsException


class ElevenLabsTextToSpeechConfig(BaseTextToSpeechConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioSpeechOptionalParams]:
        return ["voice", "speed", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return ElevenLabsException(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> dict:
        """
        Transforms the OpenAI text-to-speech request to ElevenLabs format
        """
        # Map OpenAI parameters to ElevenLabs parameters
        elevenlabs_params = {}

        # Map voice parameter
        elevenlabs_params["voice_id"] = voice

        # Map speed parameter if provided
        if "speed" in optional_params:
            elevenlabs_params["model_id"] = model
            elevenlabs_params["speed"] = optional_params.get("speed", 1.0)

        # Map response_format parameter if provided
        if "response_format" in optional_params:
            format_mapping = {
                "mp3": "mp3",
                "opus": "mp3",  # ElevenLabs might not support opus directly
                "aac": "mp3",   # ElevenLabs might not support aac directly
                "flac": "mp3",  # ElevenLabs might not support flac directly
            }
            elevenlabs_params["output_format"] = format_mapping.get(
                optional_params.get("response_format", "mp3"), "mp3"
            )

        # Add the text to convert
        elevenlabs_params["text"] = input

        return elevenlabs_params

    def transform_text_to_speech_response(
            self,
            model: str,
            raw_response: Response,
            model_response: HttpxBinaryResponseContent,
            logging_obj: LiteLLMLoggingObj,
            request_data: dict,
            optional_params: dict,
            litellm_params: dict,
            api_key: Optional[str] = None,
    ) -> HttpxBinaryResponseContent:
        """
        Transforms the raw response from ElevenLabs to HttpxBinaryResponseContent,
        adding character count information for cost calculation based on the model's
        defined pricing in model_prices_and_context_window.json.
        """

        # Get the input text from request_data
        input_text = request_data.get("input", "")

        # Get character count for cost calculation
        character_count = len(input_text)

        # Create _hidden_params if it doesn't exist
        if not hasattr(model_response, '_hidden_params'):
            model_response._hidden_params = {}

        # Set prompt_characters for the cost calculation system to use
        model_response._hidden_params["prompt_characters"] = character_count

        # Create basic logging information for cost calculation
        # The LiteLLM system will handle the standard logging automatically
        model_response._hidden_params["model"] = model
        model_response._hidden_params["api_key"] = api_key


        return model_response

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            api_base = (
                get_secret_str("ELEVENLABS_API_BASE") or "https://api.elevenlabs.io/v1"
            )
        api_base = api_base.rstrip("/")  # Remove trailing slash if present

        # Get the voice_id from optional_params
        voice_id = optional_params.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Default voice ID

        return f"{api_base}/text-to-speech/{voice_id}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        api_key = api_key or get_secret_str("ELEVENLABS_API_KEY")
        if api_key is None:
            raise ElevenLabsException(
                message="Missing ElevenLabs API key. Set the ELEVENLABS_API_KEY environment variable or pass it as api_key.",
                status_code=401,
                headers={},
            )
        return {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
