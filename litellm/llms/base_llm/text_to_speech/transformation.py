from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.types.llms.openai import (
    AllMessageValues,
    HttpxBinaryResponseContent,
    OpenAIAudioSpeechOptionalParams,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseTextToSpeechConfig(BaseConfig, ABC):
    @abstractmethod
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioSpeechOptionalParams]:
        pass

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    @abstractmethod
    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[dict, bytes]:
        raise NotImplementedError(
            "TextToSpeechConfig needs a request transformation for text-to-speech models"
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        raise NotImplementedError(
            "TextToSpeechConfig does not need a request transformation for text-to-speech models"
        )

    @abstractmethod
    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: HttpxBinaryResponseContent,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
    ) -> HttpxBinaryResponseContent:
        raise NotImplementedError(
            "TextToSpeechConfig needs a response transformation for text-to-speech models"
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: Any,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> Any:
        raise NotImplementedError(
            "TextToSpeechConfig does not need a response transformation for text-to-speech models"
        )