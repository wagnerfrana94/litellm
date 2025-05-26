# ElevenLabs Integration for LiteLLM

This integration allows you to use ElevenLabs' text-to-speech API through LiteLLM.

## Features

- Convert text to speech using ElevenLabs' high-quality voices
- Compatible with OpenAI's text-to-speech API format
- Supports various voice options and speech parameters

## Usage

### Environment Variables

Set the following environment variables:

```bash
ELEVENLABS_API_KEY=your_api_key
ELEVENLABS_API_BASE=https://api.elevenlabs.io/v1  # Optional, defaults to this value
ELEVENLABS_API_BASE=https://api.elevenlabs.io/v1  # Optional, defaults to this value
```

### Basic Usage

```python
import litellm

# Convert text to speech
response = litellm.speech(
    model="eleven_multilingual_v2",  # ElevenLabs model
    input="Hello, this is a test of the ElevenLabs text-to-speech integration.",
    voice="Rachel",  # ElevenLabs voice ID or name
    custom_llm_provider="elevenlabs",
)

# Save the audio to a file
with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### Async Usage

```python
import litellm
import asyncio

async def main():
    # Convert text to speech asynchronously
    response = await litellm.aspeech(
        model="eleven_multilingual_v2",
        input="Hello, this is a test of the ElevenLabs text-to-speech integration.",
        voice="Rachel",
        custom_llm_provider="elevenlabs",
    )
    
    # Save the audio to a file
    with open("output.mp3", "wb") as f:
        f.write(response.content)

asyncio.run(main())
```

### Using with Proxy Server

The ElevenLabs integration works with the LiteLLM proxy server. You can use the same endpoints as OpenAI's text-to-speech API:

```bash
curl -X POST http://localhost:4000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "eleven_multilingual_v2",
    "input": "Hello, this is a test of the ElevenLabs text-to-speech integration.",
    "voice": "Rachel",
    "custom_llm_provider": "elevenlabs"
  }'
```

## Parameters

- `model` (str): The ElevenLabs model to use (e.g., "eleven_multilingual_v2")
- `input` (str): The text to convert to speech
- `voice` (str): The voice ID or name to use
- `speed` (float, optional): The speed of the speech (default is 1.0)
- `response_format` (str, optional): The format of the response (default is "mp3")
- `api_key` (str, optional): Your ElevenLabs API key (if not set in environment variables)
- `api_base` (str, optional): The base URL for the ElevenLabs API (if not using the default)
- `custom_llm_provider` (str): Set to "elevenlabs" to use this integration

## Available Models

- `eleven_monolingual_v1`: Original monolingual model
- `eleven_multilingual_v1`: Original multilingual model
- `eleven_multilingual_v2`: Latest multilingual model with improved quality
- `eleven_turbo_v2`: Fastest model with good quality

## Available Voices

You can use any voice ID from your ElevenLabs account. Some examples:

- `Rachel`: Professional female voice
- `Domi`: Professional male voice
- `Bella`: Professional female voice
- `Antoni`: Professional male voice
- `Elli`: Professional female voice

For a complete list of voices, visit your ElevenLabs dashboard or use their API.