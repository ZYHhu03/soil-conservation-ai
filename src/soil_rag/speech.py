"""ASR/TTS service adapters and voice-to-RAG orchestration."""

from __future__ import annotations

import base64
import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .pipeline import RAGPipeline


class ASRClient(Protocol):
    def transcribe(self, audio: bytes, content_type: str = "audio/wav") -> str: ...


class TTSClient(Protocol):
    def synthesize(self, text: str, voice: str = "default") -> bytes: ...


@dataclass
class HTTPASRClient:
    endpoint: str
    api_key: str = ""

    def transcribe(self, audio: bytes, content_type: str = "audio/wav") -> str:
        payload = json.dumps({"audio": base64.b64encode(audio).decode(), "content_type": content_type}).encode()
        request = urllib.request.Request(self.endpoint, data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode())
        return str(body.get("text", body.get("transcript", "")))


@dataclass
class HTTPTTSClient:
    endpoint: str
    api_key: str = ""

    def synthesize(self, text: str, voice: str = "default") -> bytes:
        payload = json.dumps({"text": text, "voice": voice}).encode()
        request = urllib.request.Request(self.endpoint, data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode())
        encoded = body.get("audio_base64", body.get("audio", ""))
        return base64.b64decode(encoded)


class VoiceRAGService:
    def __init__(self, pipeline: RAGPipeline, asr: ASRClient, tts: TTSClient):
        self.pipeline = pipeline
        self.asr = asr
        self.tts = tts

    def invoke(self, audio: bytes, content_type: str = "audio/wav", voice: str = "default") -> tuple[str, object, bytes]:
        query = self.asr.transcribe(audio, content_type)
        response = self.pipeline.invoke(query)
        speech = self.tts.synthesize(response.answer, voice)
        return query, response, speech
