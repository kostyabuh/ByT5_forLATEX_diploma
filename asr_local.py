from pathlib import Path
from threading import Lock

from faster_whisper import WhisperModel


ASR_MODEL_SIZE = "small"
ASR_DEVICE = "cpu"
ASR_COMPUTE_TYPE = "int8"
ASR_LANGUAGE = "ru"
ASR_BEAM_SIZE = 1
ASR_USE_VAD = False

_model = None
_model_lock = Lock()


def get_model() -> WhisperModel:
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(
                    ASR_MODEL_SIZE,
                    device=ASR_DEVICE,
                    compute_type=ASR_COMPUTE_TYPE,
                )

    return _model


def warmup_model() -> None:
    _ = get_model()


def transcribe_audio_file(audio_path: str | Path) -> str:
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Не найден аудиофайл: {audio_path}")

    model = get_model()

    segments, _ = model.transcribe(
        str(audio_path),
        language=ASR_LANGUAGE,
        task="transcribe",
        beam_size=ASR_BEAM_SIZE,
        vad_filter=ASR_USE_VAD,
    )

    text_parts = []
    for segment in segments:
        part = (segment.text or "").strip()
        if part:
            text_parts.append(part)

    text = " ".join(text_parts).strip()
    if not text:
        raise RuntimeError("Локальная ASR-модель вернула пустой текст.")

    return text