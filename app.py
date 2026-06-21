from pathlib import Path
import tempfile
import time

from flask import Flask, jsonify, render_template, request

from asr_local import transcribe_audio_file, warmup_model
from inference import LatexGenerator


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

print("Loading LaTeX model...")
generator = LatexGenerator(
    model_dir=BASE_DIR / "project" / "artifacts" / "byt5_base_mathbridge_ru_medium_clean" / "final_model"
)

print("Loading local ASR model...")
warmup_model()

print("Warming up LaTeX generation...")
_ = generator.generate("икс плюс один")

print("All models are ready.")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/transcribe")
def api_transcribe():
    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "Файл audio не передан."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"ok": False, "error": "Пустое имя файла."}), 400

    suffix = Path(audio_file.filename).suffix or ".webm"

    temp_path = None
    started = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
            audio_file.save(tmp)
            temp_path = Path(tmp.name)

        text = transcribe_audio_file(temp_path)
        elapsed = time.perf_counter() - started

        return jsonify(
            {
                "ok": True,
                "text": text,
                "seconds": round(elapsed, 3),
                "filename": audio_file.filename,
                "asr_backend": "faster-whisper",
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@app.post("/api/latex")
def api_latex():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"ok": False, "error": "Пустой входной текст."}), 400

    started = time.perf_counter()

    try:
        latex = generator.generate(text)
        elapsed = time.perf_counter() - started

        return jsonify(
            {
                "ok": True,
                "input_text": text,
                "latex": latex,
                "seconds": round(elapsed, 3),
                "model_name": generator.model_name,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/transcribe-and-latex")
def api_transcribe_and_latex():
    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "Файл audio не передан."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"ok": False, "error": "Пустое имя файла."}), 400

    suffix = Path(audio_file.filename).suffix or ".webm"

    temp_path = None
    started = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
            audio_file.save(tmp)
            temp_path = Path(tmp.name)

        text = transcribe_audio_file(temp_path)
        latex = generator.generate(text)
        elapsed = time.perf_counter() - started

        return jsonify(
            {
                "ok": True,
                "text": text,
                "latex": latex,
                "seconds": round(elapsed, 3),
                "model_name": generator.model_name,
                "filename": audio_file.filename,
                "asr_backend": "faster-whisper",
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)