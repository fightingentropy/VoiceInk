import json
import os
import sys
import traceback

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MODEL = None
PROCESSOR = None
MODEL_ID = None
DEVICE = "cpu"
DTYPE = torch.float32
COMPILE_AVAILABLE = True
WORKER_VERSION = 1

torch.set_grad_enabled(False)


def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id):
    global MODEL, PROCESSOR, MODEL_ID, DEVICE, DTYPE

    if MODEL is not None and MODEL_ID == model_id:
        return

    DEVICE = detect_device()
    DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
    token = os.environ.get("HF_TOKEN")

    PROCESSOR = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
    )
    MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=DTYPE,
        token=token,
    )
    MODEL.to(DEVICE)
    MODEL.eval()
    MODEL_ID = model_id

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def extract_text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        if isinstance(output.get("text"), str):
            return output["text"]
        if isinstance(output.get("texts"), list) and output["texts"]:
            first = output["texts"][0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                return first["text"]
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict) and isinstance(first.get("text"), str):
            return first["text"]
    return str(output)


def transcribe(audio_path, language, prefer_compile):
    global COMPILE_AVAILABLE

    kwargs = {
        "processor": PROCESSOR,
        "audio_files": [audio_path],
        "language": language,
    }

    if DEVICE != "cpu" and COMPILE_AVAILABLE and prefer_compile:
        try:
            with torch.inference_mode():
                output = MODEL.transcribe(compile=True, **kwargs)
            return extract_text(output), True
        except Exception:
            COMPILE_AVAILABLE = False

    with torch.inference_mode():
        output = MODEL.transcribe(compile=False, **kwargs)
    return extract_text(output), False


def handle_request(request):
    request_id = request["id"]
    command = request["command"]

    if command == "load":
        load_model(request["model"])
        emit(
            {
                "id": request_id,
                "ok": True,
                "result": {
                    "device": DEVICE,
                    "compile_enabled": COMPILE_AVAILABLE,
                    "model": MODEL_ID,
                    "worker_version": WORKER_VERSION,
                },
            }
        )
        return

    if command in ("warmup", "transcribe"):
        load_model(request["model"])
        text, compile_enabled = transcribe(
            request["audioPath"],
            request["language"],
            bool(request.get("preferCompile")),
        )
        emit(
            {
                "id": request_id,
                "ok": True,
                "result": {
                    "text": text,
                    "device": DEVICE,
                    "compile_enabled": compile_enabled,
                    "model": MODEL_ID,
                    "worker_version": WORKER_VERSION,
                },
            }
        )
        return

    emit({"id": request_id, "ok": False, "error": {"message": f"Unknown command: {command}"}})


def main():
    emit({"event": "ready", "worker_version": WORKER_VERSION})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            handle_request(request)
        except Exception as exc:
            request_id = None
            try:
                request_id = request.get("id")  # type: ignore[name-defined]
            except Exception:
                request_id = None

            emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error": {
                        "message": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                }
            )


if __name__ == "__main__":
    main()
