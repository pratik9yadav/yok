"""The ears (speech to text) and the mouth (text to speech) of the voice bot, and audio as a phone line makes it.

Heavy libraries (torch, faster-whisper, kokoro, parler-tts) are imported only when a model is loaded, so this file
can be read and its plain parts tested without them.

    ears = Ears.load("faster-whisper")              or "transformers", "indic-conformer" (needs an HF token)
    text, lang, seconds = ears.hear(samples, hint="hi", allowed=("hi", "mr", "en"))
    mouth = Mouth.load("kokoro")                     or "parler" (needs an HF token), "mms" (non-commercial: test only)
    samples, rate, seconds = mouth.say("नमस्ते", "hi", voice="bot")
    phone_line(samples)                              what a phone call does to a voice: 8 kHz, 300–3400 Hz, μ-law, noise
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

RATE = 16000  # what speech to text takes
CACHE_LINES = 256  # lines of speech kept, so a task's greeting and questions are said at once

# Every model, its licence, and whether a business may use it. The results always say which one ran.
ASR_INFO = {
    "faster-whisper": {"model": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "licence": "MIT", "business": True},
    "transformers": {"model": "openai/whisper-large-v3-turbo", "licence": "MIT", "business": True},
    "indic-conformer": {"model": "ai4bharat/indic-conformer-600m-multilingual", "licence": "MIT", "business": True,
                        "token": True},
}
TTS_INFO = {
    "kokoro": {"model": "hexgrad/Kokoro-82M (kokoro-onnx)", "licence": "Apache-2.0", "business": True},
    "parler": {"model": "ai4bharat/indic-parler-tts", "licence": "Apache-2.0", "business": True, "token": True},
    "mms": {"model": "facebook/mms-tts-hin, facebook/mms-tts-mar", "licence": "CC-BY-NC-4.0", "business": False},
}
KOKORO_FILES = {  # Kokoro-82M as ONNX, from the kokoro-onnx project's release
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}
# Voices: the bot is one calm woman's voice; test callers use other voices.
KOKORO_VOICES = {"bot": "hf_alpha", "person": "hm_omega", "person2": "hf_beta", "person3": "hm_psi"}
KOKORO_EN = {"bot": "af_heart", "person": "am_adam", "person2": "bf_emma", "person3": "bm_george"}
PARLER_SPEAKERS = {  # Indic Parler-TTS speakers by language
    "hi": {"bot": "Divya", "person": "Rohit", "person2": "Aman", "person3": "Rani"},
    "mr": {"bot": "Sunita", "person": "Sanjay", "person2": "Nikhil", "person3": "Radha"},
    "en": {"bot": "Mary", "person": "Thoma", "person2": "Swapna", "person3": "Dinesh"},
}
PARLER_STYLE = {
    "bot": "speaks in a clear, calm and polite voice at a moderate pace. The recording is very clear, with no "
           "background noise.",
    "person": "speaks quickly in a casual tone. The recording is close, with a little background noise.",
}
LANG_NAMES = {"hi": "Hindi", "mr": "Marathi", "en": "English"}


def download(url: str, path: str) -> None:
    import urllib.request

    part = path + ".part"
    with urllib.request.urlopen(url, timeout=600) as resp, open(part, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    os.replace(part, path)


def ffmpeg_read(path: str):
    """Any audio file (wav, mp3, m4a, ogg, opus, webm) as 16 kHz mono float32 samples."""
    import numpy as np
    import soundfile as sf

    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "a.wav")
            done = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path, "-ac", "1", "-ar",
                                   str(RATE), wav], capture_output=True, text=True, timeout=120, check=False)
            if done.returncode != 0:
                raise RuntimeError(f"ffmpeg could not read {os.path.basename(path)}: {done.stderr[-300:]}")
            audio, _ = sf.read(wav, dtype="float32")
        return audio
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), rate, RATE).astype(np.float32)


def resample(audio, rate: int, to: int):
    import numpy as np

    if rate == to:
        return np.asarray(audio, dtype=np.float32)
    from scipy.signal import resample_poly

    g = np.gcd(int(rate), int(to))
    return resample_poly(np.asarray(audio, dtype=np.float32), to // g, rate // g).astype(np.float32)


def as_samples(audio, rate: int | None = None):
    """Samples at 16 kHz mono from a file path, (rate, samples) as Gradio gives them, or samples at rate."""
    import numpy as np

    if isinstance(audio, str):
        return ffmpeg_read(audio)
    if isinstance(audio, tuple):
        rate, audio = audio
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1) if audio.shape[1] <= 2 else audio.mean(axis=0)
    if audio.dtype.kind == "i":
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    return resample(audio.astype(np.float32), rate or RATE, RATE)


def phone_line(audio, rate: int = RATE, snr_db: float = 20.0, seed: int = 0):
    """The voice as a phone call carries it: 8 kHz, only 300–3400 Hz, 8-bit μ-law, with some line noise (snr_db),
    back at the input rate. Speech to text must work on this, not only on clean studio audio."""
    import numpy as np
    from scipy.signal import butter, sosfilt

    x = resample(audio, rate, 8000)
    x = sosfilt(butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos"), x)
    peak = float(np.max(np.abs(x))) or 1.0
    x = x / peak * 0.9
    mu = 255.0  # μ-law: what the phone network does to each sample
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round(y * 127) / 127
    x = np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu
    rng = np.random.default_rng(seed)
    power = float(np.mean(x ** 2)) or 1e-9
    x = x + rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), size=x.shape)
    return np.clip(resample(x.astype(np.float32), 8000, rate), -1, 1)


def to_int16(audio):
    import numpy as np

    return (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype(np.int16)


NO_REPEAT = 6  # Whisper can loop on a short clip ("कर दो कर दो कर दो ..."): no 6-token phrase twice


def token_limit(audio) -> int:
    """At most this many tokens for a clip: about 30 per second of speech (Devanagari takes many; 15 cut real
    sentences short in eval 4), so a hallucination loop can't run on for long."""
    return int(min(440, 30 + 30 * len(audio) / RATE))


def preload_cuda_libraries() -> None:
    """faster-whisper (CTranslate2) looks for cuDNN and cuBLAS on the system path; on Kaggle they come with torch's
    pip packages instead. Loading them first lets it find them."""
    import ctypes
    import glob
    import site

    for base in site.getsitepackages():
        for folder in ("cublas", "cudnn"):
            for lib in sorted(glob.glob(os.path.join(base, "nvidia", folder, "lib", "lib*.so*"))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


class Ears:
    """Speech to text. hear() picks the language among the allowed ones (Whisper alone may say Urdu for Hindi)."""

    def __init__(self, backend: str, run, detect=None, device: str = "?"):
        self.backend, self.run, self.detect, self.device = backend, run, detect, device
        self.info = ASR_INFO[backend]

    @classmethod
    def load(cls, backend: str, token: str | None = None) -> "Ears":
        import torch

        gpu = torch.cuda.is_available()
        if backend == "faster-whisper":
            if gpu:
                preload_cuda_libraries()
            from faster_whisper import WhisperModel

            model = WhisperModel(ASR_INFO[backend]["model"], device="cuda" if gpu else "cpu",
                                 compute_type="float16" if gpu else "int8")

            def detect(audio):
                _, _, probs = model.detect_language(audio)
                return dict(probs)

            def run(audio, lang, prompt):
                segments, _ = model.transcribe(audio, language=lang, beam_size=1, vad_filter=True,
                                               initial_prompt=prompt or None, condition_on_previous_text=False,
                                               without_timestamps=True, no_repeat_ngram_size=NO_REPEAT,
                                               max_new_tokens=token_limit(audio))
                return " ".join(s.text.strip() for s in segments)

            import numpy as np

            run(np.zeros(RATE, dtype=np.float32), "hi", None)  # some CUDA problems only show on the first use
            return cls(backend, run, detect, "GPU" if gpu else "CPU")
        if backend == "transformers":
            from transformers import pipeline

            asr = pipeline("automatic-speech-recognition", model=ASR_INFO[backend]["model"],
                           device="cuda:0" if gpu else "cpu", torch_dtype=torch.float16 if gpu else torch.float32,
                           chunk_length_s=30)  # clips over 30 s too

            def run(audio, lang, prompt):
                options = {"task": "transcribe", "max_new_tokens": token_limit(audio), "no_repeat_ngram_size": NO_REPEAT,
                           **({"language": lang} if lang else {})}
                if prompt and len(audio) <= 30 * RATE:  # a hint (the bot's question) helps with short answers
                    try:
                        ids = asr.tokenizer.get_prompt_ids(prompt, return_tensors="pt").to(asr.device)
                        return asr({"raw": audio, "sampling_rate": RATE},
                                   generate_kwargs={**options, "prompt_ids": ids})["text"]
                    except Exception as e:  # noqa: BLE001 - not supported here: without the hint
                        print("speech-to-text hint not used:", repr(e)[:200])
                return asr({"raw": audio, "sampling_rate": RATE}, generate_kwargs=options)["text"]

            return cls(backend, run, None, "GPU" if gpu else "CPU")
        if backend == "indic-conformer":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(ASR_INFO[backend]["model"], trust_remote_code=True, token=token)

            def run(audio, lang, prompt):
                wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                return model(wav, lang or "hi", "rnnt")

            return cls(backend, run, None, "GPU" if gpu else "CPU")
        raise ValueError(f"unknown speech-to-text backend {backend}")

    def pick_language(self, audio, hint: str | None, allowed) -> str:
        """The language to decode in: the most likely of the allowed ones (when the model can tell), else hint."""
        if self.detect is None:
            return hint or allowed[0]
        try:
            probs = self.detect(audio)
        except Exception:  # noqa: BLE001 - a guess only
            return hint or allowed[0]
        best = max(allowed, key=lambda lang: (probs.get(lang, 0.0) + (0.15 if lang == hint else 0.0)))
        return best

    def hear(self, audio, hint: str | None = "hi", allowed=("hi", "mr", "en"), prompt: str | None = None,
             detect: bool = True):
        """(text, language decoded in, seconds) for 16 kHz samples."""
        started = time.perf_counter()
        lang = self.pick_language(audio, hint, list(allowed)) if detect else (hint or allowed[0])
        text = self.run(audio, lang, prompt)
        return " ".join(str(text or "").split()), lang, round(time.perf_counter() - started, 3)


class Mouth:
    """Text to speech. say() returns float32 samples, the sample rate and the seconds it took."""

    def __init__(self, backend: str, run, rate: int, device: str = "?"):
        self.backend, self.run, self.rate, self.device = backend, run, rate, device
        self.info = TTS_INFO[backend]
        self.cache: dict = {}

    @classmethod
    def load(cls, backend: str, token: str | None = None) -> "Mouth":
        import numpy as np
        import torch

        gpu = torch.cuda.is_available()
        device = "cuda:0" if gpu else "cpu"
        if backend == "kokoro":
            # kokoro-onnx: the same Kokoro-82M voices, run by onnxruntime (the "kokoro" package needs Python < 3.13,
            # Kaggle has 3.13). Pronunciation by espeak-ng: Hindi rules for Hindi and Marathi (same letters).
            from kokoro_onnx import Kokoro

            files = {}
            for name, url in KOKORO_FILES.items():
                files[name] = os.path.join(tempfile.gettempdir(), name)
                if not os.path.exists(files[name]):
                    download(url, files[name])
            if gpu:
                preload_cuda_libraries()
            import onnxruntime as ort

            if gpu and hasattr(ort, "preload_dlls"):  # onnxruntime 1.21+: load CUDA and cuDNN from the pip packages
                try:
                    ort.preload_dlls()
                except Exception as e:  # noqa: BLE001 - it then runs on the CPU, and says so
                    print("onnxruntime preload_dlls:", repr(e)[:200])

            on = f"CPU (onnxruntime; providers: {', '.join(ort.get_available_providers())})"
            if "CUDAExecutionProvider" in ort.get_available_providers():
                session = ort.InferenceSession(files["kokoro-v1.0.onnx"],
                                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                net = Kokoro.from_session(session, files["voices-v1.0.bin"])
                on = "GPU (onnxruntime)" if "CUDAExecutionProvider" in session.get_providers() else on
            else:
                net = Kokoro(files["kokoro-v1.0.onnx"], files["voices-v1.0.bin"])

            def run(text, lang, voice):
                voices, code = (KOKORO_EN, "en-us") if lang == "en" else (KOKORO_VOICES, "hi")
                audio, _ = net.create(text, voice=voices.get(voice, voices["bot"]), speed=1.0, lang=code)
                return np.asarray(audio, dtype=np.float32)

            run("नमस्ते", "hi", "bot")  # some problems only show on the first use
            return cls(backend, run, 24000, on)
        if backend == "parler":
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            name = TTS_INFO[backend]["model"]
            net = ParlerTTSForConditionalGeneration.from_pretrained(name, token=token).to(device)
            tok = AutoTokenizer.from_pretrained(name, token=token)
            describe = AutoTokenizer.from_pretrained(net.config.text_encoder._name_or_path)

            def run(text, lang, voice):
                speaker = PARLER_SPEAKERS.get(lang, PARLER_SPEAKERS["hi"]).get(voice, "Divya")
                style = PARLER_STYLE["bot" if voice == "bot" else "person"]
                d = describe(f"{speaker} {style}", return_tensors="pt").to(device)
                p = tok(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = net.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                       prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
                return out.cpu().float().numpy().squeeze()

            return cls(backend, run, net.config.sampling_rate, device)
        if backend == "mms":
            from transformers import AutoTokenizer, VitsModel

            models = {}

            def model(lang):
                code = {"mr": "mar", "en": "eng"}.get(lang, "hin")
                if code not in models:
                    name = f"facebook/mms-tts-{code}"
                    models[code] = (VitsModel.from_pretrained(name).to(device), AutoTokenizer.from_pretrained(name))
                return models[code]

            romanize = []

            def run(text, lang, voice):
                net, tok = model(lang)
                if getattr(tok, "is_uroman", False):  # this voice reads Latin letters only
                    if not romanize:
                        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uroman"], check=True,
                                       capture_output=True, timeout=600)
                        import uroman

                        romanize.append(uroman.Uroman().romanize_string)
                    text = romanize[0](text)
                inputs = tok(text, return_tensors="pt").to(device)
                torch.manual_seed({"bot": 0, "person": 1, "person2": 2, "person3": 3}.get(voice, 0))
                with torch.no_grad():
                    return net(**inputs).waveform.cpu().float().numpy().squeeze()

            return cls(backend, run, 16000, device)
        raise ValueError(f"unknown text-to-speech backend {backend}")

    def say(self, text: str, lang: str = "hi", voice: str = "bot"):
        """(samples, rate, seconds). The same line said again (a task's greeting and questions) comes from a
        cache, at once."""
        started = time.perf_counter()
        key = (text, lang, voice)
        audio = self.cache.get(key)
        if audio is None:
            audio = self.run(text, lang, voice)
            if audio is None or len(audio) == 0:
                raise RuntimeError(f"{self.backend} made no sound for {text[:40]!r}")
            if len(self.cache) >= CACHE_LINES:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = audio
        return audio, self.rate, round(time.perf_counter() - started, 3)


def load_first(kind: str, backends, token: str | None = None, log=print):
    """The first backend that loads ("ears" or "mouth"), and what happened to each tried: [{backend, ok, why}]."""
    tried = []
    cls, info = (Ears, ASR_INFO) if kind == "ears" else (Mouth, TTS_INFO)
    for backend in backends:
        if info.get(backend, {}).get("token") and not token:
            tried.append({"backend": backend, "ok": False, "why": "skipped: needs HF_TOKEN (Kaggle secret)"})
            continue
        started = time.time()
        try:
            loaded = cls.load(backend, token)
        except Exception as e:  # noqa: BLE001 - try the next one
            tried.append({"backend": backend, "ok": False, "why": f"{type(e).__name__}: {e}"[:400]})
            log(f"{kind}: {backend} did not load: {e!r}"[:500])
            continue
        tried.append({"backend": backend, "ok": True, "why": f"loaded in {time.time() - started:.0f} s",
                      "device": loaded.device})
        log(f"{kind}: {backend} ({info[backend]['model']}, {info[backend]['licence']}) on {loaded.device}")
        return loaded, tried
    return None, tried
