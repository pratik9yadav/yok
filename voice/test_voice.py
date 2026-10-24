"""Checks voice/voice_v0.ipynb (Voice v0: ears → brain → mouth), voice/voicecall.py and voice/run_voice.py, offline:
the ears (ASR), the mouth (TTS) and the brain (the shop model) are replaced by fakes, and Kaggle by a fake API.
Run:  python -m pytest"""

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import run_on_kaggle
from score import FIELDS, answer_text
from test_try import FakeApi, Status
from voice import build_voice, run_voice, voicecall
from voice.voicecall import (
    SYNTHETIC,
    find_clips,
    hindi_number,
    process_clip,
    question_of,
    summary_lines,
    tts_plan,
    write_reply,
    wrong_fields,
)

NOTHING = {"in_stock": None, "price_inr": None, "delivery": None, "alternative": None}


def cell(title: str, nb: dict | None = None) -> str:
    nb = nb or json.loads((HERE / build_voice.NOTEBOOK).read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(title))["source"])


def notebook_scope(tmp_path: Path, clips_dir: Path | None = None) -> dict:
    """The settings and helpers cells, run as the notebook would, saving into tmp_path."""
    scope: dict = {}
    exec(cell("#@title 1."), scope)  # noqa: S102 - our own generated cell
    scope.update(WORKING_DIR=str(tmp_path), CLIPS_DIR=str(clips_dir or tmp_path / "no-clips"))
    exec(cell("#@title 2."), scope)  # noqa: S102
    scope["OLLAMA_LOG"] = str(tmp_path / "ollama.log")
    return scope


# The fakes: speak writes the text into the "audio" file, transcribe reads it back, think answers from a table.
def fake_speak(text: str, path: str, voice: str = "ai") -> None:
    Path(path).write_text(text, encoding="utf-8")


def fake_transcribe(path: str, language: str | None) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return text if language == "hi" else "(auto) " + text


ANSWERS = {row["said"]: row["answer"] for row in SYNTHETIC}
ANSWERS[SYNTHETIC[2]["said"]] = {**SYNTHETIC[2]["answer"], "price_inr": 205}  # one wrong price, on purpose
ANSWERS[SYNTHETIC[1]["said"]] = {**SYNTHETIC[1]["answer"], "alternative": "हैवेल्स 32A वाला"}  # Devanagari brand
OWN = "दो सौ बीस का है"  # what the owner's made-up recording "says"
ANSWERS[OWN] = {"in_stock": None, "price_inr": 220, "delivery": None, "alternative": None}


def fake_think(product: str, transcript: str) -> str:
    assert product == "16A MCB"
    return " " + answer_text(ANSWERS.get(transcript, NOTHING)) + "\n"


# --- the notebook ---------------------------------------------------------------------------------------------------

def test_notebook_is_built_from_current_files() -> None:
    current = (HERE / build_voice.NOTEBOOK).read_text(encoding="utf-8") == build_voice.text(build_voice.notebook())
    assert current, "voice/voice_v0.ipynb is out of date: run python voice/build_voice.py"


def test_every_code_cell_compiles_and_the_settings_are_one_line_each() -> None:
    nb = json.loads((HERE / build_voice.NOTEBOOK).read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "cell", "exec")
    settings = cell("#@title 1.")
    for name in ("PRODUCT", "PRODUCT_SPOKEN", "ASR_MODEL", "TTS_CANDIDATES", "CLIPS_DIR", "MAX_CLIPS", "WORKING_DIR"):
        assert len(re.findall(rf"^{name} = .+$", settings, re.MULTILINE)) == 1, name
    scope: dict = {}
    exec(settings, scope)  # noqa: S102 - our own generated cell
    assert scope["PRODUCT"] == "16A MCB" and scope["ASR_MODEL"] == "openai/whisper-large-v3-turbo"
    assert scope["TTS_CANDIDATES"] == ["ai4bharat/indic-parler-tts", "facebook/mms-tts-hin"]
    assert build_voice.PARLER_PIP in cell("#@title 3.") and "==" in build_voice.PARLER_PIP  # a pinned version
    for title, needs in (("#@title 6.", "find the model"), ("#@title 7.", "ollama.com/install.sh"),
                         ("#@title 8.", "ollama\", \"create")):
        assert needs in cell(title) and "if BRAIN_ERROR is None:" in cell(title), title


def test_the_brain_cells_record_a_missing_model_and_the_rest_still_runs(tmp_path: Path, monkeypatch) -> None:
    scope = notebook_scope(tmp_path)
    if Path("/kaggle/input").exists():
        pytest.skip("on Kaggle: a model file may be attached")
    calls = []
    monkeypatch.setattr(scope["subprocess"], "run", lambda *a, **k: calls.append(a) or pytest.fail("must not run"))
    for title in ("#@title 6.", "#@title 7.", "#@title 8."):
        exec(cell(title), scope)  # noqa: S102 - Run all goes on: no cell raises
    assert scope["BRAIN_ERROR"].startswith("RuntimeError: No model file") and not calls
    saved = json.loads((tmp_path / "voice_results.json").read_text(encoding="utf-8"))
    assert any("find the model" in e for e in saved["errors"]) and any(e.startswith("brain:") for e in saved["errors"])
    with pytest.raises(RuntimeError, match="the shop model is not running"):
        scope["think"]("16A MCB", "haan hai")


def test_notebook_end_to_end_with_fake_ears_brain_and_mouth(tmp_path: Path, capsys) -> None:
    clips = tmp_path / "input" / "my-clips"
    clips.mkdir(parents=True)
    (clips / "q2_rate.m4a").write_text(OWN, encoding="utf-8")
    (clips / "notes.txt").write_text("not audio", encoding="utf-8")
    out = tmp_path / "voice"
    out.mkdir()
    scope = notebook_scope(out, clips_dir=tmp_path / "input")
    scope.update(transcribe=fake_transcribe, think=fake_think, speak=fake_speak)
    voicecall.use_tts(scope["RESULTS"], "facebook/mms-tts-hin")  # what cell 5 records when MMS is used
    for title in ("#@title 9.", "#@title 10.", "#@title 11.", "#@title 12."):
        exec(cell(title), scope)  # noqa: S102
    r = json.loads((out / "voice_results.json").read_text(encoding="utf-8"))
    assert [q["audio"] for q in r["questions"]] == [str(out / f"q{n}.wav") for n in (1, 2, 3)]
    assert "16A MCB" in r["questions"][0]["text"] and "सोलह ए एमसीबी" in r["questions"][0]["speech"]
    assert (out / "q2.wav").read_text(encoding="utf-8") == "सोलह ए एमसीबी का दाम कितना है?"
    assert len(r["clips"]) == 7 and r["clips_planned"] == 7
    own, *synthetic = r["clips"]
    assert own["file"] == str(clips / "q2_rate.m4a") and not own["synthetic"] and own["question"] == 2
    assert own["transcript"] == OWN and own["transcript_auto"] == "(auto) " + OWN and own["expected"] is None
    assert own["answer"]["price_inr"] == 220 and own["reply_text"] == "ठीक है, 16A MCB का दाम ₹220 है। धन्यवाद!"
    assert [c["synthetic"] for c in synthetic] == [True] * 6
    assert [c["wrong"] for c in synthetic] == [[], [], ["price_inr"], [], [], []]
    assert synthetic[5]["reply_text"].startswith("माफ़ कीजिए")  # "pata nahi, baad mein call karo": asks again
    for n, c in enumerate(r["clips"], 1):
        assert c["reply_audio"] == str(out / f"reply_{n}.wav") and (out / f"reply_{n}.wav").read_text(
            encoding="utf-8") == c["reply_speech"]
        assert set(c["seconds"]) == {"asr", "asr_auto", "brain", "tts", "total"} and c["seconds"]["total"] >= 0
        assert not c["errors"]
    assert set(r["average_seconds"]) == {"asr", "brain", "tts", "total"}
    assert r["tts_model"] == "facebook/mms-tts-hin" and r["tts_licence"] == "CC-BY-NC-4.0"
    assert r["tts_business_use"] is False and "not for business use" in r["tts_note"]
    assert r["asr_model"] == "openai/whisper-large-v3-turbo" and r["asr_licence"] == "MIT"
    text = "\n".join(r["summary"])
    for want in ("NOT FOR BUSINESS USE", "your clip q2_rate.m4a q2", "Synthetic answers: 5 of 6 with all 4 fields right",
                 "3/4 right (wrong: price_inr)", "Average seconds over 7 clips"):
        assert want in text, want
    assert len(r["summary"]) == 3 + 7 + 2  # models, one line per clip, synthetic score, averages
    assert "Listen: the Output tab" in capsys.readouterr().out
    assert r["errors"] == []


def test_notebook_keeps_going_when_a_stage_fails(tmp_path: Path) -> None:
    scope = notebook_scope(tmp_path)

    def deaf(path, language):
        if "synthetic_3" in path:
            raise TimeoutError("ASR stuck")
        return fake_transcribe(path, language)

    def mute(text, path, voice="ai"):
        if path.endswith("synthetic_2.wav"):
            raise RuntimeError("no sound")
        fake_speak(text, path, voice)

    scope.update(transcribe=deaf, think=fake_think, speak=mute)
    for title in ("#@title 9.", "#@title 10.", "#@title 11.", "#@title 12."):
        exec(cell(title), scope)  # noqa: S102
    r = json.loads((tmp_path / "voice_results.json").read_text(encoding="utf-8"))
    assert len(r["clips"]) == 5  # synthetic 2 could not be spoken
    stuck = next(c for c in r["clips"] if c["file"].endswith("synthetic_3.wav"))
    assert stuck["transcript"] is None and stuck["answer"] is None and stuck["wrong"] == list(FIELDS)
    assert stuck["reply_text"].startswith("माफ़ कीजिए") and stuck["reply_audio"]  # asks again, out loud
    assert stuck["errors"] == ["asr: TimeoutError('ASR stuck')", "asr_auto: TimeoutError('ASR stuck')"]
    assert any(e.startswith("synthetic answer 2:") for e in r["errors"])
    assert any("synthetic_3.wav) asr: TimeoutError" in e for e in r["errors"])
    assert sum(not c["errors"] for c in r["clips"]) == 4


# --- voicecall.py ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("n", "words"), [
    (0, "शून्य"), (7, "सात"), (20, "बीस"), (45, "पैंतालीस"), (99, "निन्यानवे"), (100, "एक सौ"), (220, "दो सौ बीस"),
    (250, "दो सौ पचास"), (1500, "एक हज़ार पाँच सौ"), (1675, "एक हज़ार छह सौ पचहत्तर"), (100000, "एक लाख"),
    (125040, "एक लाख पच्चीस हज़ार चालीस"), (12_000_000, "12000000"),
])
def test_hindi_number(n: int, words: str) -> None:
    assert hindi_number(n) == words


def test_reply_in_stock_with_price() -> None:
    reply = write_reply({"in_stock": True, "price_inr": 220, "delivery": False, "alternative": None}, "16A MCB",
                        "सोलह ए एमसीबी")
    assert reply["text"] == "ठीक है, 16A MCB ₹220 में है, डिलीवरी नहीं। धन्यवाद!"
    assert reply["speech"] == "ठीक है, सोलह ए एमसीबी दो सौ बीस रुपये में है, डिलीवरी नहीं। धन्यवाद!"
    delivered = write_reply({"in_stock": True, "price_inr": 99.5, "delivery": True, "alternative": None}, "Bulb")
    assert delivered["text"] == "ठीक है, Bulb ₹99.50 में है, डिलीवरी हो जाएगी। धन्यवाद!"
    assert "निन्यानवे रुपये पचास पैसे" in delivered["speech"]


def test_reply_not_in_stock_with_an_alternative() -> None:
    reply = write_reply({"in_stock": False, "price_inr": None, "delivery": None, "alternative": "Havells 32A"},
                        "16A MCB")
    assert reply["text"] == "ठीक है, 16A MCB अभी नहीं है। Havells 32A के बारे में बताया, हम ग्राहक से पूछ लेंगे। धन्यवाद!"
    assert "Havells बत्तीस A के बारे में" in reply["speech"]  # digits as words: some voices skip digits


def test_reply_asks_the_price_when_it_was_not_said() -> None:
    reply = write_reply({"in_stock": True, "price_inr": None, "delivery": None, "alternative": None}, "16A MCB")
    assert reply["text"] == "ठीक है, 16A MCB है। इसका दाम कितना है?"
    only_delivery = write_reply({"in_stock": None, "price_inr": None, "delivery": True, "alternative": None}, "X")
    assert only_delivery["text"] == "ठीक है, डिलीवरी हो जाएगी। धन्यवाद!"


@pytest.mark.parametrize("answer", [NOTHING, None, "not json", {"in_stock": None, "price_inr": "?", "delivery": None,
                                                                  "alternative": "  "}])
def test_reply_asks_again_politely_when_nothing_was_understood(answer) -> None:
    reply = write_reply(answer, "16A MCB")
    assert reply["text"] == reply["speech"]
    assert reply["text"].startswith("माफ़ कीजिए, ठीक से समझ नहीं आया।") and reply["text"].endswith("धन्यवाद!")


@pytest.mark.parametrize(("name", "question"), [
    ("q2_anything.m4a", 2), ("Q1.wav", 1), ("q3-delivery.ogg", 3), ("/x/y/q1 shop.mp3", 1), ("q12.wav", None),
    ("q4_extra.wav", None), ("answer_q2.m4a", None), ("rate.m4a", None),
])
def test_clip_name_says_which_question_it_answers(name: str, question: int | None) -> None:
    assert question_of(name) == question


def test_find_clips_takes_audio_files_at_any_depth(tmp_path: Path) -> None:
    for name in ("b/q3_x.ogg", "a/deep/q1_y.M4A", "q2.mp3", "notes.txt", "c/.hidden.wav", "d/voice.opus"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    clips = find_clips(str(tmp_path))
    assert [(Path(c["file"]).name, c["question"], c["synthetic"]) for c in clips] == [
        ("q1_y.M4A", 1, False), ("q2.mp3", 2, False), ("q3_x.ogg", 3, False), ("voice.opus", None, False)]
    assert len(find_clips(str(tmp_path), limit=2)) == 2
    assert find_clips(str(tmp_path / "missing")) == [] and find_clips("") == []


def test_synthetic_answers_are_varied_and_speakable() -> None:
    assert len(SYNTHETIC) == 6
    for row in SYNTHETIC:
        assert set(row["answer"]) == set(FIELDS)
        assert not re.search(r"[A-Za-z0-9]", row["said"]), row["said"]  # every TTS can say it
    answers = [row["answer"] for row in SYNTHETIC]
    assert {a["in_stock"] for a in answers} == {True, False, None}
    assert {a["delivery"] for a in answers} == {True, False, None}
    assert any(a["alternative"] for a in answers) and NOTHING in answers
    assert any("दो सौ बीस" in row["said"] for row in SYNTHETIC)


def test_wrong_fields_accepts_the_alternative_in_either_script() -> None:
    want = {"in_stock": False, "price_inr": None, "delivery": None, "alternative": "Havells 32A"}
    words = ["havells", "हैवेल्स"]
    assert wrong_fields({**want, "alternative": "हैवेल्स का 32 ए"}, want, words) == []
    assert wrong_fields({**want, "alternative": "Havells 32A"}, want) == []
    assert wrong_fields({**want, "alternative": "Anchor"}, want, words) == ["alternative"]
    assert wrong_fields({**want, "alternative": None, "in_stock": True}, want, words) == ["in_stock", "alternative"]
    assert wrong_fields(None, want) == list(FIELDS)


def test_process_clip_saves_after_every_stage(tmp_path: Path) -> None:
    results = voicecall.new_results("openai/whisper-large-v3-turbo", "16A MCB")
    saves = []
    clip = tmp_path / "q1_a.wav"
    fake_speak(SYNTHETIC[0]["said"], str(clip))

    def think(product, transcript):
        assert saves[-1]["clips"][0]["transcript"] == transcript  # the transcript was saved before the brain ran
        return fake_think(product, transcript)

    entry = process_clip(1, {"file": str(clip), "synthetic": True, "question": 1, "expected": SYNTHETIC[0]["answer"]},
                         "16A MCB", None, fake_transcribe, think, fake_speak, str(tmp_path), results,
                         lambda: saves.append(json.loads(json.dumps(results))))
    assert len(saves) == 5  # after asr, asr_auto, brain, tts, and at the end
    assert saves[-1]["clips"][0] == entry and entry["wrong"] == [] and entry["reply_audio"] == str(tmp_path / "reply_1.wav")
    assert entry["reply_text"] == "ठीक है, 16A MCB ₹220 में है। धन्यवाद!"


def test_process_clip_records_a_failing_brain_and_still_replies(tmp_path: Path) -> None:
    results = voicecall.new_results("openai/whisper-large-v3-turbo", "16A MCB")
    clip = tmp_path / "x.wav"
    fake_speak("हाँ है", str(clip))

    def broken(product, transcript):
        raise ConnectionError("Ollama down")

    entry = process_clip(3, {"file": str(clip)}, "16A MCB", None, fake_transcribe, broken, fake_speak, str(tmp_path),
                         results, lambda: None)
    assert entry["answer"] is None and entry["wrong"] is None and entry["reply_audio"]
    assert entry["errors"] == ["brain: ConnectionError('Ollama down')"]
    assert results["errors"] == ["clip 3 (x.wav) brain: ConnectionError('Ollama down')"]
    assert entry["seconds"]["brain"] is None and entry["seconds"]["total"] >= 0


def test_tts_plan_skips_the_model_that_needs_a_token() -> None:
    models = ["ai4bharat/indic-parler-tts", "facebook/mms-tts-hin"]
    plan, skipped = tts_plan(models, None)
    assert plan == ["facebook/mms-tts-hin"]
    assert skipped == [{"model": "ai4bharat/indic-parler-tts", "ok": False,
                        "why": "skipped: no HF_TOKEN (Kaggle Secrets or env)"}]
    assert tts_plan(models, "hf_x") == (models, [])


def test_licences_are_recorded_for_each_tts() -> None:
    results = voicecall.new_results("openai/whisper-large-v3-turbo", "16A MCB")
    voicecall.use_tts(results, "ai4bharat/indic-parler-tts")
    assert (results["tts_licence"], results["tts_business_use"]) == ("Apache-2.0", True)
    assert "NOT FOR BUSINESS USE" not in "\n".join(summary_lines(results))
    voicecall.use_tts(results, "facebook/mms-tts-hin")
    assert (results["tts_licence"], results["tts_business_use"]) == ("CC-BY-NC-4.0", False)
    assert "NOT FOR BUSINESS USE" in "\n".join(summary_lines(results))
    voicecall.use_tts(results, "someone/other-tts")
    assert results["tts_licence"].startswith("unknown") and results["tts_business_use"] is False


def test_hf_token_from_kaggle_secrets_else_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)  # not on Kaggle
    assert voicecall.hf_token() is None
    monkeypatch.setenv("HF_TOKEN", " hf_env ")
    assert voicecall.hf_token() == "hf_env"

    class Client:
        def get_secret(self, name):
            assert name == "HF_TOKEN"
            return "hf_kaggle"

    monkeypatch.setitem(sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=Client))
    assert voicecall.hf_token() == "hf_kaggle"


# --- run_voice.py ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("given", "want"), [
    ("me/wimb-voice-clips", "me/wimb-voice-clips"), (" me/clips.v2 ", "me/clips.v2"), ("", None), (None, None),
    ("https://www.kaggle.com/datasets/me/wimb-voice-clips", "me/wimb-voice-clips"),
])
def test_clips_dataset_name(given, want) -> None:
    assert run_voice.dataset_from(given) == want


@pytest.mark.parametrize("bad", ["wimb-voice-clips", "me/clips; rm -rf /", "me/a/b", "-x/y"])
def test_clips_dataset_name_rejects_others(bad: str) -> None:
    with pytest.raises(ValueError, match="not a Kaggle dataset name"):
        run_voice.dataset_from(bad)


@pytest.mark.parametrize("clips", [None, "me/wimb-voice-clips"])
def test_runner_attaches_the_training_output_and_the_clips(clips) -> None:
    api = FakeApi()
    run_voice.push(api, "me/wimb-voice-v0", clips)
    meta = api.calls["meta"]
    assert meta["kernel_sources"] == ["me/wimb-shop-model-training-v4"]
    assert meta["dataset_sources"] == ([clips] if clips else [])
    assert meta["id"] == "me/wimb-voice-v0" and meta["code_file"] == "voice_v0.ipynb"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"] and api.calls["acc"] == "NvidiaTeslaT4"
    assert api.calls["notebook"] == json.loads((HERE / "voice_v0.ipynb").read_text(encoding="utf-8"))


def test_runner_stops_when_kaggle_cannot_attach_the_clips() -> None:
    class NoDataset(FakeApi):
        def kernels_push(self, folder, acc=None):
            resp = super().kernels_push(folder, acc)
            resp.invalid_dataset_sources = ["me/missing"]
            return resp

    with pytest.raises(SystemExit, match="could not attach the dataset"):
        run_voice.push(NoDataset(), "me/wimb-voice-v0", "me/missing")


def results_of_a_run(tmp_path: Path) -> dict:
    """voice_results.json as the notebook makes it, from the end-to-end run with fakes."""
    out = tmp_path / "run"
    out.mkdir()
    scope = notebook_scope(out)
    scope.update(transcribe=fake_transcribe, think=fake_think, speak=fake_speak)
    voicecall.use_tts(scope["RESULTS"], "ai4bharat/indic-parler-tts")
    for title in ("#@title 9.", "#@title 10.", "#@title 11.", "#@title 12."):
        exec(cell(title), scope)  # noqa: S102
    return json.loads((out / "voice_results.json").read_text(encoding="utf-8"))


class VoiceApi(FakeApi):
    """The fake Kaggle, handing out voice/voice_results.json and some .wav files, as Kaggle's Output tab has them."""

    def kernels_output(self, kernel, path, file_pattern=None, force=False):
        self.calls["pattern"] = file_pattern
        log = [{"stream_name": "stdout", "time": 1, "data": "RuntimeError: No model file\n"}]
        (Path(path) / "wimb-voice-v0.log").write_text(json.dumps(log))
        if self.results is not None:
            (Path(path) / "voice").mkdir()
            (Path(path) / "voice" / "voice_results.json").write_text(json.dumps(self.results, ensure_ascii=False),
                                                                     encoding="utf-8")
            for name in ("q1.wav", "reply_1.wav"):
                (Path(path) / "voice" / name).write_bytes(b"RIFF")


def test_runner_downloads_the_audio_and_prints_the_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    results = results_of_a_run(tmp_path)
    monkeypatch.setattr(run_voice, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.wav").write_bytes(b"old")  # an older run's file: removed first
    api = VoiceApi(results=results)
    assert run_voice.fetch(api, "me/wimb-voice-v0") == results
    out = capsys.readouterr().out
    for text in ("TTS (mouth): ai4bharat/indic-parler-tts (Apache-2.0)", "Synthetic answers: 5 of 6",
                 "Questions spoken: 3 of 3", "2 .wav files (q1.wav, reply_1.wav)",
                 "https://www.kaggle.com/code/me/wimb-voice-v0/output", "kaggle-output-voice"):
        assert text in out, text
    assert "NOT FOR BUSINESS USE" not in out and "Kaggle's log" not in out
    assert not (tmp_path / "out" / "old.wav").exists()
    assert re.search(api.calls["pattern"], "voice/reply_3.wav") and re.search(api.calls["pattern"],
                                                                                 "voice/voice_results.json")
    assert run_voice.gaps(results) == []


def test_runner_says_what_is_missing() -> None:
    assert run_voice.gaps({"questions": [], "clips": []}) == [
        "no question was spoken (the text to speech did not work)", "no answer clip was processed"]
    clips = [{"transcript": "x", "answer": None, "reply_audio": "r.wav"}, {"transcript": None, "reply_audio": None}]
    assert run_voice.gaps({"questions": [{"audio": "q1.wav"}], "clips": clips}) == [
        "the shop model understood no clip", "1 of 2 clips got no spoken reply"]


def run_main(api: FakeApi, tmp_path: Path, monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(run_voice, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_voice, "connect", lambda slug: (api, f"me/{slug}"))
    monkeypatch.setattr(run_voice, "check_gpu", lambda api, need_hours: None)
    monkeypatch.setattr(run_on_kaggle.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["run_voice.py", *argv])
    with pytest.raises(SystemExit) as stop:
        run_voice.main()
    return stop.value.code


def test_main_runs_and_passes_when_every_step_worked(tmp_path: Path, monkeypatch, capsys) -> None:
    api = VoiceApi(results=results_of_a_run(tmp_path))
    assert run_main(api, tmp_path, monkeypatch, "--clips-dataset", "me/wimb-voice-clips") == 0
    assert api.calls["meta"]["dataset_sources"] == ["me/wimb-voice-clips"]
    assert "Not done" not in capsys.readouterr().out


def test_main_fails_when_the_run_left_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    assert run_main(VoiceApi(results=None, final=Status.ERROR), tmp_path, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "No voice_results.json" in out and "RuntimeError: No model file" in out and "Not done" in out


def test_main_rejects_a_bad_dataset_name(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["run_voice.py", "--clips-dataset", "no slash"])
    monkeypatch.setattr(run_voice, "connect", lambda slug: pytest.fail("must not reach Kaggle"))
    with pytest.raises(SystemExit) as stop:
        run_voice.main()
    assert stop.value.code == 2 and "not a Kaggle dataset name" in capsys.readouterr().err


# --- the workflow ---------------------------------------------------------------------------------------------------

def workflow_command(tmp_path: Path, **env: str) -> list[str]:
    """What the voice workflow's main step runs, with these inputs: the arguments it gives python."""
    yaml = pytest.importorskip("yaml")  # comes with kaggle (through jupytext)
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "voice.yml").read_text(encoding="utf-8"))
    (step,) = [s for s in workflow["jobs"]["voice"]["steps"] if "run_voice.py" in s.get("run", "")]
    assert set(env) <= set(step["env"])
    bin_dir = tmp_path / "bin"  # a stand-in `python` that writes down its arguments, one per line
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python").write_text(f'#!/bin/sh\nfor a in "$@"; do echo "$a"; done > {tmp_path}/args\n')
    (bin_dir / "python").chmod(0o755)
    run_env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "CLIPS_DATASET": "", **env}
    subprocess.run(["bash", "-e", "-c", step["run"]], env=run_env, capture_output=True, check=True)
    return (tmp_path / "args").read_text().splitlines()


def test_workflow_runs_voice_v0_with_inputs_only_through_the_environment(tmp_path: Path) -> None:
    text = (REPO / ".github" / "workflows" / "voice.yml").read_text(encoding="utf-8")
    for line in text.splitlines():  # inputs reach the scripts only as environment variables
        if "${{ inputs." in line:
            assert re.fullmatch(r" +[A-Z_]+: \$\{\{ inputs\.\w+ \}\}", line), line
    for want in ("name: Voice v0 (speech to speech)", "group: kaggle-voice", "timeout-minutes: 90",
                 "path: kaggle_output_voice/", "python -m pytest -q", "KAGGLE_API_TOKEN"):
        assert want in text, want
    assert workflow_command(tmp_path) == ["voice/run_voice.py"]
    assert workflow_command(tmp_path, CLIPS_DATASET="me/clips; echo hi") == [
        "voice/run_voice.py", "--clips-dataset", "me/clips; echo hi"]
    assert "kaggle_output_voice/" in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
