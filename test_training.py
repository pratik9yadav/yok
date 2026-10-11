"""Checks the data files and the scoring.  Run:  python -m pytest"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from make_data import WORDS  # noqa: E402
from score import FIELDS, SYSTEM, answer_text, parse, score  # noqa: E402


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / "data" / name).read_text(encoding="utf-8").splitlines()]


def test_data_files_are_well_formed() -> None:
    for name in ("train.jsonl", "test.jsonl"):
        for ex in load(name):
            assert set(ex) == {"product", "said", "answer"}
            assert set(ex["answer"]) == set(FIELDS)
            a = ex["answer"]
            assert a["in_stock"] in (True, False, None)
            assert a["delivery"] in (True, False, None)
            assert a["price_inr"] is None or isinstance(a["price_inr"], int)
            if a["in_stock"] is not True:
                assert a["price_inr"] is None


def test_train_price_is_actually_said() -> None:
    for ex in load("train.jsonl"):
        p = ex["answer"]["price_inr"]
        if p is not None:
            said_as = {str(p)} | {words[p] for words in WORDS.values() if p in words}
            assert any(s in ex["said"] for s in said_as), ex


def test_test_set_is_not_in_training_data() -> None:
    train = {ex["said"] for ex in load("train.jsonl")}
    assert not train & {ex["said"] for ex in load("test.jsonl")}


def test_perfect_replies_score_full_marks() -> None:
    test = load("test.jsonl")
    result = score([answer_text(ex["answer"]) for ex in test], test)
    assert result["all_correct"] == 1.0 and result["json_ok"] == 1.0 and not result["mistakes"]


def test_scoring_catches_mistakes_and_reads_fenced_json() -> None:
    ex = {"said": "x", "answer": {"in_stock": True, "price_inr": 60, "delivery": None, "alternative": None}}
    fenced = '```json\n{"in_stock": true, "price_inr": "60", "delivery": null, "alternative": null}\n```'
    wrong = '{"in_stock": true, "price_inr": 6, "delivery": false, "alternative": null}'
    result = score([fenced, wrong, "no json here"], [ex, ex, ex])
    assert result["json_ok"] == 2 / 3
    assert result["all_correct"] == 1 / 3
    assert result["mistakes"][0]["wrong"] == ["price_inr", "delivery"]
    assert parse("no json here") is None


def test_alternative_wording_can_differ() -> None:
    ex = {"said": "x", "answer": {"in_stock": False, "price_inr": None, "delivery": None, "alternative": "40 mm"}}
    reply = '{"in_stock": false, "price_inr": null, "delivery": null, "alternative": "40 mm padlock"}'
    assert score([reply], [ex])["all_correct"] == 1.0


def test_notebook_is_built_from_current_files() -> None:
    nb = json.loads((HERE / "train_colab.ipynb").read_text(encoding="utf-8"))
    data_cell = "".join(next(c for c in nb["cells"] if "TRAIN_DATA = " in "".join(c["source"]))["source"])
    scope: dict = {}
    exec(data_cell, scope)  # noqa: S102 - our own generated cell
    assert scope["TRAIN_DATA"] == load("train.jsonl")
    assert scope["TEST"] == load("test.jsonl")
    assert scope["SYSTEM"] == SYSTEM


class FakeModel:
    """Stands in for the trained model: writes the files Unsloth would write."""

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def save_pretrained_gguf(self, path: str, tokenizer, quantization_method: str) -> dict:
        out = Path(path + "_gguf")
        out.mkdir(parents=True)
        (out / "model.BF16.gguf").write_text("x" * 100)
        (out / "model.Q4_K_M.gguf").write_text("x" * 30)
        (out / "Modelfile").write_text('FROM model.BF16.gguf\nTEMPLATE """{{ .Prompt }}"""\nPARAMETER temperature 0.6\n')
        return {"gguf_files": [str(out / "model.BF16.gguf"), str(out / "model.Q4_K_M.gguf")],
                "modelfile_location": str(out / "Modelfile")}


def test_save_cell_keeps_the_small_model_and_a_working_modelfile(tmp_path: Path) -> None:
    import shutil

    nb = json.loads((HERE / "train_colab.ipynb").read_text(encoding="utf-8"))
    save = "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith("#@title 10."))["source"])
    scope = {"model": FakeModel(), "tokenizer": FakeModel(), "shutil": shutil, "os": __import__("os"),
             "SYSTEM": SYSTEM, "OUTPUT_NAME": "m", "MAKE_OLLAMA_FILE": True,
             "WORK": str(tmp_path / "work"), "SAVE_DIR": str(tmp_path / "out")}
    exec(save, scope)  # noqa: S102 - our own generated cell
    saved = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert saved == ["Modelfile", "lora", "m.Q4_K_M.gguf"]
    modelfile = (tmp_path / "out" / "Modelfile").read_text()
    assert modelfile.startswith("FROM ./m.Q4_K_M.gguf\n") and modelfile.count("FROM ") == 1
    assert "PARAMETER temperature 0\n" in modelfile and "0.6" not in modelfile
    assert f'SYSTEM """{SYSTEM}"""' in modelfile


def test_kaggle_runner_pushes_waits_and_fetches(tmp_path: Path, monkeypatch) -> None:
    import enum
    from types import SimpleNamespace

    import run_on_kaggle

    Status = enum.Enum("Status", "RUNNING COMPLETE")
    calls: dict = {}

    class FakeApi:
        def kernels_push(self, folder, acc=None):
            calls["meta"] = json.loads((Path(folder) / "kernel-metadata.json").read_text())
            calls["acc"] = acc
            calls["has_notebook"] = (Path(folder) / "train_colab.ipynb").exists()
            return SimpleNamespace(error=None, url="https://kaggle/x")

        def kernels_status(self, kernel):
            calls["polls"] = calls.get("polls", 0) + 1
            return SimpleNamespace(status=Status.COMPLETE if calls["polls"] > 1 else Status.RUNNING,
                                   failure_message="")

        def kernels_output(self, kernel, path, file_pattern=None, force=False):
            row = {f: 1.0 for f in ("all_correct", "json_ok") + FIELDS}
            (Path(path) / "results.json").write_text(json.dumps(
                {"before": row, "after": row, "mistakes_after": [], "mistakes_before": []}))

    monkeypatch.setattr(run_on_kaggle, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_on_kaggle.time, "sleep", lambda s: None)
    api = FakeApi()
    run_on_kaggle.push(api, "me/wimb-shop-model-training")
    assert run_on_kaggle.wait(api, "me/wimb-shop-model-training") is True
    run_on_kaggle.fetch(api, "me/wimb-shop-model-training")
    assert calls["acc"] == "NvidiaTeslaT4" and calls["has_notebook"]
    assert calls["meta"]["id"] == "me/wimb-shop-model-training" and calls["meta"]["enable_internet"]
    assert (tmp_path / "out" / "results.json").exists()
