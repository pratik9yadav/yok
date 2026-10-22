"""Checks the data files and the scoring.  Run:  python -m pytest"""

import functools
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from score import FIELDS, KEYS, SYSTEM, answer_text, evaluate, make_modelfile, parse, results_payload, score, show


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / "data" / name).read_text(encoding="utf-8").splitlines()]


def assert_well_formed(ex: dict) -> None:
    assert set(ex) == {"product", "said", "answer"}
    assert set(ex["answer"]) == set(FIELDS)
    a = ex["answer"]
    assert a["in_stock"] in (True, False, None)
    assert a["delivery"] in (True, False, None)
    assert a["price_inr"] is None or isinstance(a["price_inr"], int)
    if a["in_stock"] is not True:
        assert a["price_inr"] is None


def test_data_files_are_well_formed() -> None:
    for name in ("train.jsonl", "train_v1.jsonl", "train_v4.jsonl", "test.jsonl", "hard_test.jsonl",
                 "fresh_test.jsonl"):
        for ex in load(name):
            assert_well_formed(ex)


def nfc_lower(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def price_words() -> dict[int, set[str]]:
    """Every way data/real/price_words.json says a price: Hindi and Marathi, Devanagari and Latin letters.

    Plus the hundreds style for 2100-4550 that make_data.py builds by the json's own rule (it has it only for
    1100-1999): N सौ / Nशे, then the rest, from the formal words for N and the rest
    (2550: पच्चीस सौ पचास, pachis sau pachas, पंचवीसशे पन्नास, panchvisshe pannas).
    """
    data = json.loads((HERE / "data" / "real" / "price_words.json").read_text(encoding="utf-8"))
    out: dict[int, set[str]] = {}
    formal: dict[tuple[int, str, str], set[str]] = {}  # (price, lang, "dev" or "roman") -> formal spellings
    for entry in data["prices"]:
        forms = out.setdefault(entry["price"], set())
        for lang in ("hi", "mr"):
            for kind in ("formal", "hundreds_style", "colloquial"):
                for f in entry[lang].get(kind, []):
                    forms.update(nfc_lower(t) for t in [f["text"], *f.get("spacing_variants", []), f["roman"],
                                                        *f.get("roman_variants", [])])
                    if kind == "formal":
                        formal.setdefault((entry["price"], lang, "dev"), set()).add(nfc_lower(f["text"]))
                        formal.setdefault((entry["price"], lang, "roman"), set()).update(
                            [f["roman"], *f.get("roman_variants", [])])
    joins = {("hi", "dev"): " सौ", ("hi", "roman"): " sau", ("mr", "dev"): "शे", ("mr", "roman"): "she"}
    for h in range(21, 46):
        for rest in (0, 50):
            for (lang, script), join in joins.items():
                heads = {w + join for w in formal[(h, lang, script)]}
                tails = formal[(rest, lang, script)] if rest else {""}
                out[h * 100 + rest].update((head + " " + tail).strip() for head in heads for tail in tails)
    return out


def test_train_price_is_actually_said() -> None:
    """The price is in the reply: as digits (also Devanagari digits, or 1,250) or as words from price_words.json.

    v4 also says prices with a fraction (सवा / साढ़े / पौने + N सौ / N हज़ार, off price_words.json's grid too);
    fraction_forms() below builds those by the rule, independently of make_data.py.
    """
    words = price_words()
    devanagari_digits = str.maketrans("0123456789", "०१२३४५६७८९")
    for name, extra in (("train.jsonl", False), ("train_v4.jsonl", True)):
        for ex in load(name):
            p = ex["answer"]["price_inr"]
            if p is not None:
                said_as = {str(p), str(p).translate(devanagari_digits), f"{p:,}"} | words.get(p, set())
                if extra:
                    said_as |= fraction_forms(p)
                assert any(s in nfc_lower(ex["said"]) for s in said_as), (name, ex)


# Fractions of a hundred or a thousand, as said in Hindi and Marathi (Devanagari and Latin letters).
FRACTIONS = {  # (lang, script): {quarters added: spellings}
    ("hi", "dev"): {1: ["सवा"], 2: ["साढ़े", "साढे"], -1: ["पौने"]},
    ("hi", "roman"): {1: ["sava", "sawa", "savaa"], 2: ["sadhe", "saadhe", "sade"], -1: ["paune", "pone", "pauna"]},
    ("mr", "dev"): {1: ["सव्वा"], 2: ["साडे"], -1: ["पावणे"]},
    ("mr", "roman"): {1: ["savva", "sawwa"], 2: ["sade", "saade"], -1: ["pavane", "pavne"]},
}
HUNDRED_WORD = {("hi", "dev"): [" सौ"], ("hi", "roman"): [" sau", " so"], ("mr", "dev"): ["शे"], ("mr", "roman"): ["she"]}
THOUSAND_WORD = {("hi", "dev"): ["हज़ार", "हजार"], ("hi", "roman"): ["hazar", "hajar", "hazaar"],
                 ("mr", "dev"): ["हजार"], ("mr", "roman"): ["hazar", "hajar"]}


@functools.cache
def number_words() -> dict[tuple[int, str, str], set[str]]:
    """(n, lang, "dev" or "roman") -> every spelling of n (1-99, and n x 100) in price_words.json."""
    data = json.loads((HERE / "data" / "real" / "price_words.json").read_text(encoding="utf-8"))
    out: dict[tuple[int, str, str], set[str]] = {}
    for entry in data["prices"]:
        for lang in ("hi", "mr"):
            for kind in ("formal", "hundreds_style"):
                for f in entry[lang].get(kind, []):
                    out.setdefault((entry["price"], lang, "dev"), set()).update(
                        nfc_lower(t) for t in [f["text"], *f.get("spacing_variants", [])])
                    out.setdefault((entry["price"], lang, "roman"), set()).update(
                        [f["roman"], *f.get("roman_variants", [])])
    return out


@functools.cache
def fraction_forms(p: int) -> frozenset[str]:
    """Every way to say p as सवा / साढ़े / पौने + N सौ (N = 1-45, not 10, 20, 30, 40) or + N हज़ार (N = 1-19)."""
    words, out = number_words(), set()
    for (lang, script), prefixes in FRACTIONS.items():
        joins = [" "] if lang == "hi" else ["", " "]
        for quarters, spellings in prefixes.items():
            n, rest = divmod(p - 25 * quarters, 100)
            if not rest and 1 <= n <= 45 and n % 10:
                if n == 1:
                    bases = ["सौ"] if (lang, script) == ("hi", "dev") else ["sau", "so"] if lang == "hi" else (
                        ["शे"] if script == "dev" else ["she"])
                    out |= {s + (" " if lang == "hi" else "") + b for s in spellings for b in bases}
                else:  # N x 100 as price_words.json says it, or N + सौ / शे
                    bases = words.get((n * 100, lang, script), set()) | {
                        w + h for w in words[(n, lang, script)] for h in HUNDRED_WORD[(lang, script)]}
                    out |= {s + j + b for s in spellings for j in joins for b in bases}
            n, rest = divmod(p - 250 * quarters, 1000)
            if not rest and 1 <= n <= 19:
                heads = [""] if n == 1 else sorted(words[(n, lang, script)])
                out |= {(s + j + h + " " + t).replace("  ", " ") for s in spellings for j in joins for h in heads
                        for t in THOUSAND_WORD[(lang, script)]}
    return frozenset(nfc_lower(f) for f in out)


def test_fraction_prices_by_hand() -> None:
    """Checked by hand: पौने सत्रह सौ = 1675, साडेचारशे = 450, पावणेतीनशे = 275, सवा पाँच हज़ार = 5250."""
    for p, said in ((1675, "पौने सत्रह सौ"), (450, "साडेचारशे"), (275, "पावणेतीनशे"), (5250, "सवा पाँच हज़ार"),
                    (125, "सवा सौ"), (125, "सव्वाशे"), (3750, "पौने चार हज़ार"), (1250, "साढ़े बारह सौ"),
                    (1250, "sava hazar"), (6500, "साडेसहा हजार"), (275, "paune teen sau"), (450, "sade charshe")):
        assert nfc_lower(said) in fraction_forms(p), (p, said)
    assert nfc_lower("पौने सत्रह सौ") not in fraction_forms(1700)
    assert nfc_lower("साडेचारशे") not in fraction_forms(350)


def test_test_sets_are_not_in_training_data_or_each_other() -> None:
    train, test, hard = (load(n) for n in ("train.jsonl", "test.jsonl", "hard_test.jsonl"))
    train_said, test_said, hard_said = ({ex["said"] for ex in rows} for rows in (train, test, hard))
    assert not train_said & test_said and not train_said & hard_said and not test_said & hard_said
    assert len(hard_said) == len(hard) == 56


def runs(text: str, n: int = 8) -> set[tuple[str, ...]]:
    """Every run of n words in a row (lower case, Devanagari kept)."""
    w = re.findall(r"[\w\u0900-\u097F]+", nfc_lower(text))
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def held_out_names() -> list[str]:
    """The held-out sets, never trained on; asr_test.jsonl (calls as speech-to-text writes them) when it is there."""
    return ["test.jsonl", "hard_test.jsonl", "fresh_test.jsonl"] + (
        ["asr_test.jsonl"] if (HERE / "data" / "asr_test.jsonl").exists() else [])


def test_train_shares_no_long_word_run_with_test_sets() -> None:
    """No run of 8 words in a row is the same in a training reply (v2 or v4) and a test reply, and no reply is
    the same. Only counts are reported: a failure must not print held-out text."""
    for train_name in ("train.jsonl", "train_v4.jsonl"):
        rows = load(train_name)
        train = set().union(*(runs(ex["said"]) for ex in rows))
        said = {nfc_lower(ex["said"]).strip() for ex in rows}
        for name in held_out_names():
            held = load(name)
            shared = sum(bool(runs(ex["said"]) & train) for ex in held)
            same = sum(nfc_lower(ex["said"]).strip() in said for ex in held)
            assert not shared and not same, f"{train_name} vs {name}: {shared} rows share 8 words, {same} identical"


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
    real = [{k: r[k] for k in ("product", "said", "answer")} for r in load("real/xpqa_hi_examples.jsonl")]
    assert scope["TRAIN_SET"] == "v2"  # the default stays v2 until v4 is measured
    assert scope["TRAIN_DATA"] == load("train.jsonl") + real
    assert scope["TRAIN_V1"] == load("train_v1.jsonl")
    # held-out sets: a failure says which one, without printing it
    assert scope["TEST"] == load("test.jsonl"), "TEST differs: run python build_notebook.py"
    assert scope["HARD"] == load("hard_test.jsonl"), "HARD differs: run python build_notebook.py"
    fresh = load("fresh_test.jsonl") if (HERE / "data" / "fresh_test.jsonl").exists() else []
    assert scope["FRESH"] == fresh, "FRESH differs: run python build_notebook.py"
    asr = load("asr_test.jsonl") if (HERE / "data" / "asr_test.jsonl").exists() else []
    assert scope["ASR"] == asr, "ASR differs from data/asr_test.jsonl: run python build_notebook.py"
    assert list(scope["SETS"]) == ["test", "hard"] + (["fresh"] if fresh else []) + (["asr"] if asr else [])
    assert all(scope["SETS"][k] is scope[k.upper()] for k in scope["SETS"])
    # v1 is the comparison run: only the first 400 made-up examples
    v1: dict = {"TRAIN_SET": "v1"}
    exec(data_cell, v1)  # noqa: S102
    assert v1["TRAIN_DATA"] == load("train_v1.jsonl")
    v4: dict = {"TRAIN_SET": "v4"}
    exec(data_cell, v4)  # noqa: S102
    assert v4["TRAIN_DATA"] == load("train_v4.jsonl") + real
    assert scope["SYSTEM"] == SYSTEM
    settings = "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith("#@title 3."))["source"])
    assert 'TRAIN_SET = "v2"  #@param ["v4", "v2", "v1"]' in settings and "EPOCHS = 2 " in settings


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
             "SYSTEM": SYSTEM, "make_modelfile": make_modelfile, "OUTPUT_NAME": "m", "MAKE_OLLAMA_FILE": True,
             "WORK": str(tmp_path / "work"), "SAVE_DIR": str(tmp_path / "out")}
    exec(save, scope)  # noqa: S102 - our own generated cell
    saved = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert saved == ["Modelfile", "lora", "m.Q4_K_M.gguf"]
    modelfile = (tmp_path / "out" / "Modelfile").read_text()
    assert modelfile == make_modelfile("m.Q4_K_M.gguf")  # not the template Unsloth wrote ("0.6" in the fake one)
    assert modelfile.startswith("FROM ./m.Q4_K_M.gguf\n") and modelfile.count("FROM ") == 1
    assert "PARAMETER temperature 0\n" in modelfile and "0.6" not in modelfile
    assert f'SYSTEM """{SYSTEM}"""' in modelfile


def test_modelfile_template_is_the_training_format() -> None:
    """Ollama fills the template in; the result must equal the text the model saw in training."""
    import re

    modelfile = make_modelfile("m.gguf")
    template = re.search(r'TEMPLATE """(.*?)"""', modelfile, re.DOTALL).group(1)
    system, prompt, response = "S", "P", '{"a": 1}'
    # Ollama's Go template, for these three simple blocks: {{ if X }}..{{ end }} and {{ .X }}
    filled = re.sub(r"\{\{ if \.(System|Prompt) \}\}(.*?)\{\{ end \}\}", lambda m: m.group(2), template, flags=re.DOTALL)
    filled = filled.replace("{{ .System }}", system).replace("{{ .Prompt }}", prompt).replace("{{ .Response }}", response)
    assert filled == f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
    assert (HERE / "Modelfile").read_text(encoding="utf-8") == make_modelfile("wimb-shop-model.Q4_K_M.gguf")


def test_kaggle_runner_pushes_waits_and_fetches(tmp_path: Path, monkeypatch, capsys) -> None:
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
            row = {f: 1.0 for f in KEYS}
            (Path(path) / "results.json").write_text(json.dumps(
                {"sets": {"test": {"examples": 30, "before": row, "after": row, "mistakes_after": []}},
                 "train_set": "v2", "train_rows": 1451}))
            (Path(path) / "results_gguf.json").write_text(json.dumps(
                {"gguf": "m.gguf", "seconds": 5, "sets": {"test": {"examples": 30, "of": 30, "all_correct": 1.0}}}))

        def kernels_list_files(self, kernel, page_size=20):
            return SimpleNamespace(files=[SimpleNamespace(name="wimb-shop-model/m.Q4_K_M.gguf", size=2_500_000_000)])

    monkeypatch.setattr(run_on_kaggle, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_on_kaggle.time, "sleep", lambda s: None)
    api = FakeApi()
    run_on_kaggle.push(api, "me/wimb-shop-model-training")
    assert run_on_kaggle.wait(api, "me/wimb-shop-model-training") is True
    run_on_kaggle.fetch(api, "me/wimb-shop-model-training")
    out = capsys.readouterr().out
    assert "test set: 30 calls" in out and "m.Q4_K_M.gguf  (size 2,500,000,000)" in out and "GGUF file m.gguf" in out
    assert "Training set: v2 (1451 rows)" in out
    assert calls["acc"] == "NvidiaTeslaT4" and calls["has_notebook"]
    assert calls["meta"]["id"] == "me/wimb-shop-model-training" and calls["meta"]["enable_internet"]
    assert (tmp_path / "out" / "results.json").exists()


NULL = {"in_stock": None, "price_inr": None, "delivery": None, "alternative": None}


def one(reply: str, **want) -> dict:
    return score([reply], [{"said": "x", "answer": {**NULL, **want}}])


def reply(**kw) -> str:
    return json.dumps({**NULL, **kw})


def test_garbage_is_never_correct_even_on_all_null_rows() -> None:
    for garbage in ("", "{}", "no json", '{"in_stock": true}', reply(note="x")[:-1] + ', "note": "x"}'):
        res = one(garbage)
        assert res["all_correct"] == 0 and res["json_ok"] == 0 and res["mistakes"][0]["wrong"] == list(FIELDS)


def test_booleans_are_not_numbers() -> None:
    res = one('{"in_stock": 1, "price_inr": true, "delivery": 0, "alternative": null}',
              in_stock=True, price_inr=1, delivery=False)
    assert res["mistakes"][0]["wrong"] == ["in_stock", "price_inr", "delivery"]


def test_alternative_must_really_match() -> None:
    for bad in ('""', '"m"', '"4"', "40", '["40 mm"]', '"padlock"'):
        assert one(reply(in_stock=False, alternative=json.loads(bad)), in_stock=False, alternative="40 mm")[
            "all_correct"] == 0, bad
    for good in ("40mm", "40 mm padlock", "40 MM", "Padlock 40 mm"):
        assert one(reply(in_stock=False, alternative=good), in_stock=False, alternative="40 mm")["all_correct"] == 1


def test_parse_finds_the_answer_among_other_text_and_braces() -> None:
    assert parse(reply(alternative="M-seal {2 pack}"))["alternative"] == "M-seal {2 pack}"
    assert parse("<think>shape {a,b}</think>" + reply(price_inr=5))["price_inr"] == 5
    assert parse(reply(in_stock=False) + " " + reply(in_stock=True))["in_stock"] is True  # the last one counts
    assert parse(reply() [:-1] + ', "note": "x"}') is None  # extra key


def test_evaluate_show_and_results_payload(capsys) -> None:
    rows = load("test.jsonl")
    sets = {"test": rows, "hard": load("hard_test.jsonl")}
    said_to_answer = {ex["said"]: answer_text(ex["answer"]) for rows_ in sets.values() for ex in rows_}
    before = evaluate(lambda product, said: "{}", sets)
    after = evaluate(lambda product, said: said_to_answer[said], sets)
    assert before["test"]["all_correct"] == 0 and after["hard"]["all_correct"] == 1.0
    assert len(after["hard"]["replies"]) == 56 and after["test"]["of"] == 30
    payload = results_payload("m", 2, before, after)
    json.dumps(payload)  # must be JSON-safe
    assert payload["sets"]["hard"]["examples"] == 56 and payload["sets"]["hard"]["after"]["all_correct"] == 1.0
    assert len(payload["sets"]["test"]["replies_after"]) == 30
    show("BEFORE", before)
    assert "BEFORE [test]: all 4 fields right in 0% of 30 calls" in capsys.readouterr().out
    # a deadline in the past answers nothing, and that is reported rather than crashing
    late = evaluate(lambda product, said: "{}", sets, deadline=0)
    assert late["test"]["examples"] == 0 and late["test"]["of"] == 30
    show("LATE", late)
    assert "no calls answered" in capsys.readouterr().out


def notebook_cell(title_start: str) -> str:
    nb = json.loads((HERE / "train_colab.ipynb").read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(title_start))["source"])


def data_scope() -> dict:
    scope: dict = {}
    exec(notebook_cell("#@title 4."), scope)  # noqa: S102 - our own generated cell
    return scope


def test_before_and_after_cells_run_with_a_stand_in_model(tmp_path: Path) -> None:
    import contextlib
    from types import SimpleNamespace

    scope = data_scope()
    answers = {ex["said"]: answer_text(ex["answer"]) for rows in scope["SETS"].values() for ex in rows}
    seen = {"said": None, "modes": []}

    class Encoded(dict):
        def to(self, device):
            return self

    class Tokenizer:
        reply = staticmethod(lambda said: "{}")

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["role"] == "system" and messages[1]["content"].startswith("Product: ")
            seen["said"] = messages[1]["content"].split("Shopkeeper: ", 1)[1]
            return Encoded(input_ids=SimpleNamespace(shape=(1, 7)))

        def decode(self, ids, skip_special_tokens=True):
            return "  " + self.reply(seen["said"]) + "\n"

    class Model:
        def generate(self, **inputs):
            assert inputs["max_new_tokens"] == 96 and inputs["do_sample"] is False
            return [[0] * 7 + [1, 2]]

    class FastLanguageModel:
        for_inference = staticmethod(lambda m: seen["modes"].append("inference"))
        for_training = staticmethod(lambda m: seen["modes"].append("training"))

    tokenizer = Tokenizer()
    scope.update(tokenizer=tokenizer, model=Model(), FastLanguageModel=FastLanguageModel,
                 torch=SimpleNamespace(no_grad=contextlib.nullcontext), ON_KAGGLE=True,
                 MODEL_NAME="m", EPOCHS=2)
    exec(notebook_cell("#@title 6."), scope)  # noqa: S102
    assert scope["before"]["test"]["all_correct"] == 0 and scope["before"]["hard"]["of"] == 56
    assert seen["modes"] == ["inference", "training"]

    tokenizer.reply = staticmethod(lambda said: answers[said])
    after_cell = notebook_cell("#@title 8.").replace("/kaggle/working/", str(tmp_path) + "/")
    exec(after_cell, scope)  # noqa: S102
    written = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert written["sets"]["test"]["before"]["all_correct"] == 0
    assert written["sets"]["test"]["after"]["all_correct"] == 1.0 and written["sets"]["hard"]["examples"] == 56
    assert len(written["sets"]["hard"]["replies_after"]) == 56


FAKE_LLAMA_SERVER = """#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
answers = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answers.json"), encoding="utf-8"))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.respond({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["temperature"] == 0 and body["messages"][0]["role"] == "system"
        said = body["messages"][1]["content"].split("Shopkeeper: ", 1)[1]
        self.respond({"choices": [{"message": {"content": " " + answers[said] + " "}}]})

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def gguf_cell_scope(tmp_path: Path, binary: str) -> tuple[dict, str]:
    import os
    import subprocess
    import time
    import urllib.request

    scope = data_scope()
    answers = {ex["said"]: answer_text(ex["answer"]) for rows in scope["SETS"].values() for ex in rows}
    (tmp_path / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
    server = tmp_path / "llama-server"
    server.write_text(FAKE_LLAMA_SERVER)
    server.chmod(0o755)
    cell = notebook_cell("#@title 11.").replace("/kaggle/working/", str(tmp_path) + "/")
    find = "find / -xdev -name llama-server -type f -perm -u+x 2>/dev/null | head -1"
    assert find in cell
    cell = cell.replace(find, f"echo {binary or server}" if binary != "none" else "true")
    scope.update(os=os, subprocess=subprocess, time=time, urllib=urllib, ON_KAGGLE=True,
                 MAKE_OLLAMA_FILE=True, gguf=str(tmp_path / "m.Q4_K_M.gguf"))
    return scope, cell


def test_gguf_cell_asks_the_server_and_scores_the_answers(tmp_path: Path) -> None:
    scope, cell = gguf_cell_scope(tmp_path, "")
    exec(cell, scope)  # noqa: S102
    report = json.loads((tmp_path / "results_gguf.json").read_text(encoding="utf-8"))
    assert "error" not in report and report["gguf"] == "m.Q4_K_M.gguf"
    assert report["sets"]["test"]["all_correct"] == 1.0 and report["sets"]["hard"]["examples"] == 56


def test_gguf_cell_never_fails_the_run_when_there_is_no_server(tmp_path: Path) -> None:
    scope, cell = gguf_cell_scope(tmp_path, "none")
    exec(cell, scope)  # noqa: S102 - must not raise
    assert "llama-server was not found" in json.loads((tmp_path / "results_gguf.json").read_text())["error"]


def test_runner_sets_the_training_set_and_export_choice() -> None:
    import run_on_kaggle

    assert run_on_kaggle.slug_for("v2") == "wimb-shop-model-training"
    assert run_on_kaggle.slug_for("v1") == "wimb-shop-model-training-v1"
    # v4 gets its own notebook: v2's output (the model the try, ask and voice notebooks attach) is not overwritten
    assert run_on_kaggle.slug_for("v4") == "wimb-shop-model-training-v4"
    assert run_on_kaggle.slug_for("v4", "my-test") == "my-test"
    assert run_on_kaggle.TRAIN_SETS == ["v2", "v1", "v4"] and run_on_kaggle.WAIT_MIN["v4"] > 120
    v4 = run_on_kaggle.notebook_for("v4", gguf=True)
    settings = "".join(next(c for c in v4["cells"] if "".join(c["source"]).startswith("#@title 3."))["source"])
    assert 'TRAIN_SET = "v4"' in settings and "MAKE_OLLAMA_FILE = True" in settings
    nb = run_on_kaggle.notebook_for("v1", gguf=False)
    settings = "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith("#@title 3."))["source"])
    assert 'TRAIN_SET = "v1"' in settings and "MAKE_OLLAMA_FILE = False" in settings
    assert 'TRAIN_SET = "v2"' not in settings and "MAKE_OLLAMA_FILE = True" not in settings
    default = run_on_kaggle.notebook_for("v2", gguf=True)
    settings = "".join(next(c for c in default["cells"] if "".join(c["source"]).startswith("#@title 3."))["source"])
    assert 'TRAIN_SET = "v2"' in settings and "MAKE_OLLAMA_FILE = True" in settings


def test_fresh_test_set_is_held_out_from_everything() -> None:
    fresh = {ex["said"] for ex in load("fresh_test.jsonl")}
    assert len(fresh) == len(load("fresh_test.jsonl")) == 60
    others = set()
    for name in ("train.jsonl", "train_v1.jsonl", "train_v4.jsonl", "test.jsonl", "hard_test.jsonl",
                 "real/xpqa_hi_examples.jsonl"):
        others |= {ex["said"] for ex in load(name)}
    assert not fresh & others, f"{len(fresh & others)} fresh rows are in other data"


# ---------------------------------------------------------------------------------------------
# Training set v4 (make_data.make_v4 -> data/train_v4.jsonl)

V2_MD5 = "a5b7354a18cd105d338426ac49fe400e"  # data/train.jsonl (v2): must not change


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # a checksum, not security


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


@functools.cache
def made_v4() -> tuple[list[dict], list[dict]]:
    import make_data

    return make_data.make_v4()


def test_v2_training_set_is_unchanged() -> None:
    import make_data

    assert md5((HERE / "data" / "train.jsonl").read_text(encoding="utf-8")) == V2_MD5
    assert jsonl(make_data.make()[0]) == (HERE / "data" / "train.jsonl").read_text(encoding="utf-8")


def test_v4_is_reproducible_and_written() -> None:
    import make_data

    rows, _ = made_v4()
    again, _ = make_data.make_v4()
    written = (HERE / "data" / "train_v4.jsonl").read_text(encoding="utf-8")
    assert md5(jsonl(rows)) == md5(jsonl(again)) == md5(written)
    assert len(rows) == 6000 and len({r["said"] for r in rows}) > 5990


def test_v4_rows_are_well_formed() -> None:
    for ex in load("train_v4.jsonl"):
        assert_well_formed(ex)
        assert ex["said"].strip() and ex["product"].strip()
        assert not re.search(r"[{}]|_ला", ex["said"]), ex  # no template slot left


def test_v4_proportions() -> None:
    """The owner's mix (within 5 points), noise on about 30% of the Devanagari rows only, about 8% all-null."""
    import make_data

    rows, meta = made_v4()
    n = len(rows)
    want = {"Hindi, Devanagari": 25, "Hinglish, Devanagari": 25, "Hinglish, Latin": 20, "Marathi, Devanagari": 20,
            "Marathi, Latin": 10}
    assert {k: round(v[0] * 100) for k, v in make_data.MIX_V4.items()} == want
    for cls, share in want.items():
        got = 100 * sum(m["mix"] == cls for m in meta) / n
        assert abs(got - share) <= 5, (cls, got)
    # the text agrees: Latin-letter rows have no Devanagari, Devanagari rows have it
    for r, m in zip(rows, meta, strict=True):
        assert bool(re.search(r"[ऀ-ॿ]", r["said"])) == m["mix"].endswith("Devanagari"), r
    marathi = {"mr_dev", "mx_dev", "mr_rom", "mren"}
    assert all((m["lang"] in marathi) == m["mix"].startswith("Marathi") for m in meta)
    assert {m["lang"] for m in meta if m["mix"] == "Hinglish, Devanagari"} == {"hx_dev"}
    assert {m["lang"] for m in meta if m["mix"] == "Hindi, Devanagari"} == {"hi_dev"}
    dev = [m for m in meta if m["mix"].endswith("Devanagari")]
    assert abs(100 * sum(m["noisy"] for m in dev) / len(dev) - 30) <= 5
    assert all(m["noise"] for m in dev if m["noisy"]) and not any(m["noise"] for m in meta if not m["noisy"])
    nulls = 100 * sum(all(v is None for v in r["answer"].values()) for r in rows) / n
    assert abs(nulls - 8) <= 5, nulls
    kinds = {m["kind"] for m in meta}
    assert {"spoken_fraction", "discount", "unit_vs_bulk", "price_correction", "delivery_said", "wrong_number",
            "unclear"} <= kinds
    priced = [m for m in meta if m["style"] is not None]
    assert sum("सवा/साढ़े/पौने" in m["style"] for m in priced) >= 0.15 * len(priced)
    assert sum(r["answer"]["delivery"] is None for r in rows) > n / 2  # most calls do not say


def test_v4_alternative_is_said_as_labelled() -> None:
    """The alternative is written exactly as the shopkeeper said it (brand + size/type + item), in that script."""
    alts = [ex for ex in load("train_v4.jsonl") if ex["answer"]["alternative"] is not None]
    assert len(alts) > 500
    for ex in alts:
        assert nfc_lower(ex["answer"]["alternative"]) in nfc_lower(ex["said"]), ex
        assert ex["answer"]["in_stock"] is False and ex["answer"]["price_inr"] is None
    full = sum(len(ex["answer"]["alternative"].split()) >= 2 for ex in alts)
    assert full >= 0.85 * len(alts)


def test_v4_noise_keeps_the_facts() -> None:
    """In noisy rows the price (test above), the alternative and the "not in stock" words are still there."""
    rows, meta = made_v4()
    noisy = [(r, m) for r, m in zip(rows, meta, strict=True) if m["noisy"]]
    assert len(noisy) > 1000
    no = r"नही|नाही|नाय|संप|ख़?त्म|खतम|स्टॉक|stock|अवेलेबल|available|फिनिश|शिल्लक|नसतं|मिळत"
    for r, m in noisy:
        a, said = r["answer"], nfc_lower(r["said"])
        if a["alternative"]:
            assert nfc_lower(a["alternative"]) in said
        if m["kind"] == "no_stock":
            assert re.search(nfc_lower(no), said), r


def test_train_workflow_offers_v4() -> None:
    text = (HERE / ".github" / "workflows" / "train.yml").read_text(encoding="utf-8")
    assert "options: [v2, v1, v4]" in text and "default: v2" in text
    assert "inputs.train_set == 'v4' && 330 || 150" in text
