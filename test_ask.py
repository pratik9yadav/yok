"""Checks ask_model.ipynb (run the model once, then ask it many questions) and try_on_kaggle.py --make-ask-notebook,
with a fake Ollama, a fake ipywidgets and a fake Kaggle.  Run:  python -m pytest"""

import io
import json
import os
import re
import socket
import subprocess
import sys
import types
import urllib.error
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import build_notebook
import try_on_kaggle
from score import user_message
from test_try import (
    OTHER,
    FakeApi,
    Status,
    fake_ollama,  # noqa: F401 - a pytest fixture, used by name below
    run_main,
    try_cell,
)

CHECKED = ("16A MCB", "haan hai, 220 ka, delivery nahi")  # what the self-check cell asks
IN_WORDS = "in stock: yes; price: ₹450; delivery: no; other product offered: none"  # OTHER, in words
SENT = {"stream": False, "keep_alive": "60m", "options": {"temperature": 0, "num_predict": 96}}  # with every question


@pytest.fixture(autouse=True)
def in_the_editor(monkeypatch) -> None:
    """Run the cells as in Kaggle's editor, where the owner asks (KAGGLE_KERNEL_RUN_TYPE is "Batch" in a saved run)."""
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")


def ask_cell(title: str) -> str:
    nb = json.loads((HERE / "ask_model.ipynb").read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(title))["source"])


def no_keep_alive(monkeypatch) -> None:
    """Unset OLLAMA_KEEP_ALIVE for the test, and remove what a cell sets afterwards (delenv alone records nothing
    when it is not set, so the cell's value would stay for every later test)."""
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "unset")
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE")


def dead_url() -> str:
    """An address nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


def ask_scope(tmp_path: Path, url: str | None = None) -> dict:
    """The hidden helpers cell, run as the notebook would (saving into tmp_path), plus what cells 2-4 would set."""
    scope: dict = {}
    exec(ask_cell("#@title 1."), scope)  # noqa: S102 - our own generated cell
    scope.update(WORKING_DIR=str(tmp_path), OLLAMA_LOG=str(tmp_path / "ollama.log"), OLLAMA_URL=url or dead_url(),
                 MODEL_FILE="/kaggle/input/t/wimb-shop-model/wimb-shop-model.Q4_K_M.gguf")
    scope["RESULTS"].update(ollama_version="0.0.0-fake", gpu="Tesla T4", model_file=scope["MODEL_FILE"],
                            model_mb=2497, create_seconds=44.0)
    return scope


def saved(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "ask_check.json").read_text(encoding="utf-8"))


def body(text: str) -> str:
    """A cell's code without its #@title line."""
    return text.split("\n", 1)[1]


def test_ask_notebook_reuses_the_try_cells_and_never_waits_for_input() -> None:
    nb = json.loads((HERE / "ask_model.ipynb").read_text(encoding="utf-8"))
    assert nb["metadata"] == json.loads((HERE / "try_model.ipynb").read_text(encoding="utf-8"))["metadata"]
    assert nb["metadata"]["accelerator"] == "GPU"
    assert body(ask_cell("#@title 2.")) == body(try_cell("#@title 3."))  # find the model
    assert body(ask_cell("#@title 4.")) == body(try_cell("#@title 5."))  # ollama create
    keep, start = body(ask_cell("#@title 3.")).split("\n", 1)
    assert keep.startswith('os.environ["OLLAMA_KEEP_ALIVE"] = "60m"')
    # Install and start Ollama: as in the try notebook, except that a failed install says to check Internet.
    assert build_notebook.NO_INTERNET in start and "check Internet is on" in build_notebook.NO_INTERNET
    assert start.strip("\n").replace(build_notebook.NO_INTERNET, build_notebook.INSTALL_FAILED) == body(
        try_cell("#@title 4."))
    options = "{" + build_notebook.OPTIONS  # both notebooks send the same options with every question
    assert options in body(ask_cell("#@title 1.")) and options in body(try_cell("#@title 6."))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "input(" not in code and "ask_check.json" in code
    intro = "".join(nb["cells"][0]["source"])
    for words in ("Ask the shop model", "Edit", "wimb-shop-model-training", "GPU T4", "Internet", "Run all", "Ask",
                  "Stop session", "Accelerator **None**", "not a chatbot", "each time you start it",
                  try_on_kaggle.ASK_NO_BOX):
        assert words in intro, words
    assert try_on_kaggle.ASK_NO_BOX in "".join(nb["cells"][-1]["source"])  # the tips
    # The helpers cell is folded on Kaggle and in Jupyter too, not only in Colab; the other cells are not.
    assert nb["cells"][1]["metadata"] == {"cellView": "form", "_kg_hide-input": True, "jupyter": {"source_hidden": True}}
    assert [c["metadata"] for c in nb["cells"][2:] if c["cell_type"] == "code"] == [{}] * 6


def test_start_cell_gives_ollama_serve_the_60_minute_keep_alive(tmp_path: Path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"  # a stand-in `ollama`: `serve` notes its OLLAMA_KEEP_ALIVE and stops
    bin_dir.mkdir()
    (bin_dir / "ollama").write_text(f'#!/bin/sh\n[ "$1" = "serve" ] && echo "$OLLAMA_KEEP_ALIVE" > {tmp_path}/env\n'
                                    'echo "ollama version is 0.0.0-fake"\n')
    (bin_dir / "ollama").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    no_keep_alive(monkeypatch)
    scope = ask_scope(tmp_path)
    cell = ask_cell("#@title 3.").replace('OLLAMA_URL = "http://127.0.0.1:11434"', f'OLLAMA_URL = "{dead_url()}"')
    with pytest.raises(RuntimeError, match="Ollama did not start"):
        exec(cell, scope)  # noqa: S102 - our own generated cell
    assert (tmp_path / "env").read_text().strip() == "60m"
    assert saved(tmp_path)["errors"][0].startswith("start Ollama")


def test_start_cell_reuses_a_running_ollama(tmp_path: Path, fake_ollama: dict, monkeypatch) -> None:  # noqa: F811
    no_keep_alive(monkeypatch)
    scope = ask_scope(tmp_path)
    default = 'OLLAMA_URL = "http://127.0.0.1:11434"'
    cell = ask_cell("#@title 3.").replace(default, f'OLLAMA_URL = "{fake_ollama["url"]}"')
    exec(cell, scope)  # noqa: S102
    check = saved(tmp_path)
    assert check["ollama_version"] == "0.0.0-fake" and check["errors"] == []


def test_self_check_loads_the_model_asks_and_saves(tmp_path: Path, fake_ollama: dict, capsys) -> None:  # noqa: F811
    scope = ask_scope(tmp_path, fake_ollama["url"])
    exec(ask_cell("#@title 5."), scope)  # noqa: S102
    check = saved(tmp_path)
    out = capsys.readouterr().out

    assert check["errors"] == []
    seconds = check["self_check"].pop("seconds")
    assert isinstance(seconds, float) and 0 <= seconds < 5
    assert check["self_check"] == {"product": CHECKED[0], "said": CHECKED[1], "reply": OTHER,
                                   "parsed": json.loads(OTHER)}
    assert check["processor"] == "100% GPU" and isinstance(check["load_seconds"], float)
    assert check["ollama_version"] == "0.0.0-fake" and check["gpu"] == "Tesla T4" and check["create_seconds"] == 44.0
    assert try_on_kaggle.answered(check)
    # First only load the model, then ask; both keep it loaded for 60 minutes. No system message: the Modelfile has it.
    assert fake_ollama["requests"] == [{"model": "wimb-shop", "keep_alive": "60m"},
                                       {"model": "wimb-shop", "prompt": user_message(*CHECKED), **SENT}]
    assert f"Answer: {IN_WORDS}" in out and f"Model's reply: {OTHER}" in out and "Took " in out
    assert "running on the GPU: about half a second per answer" in out and "Ready" in out
    assert "saved check" not in out


def test_self_check_in_kaggles_saved_run_says_to_tap_edit(tmp_path, fake_ollama, monkeypatch, capsys) -> None:  # noqa: F811
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")  # the check run, or Save Version: no one can type there
    scope = ask_scope(tmp_path, fake_ollama["url"])
    exec(ask_cell("#@title 5."), scope)  # noqa: S102
    out = capsys.readouterr().out
    assert try_on_kaggle.answered(saved(tmp_path)) and "Ready" not in out
    assert "This is the saved check, where nobody can type. To ask questions: tap Edit, then Run all." in out


def test_self_check_on_cpu_only_says_so_and_does_not_stop(tmp_path, fake_ollama, capsys) -> None:  # noqa: F811
    fake_ollama["no_gpu"] = True
    scope = ask_scope(tmp_path, fake_ollama["url"])
    scope["RESULTS"]["gpu"] = None  # Accelerator None: no GPU at all
    exec(ask_cell("#@title 5."), scope)  # noqa: S102 - must not raise
    check = saved(tmp_path)
    assert check["processor"] == "100% CPU" and check["errors"] == [] and try_on_kaggle.answered(check)
    assert "running on CPU only: about 10 s per answer" in capsys.readouterr().out

    scope["RESULTS"].update(gpu="Tesla T4", errors=[])  # a GPU that Ollama did not use: noted, still no stop
    exec(ask_cell("#@title 5."), scope)  # noqa: S102
    assert [e.split(":", 1)[0] for e in saved(tmp_path)["errors"]] == ["GPU"]


def test_when_ollama_is_not_running_it_says_run_all_again(tmp_path: Path, capsys) -> None:
    scope = ask_scope(tmp_path)  # nothing listens at OLLAMA_URL
    with pytest.raises(RuntimeError, match=r"^The model is not running \(the session may have restarted\): tap Run"):
        scope["ask"](*CHECKED)
    exec(ask_cell("#@title 5."), scope)  # noqa: S102 - records it, does not stop
    check = saved(tmp_path)
    assert check["self_check"] is None and not try_on_kaggle.answered(check)
    assert [e.split(":", 1)[0] for e in check["errors"]] == ["load the model", "self-check"]
    assert "tap Run all again" in check["errors"][1]
    out = capsys.readouterr().out
    assert ("The model is NOT running (see FAILED above): check the right panel (Input, Accelerator, Internet) and "
            "tap Run all again") in out
    assert "The model is running" not in out and "Ready" not in out
    exec(ask_cell("#@title 7."), scope)  # noqa: S102 - the "type here" cell prints it, no traceback
    assert "Could not ask: The model is not running" in capsys.readouterr().out


@pytest.mark.parametrize("title", ["#@title 6.", "#@title 7."])
def test_ask_cells_in_a_new_session_say_run_all_again(title, capsys) -> None:
    exec(ask_cell(title), {})  # noqa: S102 - a new session that runs only this cell: no NameError
    assert capsys.readouterr().out == "The session restarted: tap Run all again (about 3–4 minutes)\n"


def test_ask_passes_on_what_ollama_said_when_it_refuses(tmp_path: Path) -> None:
    def refuse(_body):
        raise urllib.error.HTTPError("http://x/api/generate", 404, "Not Found", {},
                                     io.BytesIO(b'{"error":"model \\"wimb-shop\\" not found"}'))

    scope = ask_scope(tmp_path)
    scope["generate"] = refuse
    with pytest.raises(RuntimeError, match='Ollama said: {"error":"model \\\\"wimb-shop\\\\" not found"}'):
        scope["ask"](*CHECKED)


def test_type_here_cell_asks_and_prints(tmp_path: Path, fake_ollama: dict, capsys) -> None:  # noqa: F811
    scope = ask_scope(tmp_path, fake_ollama["url"])
    exec(ask_cell("#@title 7."), scope)  # noqa: S102
    out = capsys.readouterr().out
    assert fake_ollama["requests"] == [{"model": "wimb-shop", "prompt": user_message(*CHECKED), **SENT}]
    assert out.splitlines()[:4] == [f"Product: {CHECKED[0]}", f"Shopkeeper: {CHECKED[1]}", f"Answer: {IN_WORDS}",
                                    f"Model's reply: {OTHER}"]
    assert re.fullmatch(r"Took \d+\.\d+ s", out.splitlines()[4])


def test_answer_lines(tmp_path: Path) -> None:
    lines = ask_scope(tmp_path)["answer_lines"]
    reply = '{"in_stock": false, "price_inr": null, "delivery": null, "alternative": "32A MCB"}'
    assert lines("16A MCB", "nahi hai, 32A hai", reply, 0.55) == [
        "Product: 16A MCB", "Shopkeeper: nahi hai, 32A hai",
        "Answer: in stock: no; price: not said; delivery: not said; other product offered: 32A MCB",
        f"Model's reply: {reply}", "Took 0.55 s"]
    assert lines("x", "y", OTHER, 9.8)[2] == f"Answer: {IN_WORDS}"
    assert lines("x", "y", "sorry, no idea", 1.0)[2] == "Answer: no answer (the reply was not the 4-field JSON)"


def fake_widgets() -> types.ModuleType:
    """Just enough of ipywidgets for the ask box. Like the real one, Button.click() runs the on_click handlers."""
    module = types.ModuleType("ipywidgets")
    module.__version__ = "0.0-fake"

    class Widget:
        def __init__(self, value="", **settings) -> None:
            self.value, self.disabled = value, False
            self.__dict__.update(settings)

    class Button(Widget):
        def __init__(self, **settings) -> None:
            super().__init__(**settings)
            self.handlers = []

        def on_click(self, handler) -> None:
            self.handlers.append(handler)

        def click(self) -> None:
            for handler in self.handlers:
                handler(self)

    class VBox:
        def __init__(self, children=(), **settings) -> None:
            self.children = list(children)

    class Layout:
        def __init__(self, **settings) -> None:
            self.__dict__.update(settings)

    module.Button, module.VBox, module.Layout = Button, VBox, Layout
    for name in ("Text", "Textarea", "Label", "HTML"):
        setattr(module, name, type(name, (Widget,), {}))
    return module


def parts(box) -> types.SimpleNamespace:
    """The ask box's product and said boxes, the button, the status line and the history."""
    by_type: dict = {}
    for child in box.children:
        by_type.setdefault(type(child).__name__, []).append(child)
    assert [len(by_type[n]) for n in ("Text", "Textarea", "Button", "HTML")] == [1, 1, 1, 1]
    return types.SimpleNamespace(product=by_type["Text"][0], said=by_type["Textarea"][0], button=by_type["Button"][0],
                                 status=by_type["Label"][-1], history=by_type["HTML"][0])


def shown(history) -> list[str]:
    """The answers in the history box, top first, as plain text."""
    import html

    return [html.unescape(t) for t in re.findall(r"<pre[^>]*>(.*?)</pre>", history.value, re.DOTALL)]


def test_ask_box_asks_and_shows_answers_and_errors_newest_first(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ipywidgets", fake_widgets())
    scope = ask_scope(tmp_path)
    asked = []

    def fake_ask(product, said):
        asked.append((product, said, box.button.disabled))
        if said == "boom":
            raise RuntimeError(scope["NOT_RUNNING"])
        if said == "silent":
            raise TimeoutError
        return OTHER, 0.42

    box = parts(scope["make_ask_box"](fake_ask))
    box.product.value, box.said.value = " 16A <MCB> ", " haan hai, 450 ka "
    box.button.click()
    assert asked == [("16A <MCB>", "haan hai, 450 ka", True)]  # the button is off while it asks
    assert not box.button.disabled and box.status.value == ""
    assert box.said.value == "" and box.product.value == " 16A <MCB> "  # ready for the next; the product stays
    assert shown(box.history) == ["\n".join(["Product: 16A <MCB>", "Shopkeeper: haan hai, 450 ka",
                                             f"Answer: {IN_WORDS}", f"Model's reply: {OTHER}", "Took 0.42 s"])]
    assert "<MCB>" not in box.history.value  # shown as text, not as HTML

    box.said.value = "boom"
    box.button.click()  # must not raise: ipywidgets would hide the error
    box.said.value = "silent"
    box.button.click()
    assert not box.button.disabled and box.said.value == "silent"  # kept, to try again
    history = shown(box.history)
    assert len(history) == 3 and history[2].startswith("Product: 16A <MCB>\nShopkeeper: haan hai, 450 ka\nAnswer:")
    assert history[1].endswith("Could not ask: The model is not running (the session may have restarted): "
                               "tap Run all again")
    assert history[0].endswith("Could not ask: TimeoutError()")

    box.said.value = "   "
    box.button.click()  # nothing to ask: a hint, no question, no new entry
    assert len(asked) == 3 and len(shown(box.history)) == 3 and "Type what the shopkeeper said" in box.status.value


def fake_display(monkeypatch) -> list:
    """IPython.display.display, recording what it was given."""
    displayed: list = []
    ipython, display = types.ModuleType("IPython"), types.ModuleType("IPython.display")
    display.display = displayed.append
    ipython.display = display
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    return displayed


def test_ask_box_cell_shows_the_box_and_it_asks_ollama(tmp_path, fake_ollama, monkeypatch, capsys) -> None:  # noqa: F811
    monkeypatch.setitem(sys.modules, "ipywidgets", fake_widgets())
    displayed = fake_display(monkeypatch)
    scope = ask_scope(tmp_path, fake_ollama["url"])
    exec(ask_cell("#@title 6."), scope)  # noqa: S102 - returns at once (no waiting for a person)
    assert len(displayed) == 1 and saved(tmp_path)["ask_box"] == "shown (ipywidgets 0.0-fake)"
    assert capsys.readouterr().out == try_on_kaggle.ASK_NO_BOX + "\n"  # if the editor can't draw it: cell 7
    box = parts(displayed[0])
    box.product.value, box.said.value = "Fevicol 1 kg", "haan hai, 450 ka"
    box.button.click()
    prompt = user_message("Fevicol 1 kg", "haan hai, 450 ka")
    assert fake_ollama["requests"] == [{"model": "wimb-shop", "prompt": prompt, **SENT}]
    assert f"Answer: {IN_WORDS}" in shown(box.history)[0]


def test_ask_box_cell_in_kaggles_saved_run_builds_the_box_but_does_not_show_it(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setitem(sys.modules, "ipywidgets", fake_widgets())
    displayed = fake_display(monkeypatch)
    scope = ask_scope(tmp_path)
    exec(ask_cell("#@title 6."), scope)  # noqa: S102
    check = saved(tmp_path)
    assert displayed == [] and check["errors"] == []
    assert check["ask_box"] == "built, not shown (Kaggle's saved run; ipywidgets 0.0-fake)"
    assert capsys.readouterr().out == scope["SAVED_RUN"] + "\n"


def test_ask_box_cell_without_ipywidgets_points_to_the_next_cell(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "ipywidgets", None)  # import ipywidgets now raises ImportError
    fake_display(monkeypatch)
    scope = ask_scope(tmp_path)
    exec(ask_cell("#@title 6."), scope)  # noqa: S102
    assert "use the next cell instead" in capsys.readouterr().out
    check = saved(tmp_path)
    assert check["ask_box"] == "no ipywidgets" and check["errors"] == []


def test_ask_box_cell_records_a_broken_box_and_goes_on(tmp_path: Path, monkeypatch, capsys) -> None:
    widgets = fake_widgets()
    del widgets.Textarea  # e.g. an ipywidgets version that differs
    monkeypatch.setitem(sys.modules, "ipywidgets", widgets)
    fake_display(monkeypatch)
    scope = ask_scope(tmp_path)
    exec(ask_cell("#@title 6."), scope)  # noqa: S102 - must not raise: Run all goes on to the next cell
    check = saved(tmp_path)
    assert check["ask_box"] == "failed" and check["errors"][0].startswith("ask box: AttributeError")
    assert "Use the next cell instead" in capsys.readouterr().out


# What ask_check.json holds after a good run (the numbers of the first try run).
CHECK = {"ollama_version": "0.40.3", "gpu": "Tesla T4", "model_file": "/kaggle/input/t/wimb-shop-model.Q4_K_M.gguf",
         "model_mb": 2497, "create_seconds": 44.0, "load_seconds": 46.0, "processor": "100% GPU",
         "self_check": {"product": CHECKED[0], "said": CHECKED[1],
                        "reply": '{"in_stock": true, "price_inr": 220, "delivery": false, "alternative": null}',
                        "parsed": {"in_stock": True, "price_inr": 220, "delivery": False, "alternative": None},
                        "seconds": 0.55},
         "ask_box": "built, not shown (Kaggle's saved run; ipywidgets 8.1.5)", "errors": []}


class FakeAskApi(FakeApi):
    """The fake Kaggle of test_try.py, handing out ask_check.json instead."""

    def kernels_output(self, kernel, path, file_pattern=None, force=False):
        self.calls["pattern"] = file_pattern
        log = [{"stream_name": "stderr", "time": 1, "data": "RuntimeError: Ollama did not start\n"}]
        (Path(path) / "wimb-shop-model-ask.log").write_text(json.dumps(log))
        if self.results is not None:
            (Path(path) / "ask_check.json").write_text(json.dumps(self.results, ensure_ascii=False), encoding="utf-8")


def make_ask(api: FakeApi, tmp_path: Path, monkeypatch, *argv: str) -> int:
    """try_on_kaggle.py --make-ask-notebook against the fake Kaggle: its exit code."""
    monkeypatch.setattr(try_on_kaggle, "ASK_OUT", tmp_path / "out_ask")
    return run_main(api, tmp_path, monkeypatch, "--make-ask-notebook", *argv)


def test_make_ask_notebook_pushes_it_checks_it_and_prints_the_link(tmp_path: Path, monkeypatch, capsys) -> None:
    api = FakeAskApi(CHECK)
    assert make_ask(api, tmp_path, monkeypatch) == 0
    meta = api.calls["meta"]
    assert meta["id"] == f"me/{try_on_kaggle.ASK_SLUG}" == "me/wimb-shop-model-ask"
    assert meta["code_file"] == "ask_model.ipynb" and api.calls["files"] == ["ask_model.ipynb", "kernel-metadata.json"]
    assert meta["kernel_sources"] == ["me/wimb-shop-model-training-v4"]
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"] and api.calls["acc"] == "NvidiaTeslaT4"
    assert api.calls["notebook"] == json.loads((HERE / "ask_model.ipynb").read_text(encoding="utf-8"))
    assert api.calls["pattern"] == r"(ask_check\.json|.*\.log)$"
    out = capsys.readouterr().out
    for text in ("Training notebook me/wimb-shop-model-training-v4: last run COMPLETE", "Ollama 0.40.3; GPU: Tesla T4",
                 "loading the model took 46 s", "Self-check: 16A MCB / haan hai, 220 ka, delivery nahi",
                 "Answer: in stock: yes; price: ₹220; delivery: no; other product offered: none",
                 "0.55 s on the GPU", "the model answered",
                 "Ask box: built, not shown (Kaggle's saved run; ipywidgets 8.1.5)", "this check can only build the box",
                 "Your notebook: https://www.kaggle.com/code/me/wimb-shop-model-ask", "1. Open it → Edit",
                 "Run all", "each time you start it", "Ask box, tap Ask", try_on_kaggle.ASK_NO_BOX, "Stop session"):
        assert text in out, text
    assert "Kaggle's log" not in out and "not ready" not in out


@pytest.mark.parametrize(("check", "final", "says"), [
    (None, Status.COMPLETE, "No ask_check.json: the run stopped early. Kaggle's log:"),
    (CHECK, Status.ERROR, "The run did not finish. Kaggle's log:"),
    ({**CHECK, "self_check": None, "errors": ["self-check: RuntimeError('The model is not running')"]}, Status.COMPLETE,
     "NO ANSWER from the model"),
    ({**CHECK, "self_check": {**CHECK["self_check"], "reply": "sorry", "parsed": None}}, Status.COMPLETE,
     "NO ANSWER from the model"),
    ({**CHECK, "ask_box": "failed", "errors": ["ask box: AttributeError()"]}, Status.COMPLETE, "Ask box: failed"),
])
def test_make_ask_notebook_fails_when_the_check_did_not_work(check, final, says, tmp_path, monkeypatch, capsys) -> None:
    out_dir = tmp_path / "out_ask"
    out_dir.mkdir()
    (out_dir / "ask_check.json").write_text(json.dumps(CHECK))  # an older run's good check must not count
    assert make_ask(FakeAskApi(check, final=final), tmp_path, monkeypatch) == 1
    out = capsys.readouterr().out
    assert says in out and "It is not ready yet" in out and "1. Open it → Edit" not in out
    if "log" in says:
        assert "RuntimeError: Ollama did not start" in out


def test_make_ask_notebook_stops_when_kaggle_cannot_attach_the_training_output(tmp_path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="could not attach"):
        try_on_kaggle.push_ask(FakeAskApi(invalid_sources=["me/wimb-shop-model-training-v4"]), "me/wimb-shop-model-ask")


@pytest.mark.parametrize("other", [("--ask", "16A MCB", "haan hai"), ("--questions-json", "[]"), ("--wait",)])
def test_make_ask_notebook_goes_alone(other, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["try_on_kaggle.py", "--make-ask-notebook", *other])
    monkeypatch.setattr(try_on_kaggle, "connect", lambda slug: pytest.fail("must not reach Kaggle"))
    with pytest.raises(SystemExit) as stop:
        try_on_kaggle.main()
    assert stop.value.code == 2 and "--make-ask-notebook can't be used with" in capsys.readouterr().err


def workflow_command(tmp_path: Path, **env: str) -> list[str]:
    """What the ask workflow's main step runs, with these inputs: the arguments it gives try_on_kaggle.py."""
    yaml = pytest.importorskip("yaml")  # comes with kaggle (through jupytext)
    workflow = yaml.safe_load((HERE / ".github" / "workflows" / "ask.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["ask"]["steps"]
    (step,) = [s for s in steps if "try_on_kaggle.py" in s.get("run", "")]
    assert set(env) <= set(step["env"])
    bin_dir = tmp_path / "bin"  # a stand-in `python` that writes down its arguments, one per line
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python").write_text(f'#!/bin/sh\nfor a in "$@"; do echo "$a"; done > {tmp_path}/args\n')
    (bin_dir / "python").chmod(0o755)
    run_env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "RUN_MODE": "", "PRODUCT": "", "SAID": "",
               "QUESTIONS_JSON": "", **env}
    subprocess.run(["bash", "-e", "-c", step["run"]], env=run_env, capture_output=True, check=True)
    return (tmp_path / "args").read_text().splitlines()


def test_workflow_makes_the_ask_notebook_or_asks_the_questions(tmp_path: Path) -> None:
    text = (HERE / ".github" / "workflows" / "ask.yml").read_text(encoding="utf-8")
    for line in text.splitlines():  # inputs reach the scripts only as environment variables
        if "${{ inputs." in line:
            assert re.fullmatch(r" +[A-Z_]+: \$\{\{ inputs\.\w+ \}\}", line), line
    assert "options: [questions, notebook]" in text and "kaggle_output_ask/" in text
    assert workflow_command(tmp_path, RUN_MODE="notebook") == ["try_on_kaggle.py", "--make-ask-notebook"]
    assert workflow_command(tmp_path, RUN_MODE="notebook", SAID="haan") == ["try_on_kaggle.py", "--make-ask-notebook"]
    assert workflow_command(tmp_path, RUN_MODE="questions", PRODUCT="16A MCB", SAID="haan hai, 220 ka") == [
        "try_on_kaggle.py", "--ask", "16A MCB", "haan hai, 220 ka"]
    assert workflow_command(tmp_path, RUN_MODE="questions", QUESTIONS_JSON='[{"said": "x"}]') == [
        "try_on_kaggle.py", "--questions-json", '[{"said": "x"}]']
    assert workflow_command(tmp_path, RUN_MODE="questions") == ["try_on_kaggle.py"]
