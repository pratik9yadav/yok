"""Checks the free web page (space/app.py) and how it is put on Hugging Face (space/deploy.py), offline: gradio,
llama_cpp and huggingface_hub are not needed (fakes stand in for the model, Hugging Face and Kaggle).
Run:  python -m pytest"""

import json
import os
import re
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import run_on_kaggle
import try_on_kaggle
from score import SYSTEM, make_modelfile, user_message
from space import app, deploy
from test_training import held_out_names, load, nfc_lower, runs

TOKEN = "hf_SECRETsecret1234567890"  # a made-up token: it must never be printed
REPLY = '{"in_stock": true, "price_inr": 2850, "delivery": true, "alternative": null}'
NO_MODULES = "import sys\nfor name in ('gradio', 'llama_cpp', 'huggingface_hub'):\n    sys.modules[name] = None\n"


# --- the prompt and the answer ----------------------------------------------------------------------------------------

def test_prompt_is_the_training_text() -> None:
    """The text the Modelfile's TEMPLATE makes (the training text), up to where the answer starts."""
    template = re.search(r'TEMPLATE """(.*?)"""', make_modelfile("x.gguf"), re.DOTALL).group(1)
    for tag in ("{{ if .System }}", "{{ if .Prompt }}", "{{ end }}"):
        template = template.replace(tag, "")
    product, said = "Asian Paints 10 L", "हाँ है, 2850 का\nghar bhej denge"
    want = (template.split("{{ .Response }}")[0].replace("{{ .System }}", SYSTEM)
            .replace("{{ .Prompt }}", user_message(product, said)))
    assert app.prompt(product, said) == want
    assert want == (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nProduct: {product}\n"
                    f"Shopkeeper: {said}<|im_end|>\n<|im_start|>assistant\n")
    assert (app.N_CTX, app.MAX_TOKENS, app.STOP) == (2048, 96, ["<|im_end|>"])
    assert "PARAMETER num_ctx 2048" in make_modelfile("x.gguf")


@pytest.mark.parametrize("reply, lines", [
    (REPLY, ["- In stock: yes", "- Price: ₹2850", "- Delivery: yes", "- Other product offered: none"]),
    ('{"in_stock": false, "price_inr": null, "delivery": null, "alternative": "Black and Decker 13 mm drill; red"}',
     ["- In stock: no", "- Price: not said", "- Delivery: not said",
      "- Other product offered: Black and Decker 13 mm drill; red"]),
    ('Sure: {"in_stock": true, "price_inr": 13500.0, "delivery": false, "alternative": null}',
     ["- In stock: yes", "- Price: ₹13500", "- Delivery: no", "- Other product offered: none"]),
])
def test_answer_in_plain_words(reply: str, lines: list[str]) -> None:
    assert app.words(reply).splitlines() == lines


@pytest.mark.parametrize("reply", ['{"in_stock": true, "price_inr": 220', "sorry, I don't know", "",
                                   '{"in_stock": true, "price_inr": 220, "delivery": false}'])
def test_bad_reply_gets_a_clear_message(reply: str) -> None:
    assert app.words(reply).startswith("**No answer:** the model's reply was not the 4-part JSON")


def test_questions_are_checked() -> None:
    assert app.problem("16A MCB", "  ") == "Please type what the shopkeeper said."
    assert "500 characters" in app.problem("x" * 501, "haan hai")
    assert app.problem("", "haan hai") is None  # the product may be empty


def test_examples_are_new_sentences() -> None:
    """Written for the page: not from the test sets or the demo questions, and not even close to them."""
    rows = [{"product": p, "said": s} for p, s in app.EXAMPLES]
    assert len(rows) >= 3 and all(p.strip() and s.strip() for p, s in app.EXAMPLES)
    others = [ex for name in [*held_out_names(), "demo_questions.jsonl", "train.jsonl", "train_v4.jsonl"]
              for ex in load(name)]
    said = {nfc_lower(ex["said"]).strip() for ex in others}
    assert not {nfc_lower(ex["said"]).strip() for ex in rows} & said
    six = set().union(*(runs(ex["said"], 6) for ex in others))
    assert not [ex["said"] for ex in rows if runs(ex["said"], 6) & six]
    held_out = [ex for name in [*held_out_names(), "demo_questions.jsonl"] for ex in load(name)]
    products = {nfc_lower(ex["product"]).strip() for ex in held_out}
    assert not [ex["product"] for ex in rows if nfc_lower(ex["product"]).strip() in products]


# --- the app without gradio and llama_cpp ----------------------------------------------------------------------------

def test_app_imports_without_gradio_llama_cpp_or_huggingface_hub() -> None:
    code = NO_MODULES + (f"sys.path.insert(0, {str(REPO)!r})\nfrom space import app\n"
                         "print(app.prompt('a', 'b').count('<|im_start|>'), app.words('x')[:14])\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["3", "**No", "answer:**"]


class FakeLlama:
    def __init__(self, reply: str = REPLY, **settings) -> None:
        self.reply, self.settings, self.asked = reply, settings, []

    def create_completion(self, text, **kw):
        self.asked.append((text, kw))
        return {"choices": [{"text": " " + self.reply + "\n"}]}


def ready_model(reply: str = REPLY) -> app.Model:
    model = app.Model()
    model.llm = FakeLlama(reply)
    model.done.set()
    return model


def test_answer_asks_with_the_training_prompt_and_settings(capsys) -> None:
    model = ready_model()
    (shown,) = list(app.answer(model, " Asian Paints 10 L ", " haan hai, 2850 lagega "))
    plain, raw, seconds = shown
    assert plain.startswith("- In stock: yes\n- Price: ₹2850") and raw == REPLY
    assert re.fullmatch(r"Answered in \d+\.\d s on a free CPU\.", seconds)
    assert model.llm.asked == [(app.prompt("Asian Paints 10 L", "haan hai, 2850 lagega"),
                                {"max_tokens": 96, "temperature": 0, "stop": ["<|im_end|>"]})]
    assert "haan hai" not in capsys.readouterr().out  # the log has the speed, not what was asked


def test_answer_says_loading_until_the_model_is_ready() -> None:
    model = app.Model()
    steps = app.answer(model, "16A MCB", "haan hai")
    assert next(steps) == (app.LOADING, "", "")
    assert "1–2 minutes" in app.LOADING and app.LOADING in model.status()
    model.llm = FakeLlama()
    threading.Timer(0.05, model.done.set).start()
    assert next(steps)[1] == REPLY
    assert model.status() == "The model is ready."


def test_answer_checks_the_question_and_shows_a_load_error() -> None:
    model = ready_model()
    assert list(app.answer(model, "x", "")) == [("Please type what the shopkeeper said.", "", "")]
    assert model.llm.asked == []
    model.error = "RuntimeError: no file"
    assert list(app.answer(model, "x", "haan")) == [("The model did not load: RuntimeError: no file", "", "")]
    assert "did not load" in model.status()


def test_model_loads_from_the_private_repo_and_warms_up(monkeypatch) -> None:
    calls = {}

    def download(**kw):
        calls["download"] = kw
        return "/cache/model.gguf"

    def llama(**kw):
        calls["llama"] = FakeLlama(**kw)
        return calls["llama"]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=llama))
    monkeypatch.setenv("MODEL_REPO", "me/wimb-shop-model-v4")
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("N_THREADS", "2")
    model = app.Model()
    model.load()
    assert model.error is None and model.done.is_set() and model.status() == "The model is ready."
    assert calls["download"] == {"repo_id": "me/wimb-shop-model-v4", "filename": app.MODEL_FILE, "token": TOKEN}
    assert calls["llama"].settings == {"model_path": "/cache/model.gguf", "n_ctx": 2048, "n_threads": 2,
                                       "n_threads_batch": 2, "verbose": False}
    assert calls["llama"].asked[0][0] == app.prompt("test", "haan hai")  # the warm-up


def test_model_without_a_repo_says_so(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_REPO", raising=False)
    model = app.Model()
    model.load()
    assert model.done.is_set() and "MODEL_REPO is not set" in model.status()


def test_threads_follow_the_container_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("N_THREADS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    assert app.cpu_threads(tmp_path) == 2
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert app.cpu_threads(tmp_path) == 16
    assert app.cpu_threads(tmp_path / "none") == 16
    monkeypatch.setenv("N_THREADS", "3")
    assert app.cpu_threads(tmp_path) == 3


# --- the Space's files -----------------------------------------------------------------------------------------------

def test_space_files_hold_a_copy_of_score_and_in_words(tmp_path: Path) -> None:
    assert deploy.write_space(tmp_path) == ["README.md", "app.py", "requirements.txt", "score.py", "shop_words.py"]
    assert (tmp_path / "score.py").read_text(encoding="utf-8") == (REPO / "score.py").read_text(encoding="utf-8")
    assert (tmp_path / "app.py").read_bytes() == (HERE / "app.py").read_bytes()
    # The Space on its own (no repo around it): app.py finds score.py and shop_words.py next to it.
    code = NO_MODULES + (f"sys.path.insert(0, {str(tmp_path)!r})\nimport app, score, shop_words, json\n"
                         "assert app.in_words is shop_words.in_words and app.SYSTEM is score.SYSTEM\n"
                         f"print(json.dumps([app.words({REPLY!r}), score.__file__, 'try_on_kaggle' in sys.modules]))\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True,
                         cwd=tmp_path)
    plain, score_file, repo_used = json.loads(out.stdout)
    assert plain == app.words(REPLY) and Path(score_file).parent == tmp_path and not repo_used
    answer = {"in_stock": True, "price_inr": 99.5, "delivery": None, "alternative": "x"}
    scope: dict = {}
    exec((tmp_path / "shop_words.py").read_text(encoding="utf-8"), scope)  # noqa: S102 - our own file
    assert scope["in_words"](answer) == try_on_kaggle.in_words(answer)


def pins() -> dict[str, str]:
    lines = [line.strip() for line in (HERE / "requirements.txt").read_text().splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    assert lines[0] == "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
    found = dict(re.fullmatch(r"([A-Za-z0-9_-]+)==(\d+(?:\.\d+)+)", line).groups() for line in lines[1:])
    assert set(found) == {"gradio", "huggingface_hub", "llama-cpp-python"}
    return found


def test_requirements_are_pinned() -> None:
    found = pins()
    assert tuple(map(int, found["llama-cpp-python"].split("."))) >= (0, 3, 9)  # Qwen3 support
    workflow = (REPO / ".github" / "workflows" / "hf_space.yml").read_text(encoding="utf-8")
    assert f"huggingface_hub=={found['huggingface_hub']}" in workflow  # the runner uses the same version


def test_space_readme_header() -> None:
    yaml = pytest.importorskip("yaml")  # comes with kaggle (through jupytext)
    text = (HERE / "README.md").read_text(encoding="utf-8")
    header = yaml.safe_load(re.match(r"---\n(.*?)\n---\n", text, re.DOTALL).group(1))
    assert header["title"] == "WIMB shop model" and header["sdk"] == "gradio" and header["app_file"] == "app.py"
    assert str(header["sdk_version"]) == pins()["gradio"]
    assert header["pinned"] is False and header["license"] == "other"
    assert len(header["short_description"]) <= 60


# --- deploy.py with a fake Hugging Face and a fake Kaggle -------------------------------------------------------------

class FakeHf:
    """Stands in for huggingface_hub.HfApi: records every call, in order."""

    def __init__(self, stages=("RUNNING", "BUILDING", "APP_STARTING", "RUNNING"), has_model=True,
                 has_space=False) -> None:
        self.calls, self.stages, self.has_model, self.uploaded = [], list(stages), has_model, {}
        self.has_space = has_space

    def repo_exists(self, repo_id, repo_type=None):
        return self.has_space if repo_type == "space" else False

    def __call__(self, token=None):  # HfApi(token=...)
        self.calls.append(("HfApi", token))
        return self

    def whoami(self):
        return {"name": "Pratik_9"}

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def update_repo_settings(self, repo_id, **kw):
        self.calls.append(("update_repo_settings", repo_id, kw))

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, **kw):
        data = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(path_or_fileobj).read_bytes()
        self.uploaded[(repo_id, path_in_repo)] = data
        self.calls.append(("upload_file", repo_id, path_in_repo))

    def upload_folder(self, *, repo_id, folder_path, **kw):
        for p in Path(folder_path).iterdir():
            self.uploaded[(repo_id, p.name)] = p.read_bytes()
        self.calls.append(("upload_folder", repo_id, kw["repo_type"]))

    def add_space_secret(self, repo_id, key, value, **kw):
        self.calls.append(("secret", repo_id, key, value))

    def add_space_variable(self, repo_id, key, value, **kw):
        self.calls.append(("variable", repo_id, key, value))

    def file_exists(self, repo_id, filename):
        return self.has_model

    def get_space_runtime(self, repo_id):
        stage = self.stages.pop(0) if len(self.stages) > 1 else self.stages[0]
        return SimpleNamespace(stage=SimpleNamespace(value=stage))

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


class FakeKaggle:
    """The training notebook's output: two pages of files; only the .gguf is downloaded."""

    def __init__(self) -> None:
        self.calls = []
        self.files = [["results.json", "lora/adapter_model.safetensors", "wimb-shop-model/Modelfile"],
                      ["wimb-shop-model/wimb-shop-model.Q4_K_M.gguf", "wimb-shop-model-training-v4.log"]]

    def kernels_output(self, kernel, path, file_pattern=None, force=False, page_token=None, page_size=20):
        self.calls.append((kernel, file_pattern, page_token))
        page = 1 if page_token == "next" else 0
        saved = []
        for name in self.files[page]:
            if re.search(file_pattern, name):
                (Path(path) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(path) / name).write_bytes(b"GGUF" + b"0" * 2000)
                saved.append(str(Path(path) / name))
        return saved, "next" if page == 0 else ""


def run_deploy(monkeypatch, hf: FakeHf, *argv: str, kaggle: FakeKaggle | None = None, token: str = TOKEN):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=hf))
    kaggle = kaggle or FakeKaggle()
    connected = []

    def connect(slug):
        connected.append(slug)
        return kaggle, f"ypk2748/{slug}"

    monkeypatch.setattr(run_on_kaggle, "connect", connect)
    monkeypatch.setattr(deploy.time, "sleep", lambda s: None)
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(sys, "argv", ["deploy.py", *argv])
    deploy.main()
    return kaggle, connected


def test_deploy_uploads_the_model_and_makes_a_private_space(monkeypatch, capsys) -> None:
    hf = FakeHf()
    kaggle, connected = run_deploy(monkeypatch, hf)
    out = capsys.readouterr().out
    assert connected == ["wimb-shop-model-training-v4"]
    assert [c[2] for c in kaggle.calls] == [None, "next"]  # every page of the output list
    assert all(c[1] == deploy.GGUF_PATTERN for c in kaggle.calls)
    model, space = "Pratik_9/wimb-shop-model-v4", "Pratik_9/wimb-shop-model"
    assert hf.named("create_repo") == [
        ("create_repo", model, {"repo_type": "model", "private": True, "exist_ok": True}),
        ("create_repo", space, {"repo_type": "space", "space_sdk": "gradio", "space_hardware": "cpu-basic",
                                "private": True, "exist_ok": True})]
    assert hf.named("update_repo_settings") == [("update_repo_settings", space, {"private": True, "repo_type": "space"})]
    assert hf.uploaded[(model, "wimb-shop-model.Q4_K_M.gguf")].startswith(b"GGUF")
    assert hf.uploaded[(model, "Modelfile")] == (REPO / "Modelfile").read_bytes()
    card = hf.uploaded[(model, "README.md")].decode("utf-8")
    for want in ("Qwen/Qwen3-4B-Instruct-2507", "Apache-2.0", "CC BY-SA 4.0", "CDLA-Sharing-1.0", "ODC-By",
                 "private by", "| test (30) | 100% | 100% |", "| hard (56) | 77% | 80% |", "82%", "95%",
                 "ypk2748/wimb-shop-model-training-v4", f"https://huggingface.co/spaces/{space}"):
        assert want in card, want
    assert {name for repo, name in hf.uploaded if repo == space} == {
        "app.py", "requirements.txt", "README.md", "score.py", "shop_words.py"}
    assert hf.uploaded[(space, "score.py")] == (REPO / "score.py").read_bytes()
    assert hf.named("secret") == [("secret", space, "HF_TOKEN", TOKEN)]
    assert hf.named("variable") == [("variable", space, "MODEL_REPO", model),
                                    ("variable", space, "MODEL_FILE", "wimb-shop-model.Q4_K_M.gguf")]
    order = [c[0] for c in hf.calls]
    assert order.index("secret") < order.index("upload_folder")  # the first start already has the token
    assert TOKEN not in out and not any(TOKEN.encode() in data for data in hf.uploaded.values())
    assert f"https://huggingface.co/spaces/{space}" in out and "https://pratik-9-wimb-shop-model.hf.space" in out
    assert "read-only token" in out and "Space: BUILDING" in out and "The Space is running" in out
    assert "Model file: wimb-shop-model.Q4_K_M.gguf, 0.00 GB" in out


def test_deploy_v2_public(monkeypatch, capsys) -> None:
    hf = FakeHf()
    _, connected = run_deploy(monkeypatch, hf, "--model", "v2", "--public", "--space-name", "shop")
    assert connected == ["wimb-shop-model-training"]
    creates = {c[1]: c[2] for c in hf.named("create_repo")}
    assert creates["Pratik_9/wimb-shop-model-v2"]["private"] is True  # the model file stays private
    assert creates["Pratik_9/shop"]["private"] is False
    assert ("variable", "Pratik_9/shop", "MODEL_REPO", "Pratik_9/wimb-shop-model-v2") in hf.calls
    assert "| fresh (60, written blind) | 77% | 70% |" in hf.uploaded[("Pratik_9/wimb-shop-model-v2", "README.md")].decode()
    assert "public page" in capsys.readouterr().out


def test_deploy_can_skip_the_model_upload(monkeypatch) -> None:
    hf = FakeHf()
    _, connected = run_deploy(monkeypatch, hf, "--skip-upload-model")
    assert connected == [] and not [k for k in hf.uploaded if k[0].endswith("-v4")]
    assert ("upload_folder", "Pratik_9/wimb-shop-model", "space") in hf.calls
    with pytest.raises(SystemExit) as stop:
        run_deploy(monkeypatch, FakeHf(has_model=False), "--skip-upload-model")
    assert "has no wimb-shop-model.Q4_K_M.gguf" in str(stop.value.code)


def test_deploy_keeps_the_token_of_a_space_that_is_already_there(monkeypatch, capsys) -> None:
    hf = FakeHf(has_space=True)
    run_deploy(monkeypatch, hf, "--skip-upload-model")
    assert hf.named("secret") == [] and "HF_TOKEN kept as it is" in capsys.readouterr().out
    hf = FakeHf(has_space=True)
    run_deploy(monkeypatch, hf, "--skip-upload-model", "--set-space-token")
    assert hf.named("secret") == [("secret", "Pratik_9/wimb-shop-model", "HF_TOKEN", TOKEN)]
    assert TOKEN not in capsys.readouterr().out


def test_deploy_stops_when_the_kaggle_output_has_no_model(monkeypatch) -> None:
    kaggle = FakeKaggle()
    kaggle.files = [["results.json"], []]
    with pytest.raises(SystemExit) as stop:
        run_deploy(monkeypatch, FakeHf(), kaggle=kaggle)
    assert "No .Q4_K_M.gguf" in str(stop.value.code)


def test_deploy_fails_with_the_log_link_when_the_space_does_not_build(monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        run_deploy(monkeypatch, FakeHf(stages=("BUILDING", "BUILD_ERROR")), "--skip-upload-model")
    assert "BUILD_ERROR" in str(stop.value.code) and "?logs=build" in str(stop.value.code)
    assert TOKEN not in capsys.readouterr().out


def test_deploy_needs_the_hf_token(monkeypatch) -> None:
    with pytest.raises(SystemExit) as stop:
        run_deploy(monkeypatch, FakeHf(), token="")
    assert "Set HF_TOKEN" in str(stop.value.code)


def test_wait_counts_running_only_for_the_new_build(monkeypatch, capsys) -> None:
    monkeypatch.setattr(deploy.time, "sleep", lambda s: None)
    hf = FakeHf(stages=["RUNNING"] * 3 + ["RUNNING_BUILDING", "RUNNING"])
    assert deploy.wait_running(hf, "me/s") == "RUNNING"
    assert hf.stages == ["RUNNING"]  # it waited for the build, then saw RUNNING again
    hf = FakeHf(stages=["RUNNING"])  # no build seen (nothing changed): RUNNING counts after 2 minutes
    assert deploy.wait_running(hf, "me/s") == "RUNNING"
    hf = FakeHf(stages=["BUILDING"])
    assert deploy.wait_running(hf, "me/s", minutes=1) == "BUILDING"  # time is up
    hf.get_space_runtime = lambda repo_id: SimpleNamespace(stage="RUNTIME_ERROR")  # a plain string works too
    assert deploy.wait_running(hf, "me/s") == "RUNTIME_ERROR"


def test_app_url() -> None:
    assert deploy.app_url("Pratik_9", "wimb-shop-model") == "https://pratik-9-wimb-shop-model.hf.space"
    assert deploy.gguf_name() == "wimb-shop-model.Q4_K_M.gguf" == app.MODEL_FILE


# --- the workflow ----------------------------------------------------------------------------------------------------

WORKFLOW = REPO / ".github" / "workflows" / "hf_space.yml"


def step_with(text: str) -> dict:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    (step,) = [s for s in workflow["jobs"]["deploy"]["steps"] if text in s.get("run", "")]
    return step


def workflow_command(tmp_path: Path, **env: str) -> list[str]:
    """What the workflow's main step runs, with these inputs: the arguments it gives python."""
    step = step_with("space/deploy.py")
    assert set(env) <= set(step["env"])
    bin_dir = tmp_path / "bin"  # a stand-in `python` that writes down its arguments, one per line
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python").write_text(f'#!/bin/sh\nfor a in "$@"; do echo "$a"; done > {tmp_path}/args\n')
    (bin_dir / "python").chmod(0o755)
    run_env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", **env}
    subprocess.run(["bash", "-e", "-c", step["run"]], env=run_env, capture_output=True, check=True)
    return (tmp_path / "args").read_text().splitlines()


def test_workflow_deploys_with_inputs_only_through_the_environment(tmp_path: Path) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for line in text.splitlines():  # inputs reach the scripts only as environment variables
        if "${{ inputs." in line:
            assert re.fullmatch(r" +[A-Z_]+: \$\{\{ inputs\.\w+ \}\}", line), line
    for want in ("name: Put the model on Hugging Face (free web page)", "group: hf-space", "timeout-minutes: 60",
                 "python -m pytest -q", "options: [v4, v2]", "options: [private, public]", "default: private"):
        assert want in text, want
    assert workflow_command(tmp_path, MODEL="v4", VISIBILITY="private") == [
        "space/deploy.py", "--model", "v4", "--private"]
    assert workflow_command(tmp_path, MODEL="v2", VISIBILITY="public") == [
        "space/deploy.py", "--model", "v2", "--public"]
    assert workflow_command(tmp_path, MODEL="v4; echo hi", VISIBILITY="") == [
        "space/deploy.py", "--model", "v4; echo hi", "--private"]
    assert step_with("space/deploy.py")["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"


def test_workflow_stops_when_a_secret_is_missing(tmp_path: Path) -> None:
    step = step_with("KAGGLE_API_TOKEN")
    assert set(step["env"]) == {"KAGGLE_API_TOKEN", "HF_TOKEN"}
    script = step["run"].replace("${{ github.repository }}", "o/r")
    run = lambda **env: subprocess.run(["bash", "-e", "-c", script], env={"PATH": os.environ["PATH"], **env},
                                       capture_output=True, text=True, check=False)
    missing = run(KAGGLE_API_TOKEN="k")
    assert missing.returncode == 1 and "HF_TOKEN" in missing.stdout and "KAGGLE" not in missing.stdout
    assert "https://github.com/o/r/settings/secrets/actions/new" in missing.stdout
    assert run(HF_TOKEN="h").returncode == 1
    assert run(KAGGLE_API_TOKEN="k", HF_TOKEN="h").returncode == 0
