"""Checks the voice bot's parts offline (no model, no audio): the number reader, the language guess, the text for the
voice, the tasks, the understanding (rules, and a fake model), the conversation and the test calls.
Run:  python -m pytest voicebot"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from voicebot import evaluate, numerals, tasks
from voicebot.engine import Call
from voicebot.lang import detect
from voicebot.speech_text import clean_heard, for_voice
from voicebot.understand import View, by_rules, check, prompt, schema, understand


@pytest.mark.parametrize("said, value", [
    ("ढाई सौ का है", 250), ("दो सौ बीस रुपये", 220), ("साढ़े तीन सौ", 350), ("पौने दो हज़ार", 1750),
    ("सवा दो सौ", 225), ("डेढ़ सौ रुपये", 150), ("एक हज़ार पाँच सौ", 1500), ("पंद्रह सौ", 1500), ("बारह सौ पचास", 1250),
    ("दोनशे वीस रुपये", 220), ("एकशे ऐंशी", 180), ("अडीचशे रुपये", 250), ("साडेतीनशे", 350), ("पावणेतीनशे", 275),
    ("दीडशे रुपयांना", 150), ("do sau pachas", 250), ("dhai sau", 250), ("saade teen sau", 350),
    ("two hundred and twenty rupees", 220), ("रेट टू थर्टी है", 230), ("₹245", 245), ("2 सौ", 200), ("1,250/-", 1250),
    ("डेढ़ लाख", 150_000), ("सवा लाख", 125_000), ("दोन हजार तीनशे", 2300),
])
def test_spoken_prices(said, value):
    assert numerals.money(said, expected=True) == value


@pytest.mark.parametrize("said, value", [
    ("सोलह ए वाला है, दो सौ बीस रुपये", 220),  # 16A is a size
    ("दो पीस हैं, एक सौ नब्बे का", 190),  # 2 is a count
    ("5 किलो 300 रुपये", 300),
    ("एक का एक सौ नब्बे", 190),
])
def test_prices_skip_sizes_and_counts(said, value):
    assert numerals.money(said, expected=True) == value


def test_a_lone_small_word_is_no_price():
    assert numerals.money("नहीं, बता दो") is None
    assert numerals.money("हाँ, दस बारह पीस हैं") is None  # not asked, no sign of money
    assert numerals.money("पचास") is None and numerals.money("पचास", expected=True) == 50


@pytest.mark.parametrize("said, hhmm", [
    ("शाम साढ़े चार बजे", "16:30"), ("सकाळी दहा वाजता", "10:00"), ("चार बजे", "16:00"), ("10:30 को", "10:30"),
    ("चार बजकर बीस मिनट", "16:20"), ("शाम पाँच", "17:00"), ("दुपारी दोन वाजता", "14:00"), ("रात नौ बजे", "21:00"),
    ("subah das baje", "10:00"), ("around 3:30 in the afternoon", "15:30"), ("बारह बजे", "12:00"),
])
def test_times(said, hhmm):
    assert numerals.time_of_day(said) == hhmm


@pytest.mark.parametrize("said, d", [
    ("कल आऊँगा", "tomorrow"), ("उद्या", "tomorrow"), ("परसों", "day_after"), ("परवा", "day_after"),
    ("सोमवार को", "monday"), ("मंगळवारी", "tuesday"), ("kal shaam", "tomorrow"), ("Wednesday", "wednesday"),
    ("the day after tomorrow", "day_after"), ("आज", "today"),
])
def test_days(said, d):
    assert numerals.day(said) == d


@pytest.mark.parametrize("said, number", [
    ("नौ आठ सात छह पाँच, चार तीन दो एक शून्य", "9876543210"), ("98765 43210", "9876543210"),
    ("अट्ठानवे सत्तासी छिहत्तर पचपन चौंतीस", "9887765534"), ("नऊ आठ डबल सात सहा पाच चार तीन दोन एक", "9877654321"),
    ("+91 98123 45678", "9812345678"), ("09812345678", "9812345678"),
])
def test_phone_numbers(said, number):
    assert numerals.phone(said) == number


def test_too_short_is_no_phone():
    assert numerals.phone("नौ आठ सात") is None


def test_numbers_back_to_words():
    assert numerals.to_words(220, "hi") == "दो सौ बीस"
    assert numerals.to_words(220, "mr") == "दोनशे वीस"
    assert numerals.to_words(225, "en") == "two hundred twenty five"
    assert numerals.money_words(12.5, "hi") == "बारह रुपये पचास पैसे"
    assert numerals.time_words("16:30", "hi") == "शाम साढ़े चार बजे"
    assert numerals.time_words("13:30", "mr") == "दुपारी दीड वाजता"
    assert numerals.time_words("09:45", "hi") == "सुबह पौने दस बजे"
    assert numerals.time_words("16:05", "en") == "four oh five in the evening"
    assert numerals.digit_words("9876543210", "hi") == "नौ आठ सात छह पाँच, चार तीन दो एक शून्य"


def test_annotate_adds_digits_for_the_model():
    assert numerals.annotate("ढाई सौ का है") == "ढाई सौ (=250) का है"
    assert numerals.annotate("₹220 है") == "₹220 है"


@pytest.mark.parametrize("said, lang", [
    ("हो, आहे ना, दोनशे रुपये", "mr"), ("haan bhai hai, do sau ka", "hi"), ("हाँ जी है, दो सौ बीस का", "hi"),
    ("yes it is available, two hundred", "en"), ("नाही, संपलं आहे", "mr"), ("udya milel", "mr"),
])
def test_language(said, lang):
    assert detect(said)[0] == lang


def test_text_for_the_voice():
    assert for_voice("16A MCB ₹220 में है", "hi") == "सोलह ए एम सी बी दो सौ बीस रुपये में है"
    assert for_voice("नंबर 98765 43210", "mr") == "नंबर नऊ आठ सात सहा पाच, चार तीन दोन एक शून्य"
    assert for_voice("5kg चावल", "hi") == "पाँच किलो चावल"
    assert "प्रतिशत" in for_voice("25% छूट", "hi")


def test_known_hallucinations_are_dropped():
    assert clean_heard("Thank you for watching!") == ""
    assert clean_heard("ご視聴ありがとうございました") == ""
    assert clean_heard("हाँ हाँ हाँ हाँ हाँ") == "हाँ हाँ"
    assert clean_heard("दो सौ बीस दो सौ बीस दो सौ बीस दो सौ बीस") == "दो सौ बीस"
    assert clean_heard("  हाँ जी है  ") == "हाँ जी है"


def test_every_task_is_valid_and_every_scenario_names_one():
    for task_id in tasks.available():
        spec = json.loads((tasks.TASKS_DIR / f"{task_id}.json").read_text(encoding="utf-8"))
        assert tasks.problems(spec) == [], task_id
    assert set(tasks.available()) >= {"shop_stock", "clinic_booking", "order_confirm"}
    for row in evaluate.scenarios():
        assert row["task"] in tasks.available()
        names = set(tasks.load(row["task"]).slot)
        assert set(row.get("replies") or {}) <= names | {"confirm", "fix", "*"}, row["id"]
        assert set((row.get("expect") or {}).get("values") or {}) <= names, row["id"]


def test_scenario_ids_are_unique():
    ids = [r["id"] for r in evaluate.scenarios()]
    assert len(ids) == len(set(ids))


def test_task_problems_are_explained():
    spec = json.loads((tasks.TASKS_DIR / "shop_stock.json").read_text(encoding="utf-8"))
    spec["slots"][1]["type"] = "price"
    spec["slots"][2]["when"] = {"nothing": True}
    spec["greeting"] = {"hi": "नमस्ते {customer}"}
    found = " | ".join(tasks.problems(spec))
    assert "type must be one of" in found and "'nothing' must be a slot" in found
    assert "no mr, en text" in found and "{customer}" in found
    with pytest.raises(tasks.TaskError):
        tasks.from_json("{not json")


def test_context_changes_the_product():
    call = Call(tasks.load("shop_stock", {"product": "LED बल्ब"}))
    assert "LED बल्ब" in call.start().text


def test_check_values_by_type():
    task = tasks.load("clinic_booking")
    assert check(task.slot["phone"], "+91 98765-43210") == "9876543210"
    assert check(task.slot["phone"], "12345") is None
    assert check(task.slot["time"], "4:30") == "04:30" and check(task.slot["time"], 17) == "17:00"
    assert check(task.slot["day"], "Day after tomorrow") == "day_after" and check(task.slot["day"], "someday") is None
    order = tasks.load("order_confirm")
    assert check(order.slot["delivery_time"], "शाम") == "evening" and check(order.slot["delivery_time"], "x") is None
    shop = tasks.load("shop_stock")
    assert check(shop.slot["price_inr"], "₹2,500") == 2500 and check(shop.slot["price_inr"], -5) is None
    assert check(shop.slot["in_stock"], "yes") is True and check(shop.slot["in_stock"], 1) is None


def test_rules_read_short_replies_without_the_model():
    shop = tasks.load("shop_stock")
    u = by_rules("हाँ है", shop, View(pending="in_stock"))
    assert u.values == {"in_stock": True} and u.fast
    u = by_rules("हाँ है ना", shop, View(pending="in_stock"))
    assert u.values == {"in_stock": True}  # "है ना" is a yes
    u = by_rules("नहीं जी", shop, View(pending="delivery"))
    assert u.values == {"delivery": False} and u.fast
    u = by_rules("पता नहीं", shop, View(pending="price_inr"))
    assert u.intent == "dont_know" and not u.values
    assert by_rules("बार बार फोन मत करो, मैं बिलकुल भी इंटरेस्टेड नहीं हूँ भाई", shop, View()).intent == "stop_calling"
    assert by_rules("हाँ सही है", shop, View(pending="confirm")).intent == "confirm_yes"


class FakeModel:
    """A pretend model: returns the JSON it was given, and remembers the prompts."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, text, form):
        self.prompts.append((text, form))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def test_the_model_reads_long_replies_and_its_slips_are_caught():
    shop = tasks.load("shop_stock")
    said = "हाँ है, ढाई सौ का पड़ेगा, और घर पे भी पहुँचा देंगे"
    model = FakeModel({"intent": "answer", "in_stock": True, "price_inr": 205, "delivery": True, "alternative": None,
                       "fact": None})
    u = understand(said, shop, View(pending="in_stock"), model)
    assert u.values == {"in_stock": True, "price_inr": 250, "delivery": True}  # 205 was the model's slip
    text, form = model.prompts[0]
    assert "ढाई सौ (=250)" in text and "price_inr" in text and form["properties"]["price_inr"]["type"] == ["number",
                                                                                                         "null"]


def test_a_price_nobody_said_is_dropped():
    shop = tasks.load("shop_stock")
    model = FakeModel({"intent": "answer", "in_stock": True, "price_inr": 300, "delivery": None, "alternative": None})
    assert understand("हाँ रखा है अपने पास", shop, View(pending="in_stock"), model).values == {"in_stock": True}


def test_a_model_failure_falls_back_to_the_rules():
    shop = tasks.load("shop_stock")
    for broken in (TimeoutError("slow"), "not json at all", "[1, 2]"):
        u = understand("हाँ है भाई, दुकान पर रखा है अभी", shop, View(pending="in_stock"), FakeModel(broken))
        assert u.values == {"in_stock": True} and u.error and u.source == "rules"


def test_a_clear_stop_wins_over_the_model():
    shop = tasks.load("shop_stock")
    model = FakeModel({"intent": "answer", "in_stock": None, "price_inr": None, "delivery": None, "alternative": None})
    u = understand("दोबारा कॉल मत करना, हमें नहीं चाहिए कुछ भी समझे", shop, View(pending="in_stock"), model)
    assert u.intent == "stop_calling"


def test_schema_and_prompt_cover_every_slot_and_fact():
    clinic = tasks.load("clinic_booking")
    form = schema(clinic)
    assert set(form["properties"]) == {"intent", "fact", *clinic.slot}
    assert form["required"] == list(form["properties"])
    assert None in form["properties"]["day"]["enum"] and "fee" in form["properties"]["fact"]["enum"]
    text = prompt(clinic, View(pending="phone", last_bot="आपका मोबाइल नंबर बता दीजिए।"), "नौ आठ सात")
    assert "phone" in text and "10 digits" in text and "fee:" in text and "आपका मोबाइल नंबर" in text


def talk(task_id, lines, llm=None, **context):
    call = Call(tasks.load(task_id, context or None), llm=llm)
    replies = [call.start()]
    for line in lines:
        replies.append(call.hear(line))
        if replies[-1].done:
            break
    return call, replies


def test_a_whole_shop_call():
    call, replies = talk("shop_stock", ["हाँ है", "ढाई सौ का", "हाँ कर देंगे", "हाँ सही है"])
    assert "क्या आपके पास 16A MCB है?" in replies[0].text and "सोलह ए एम सी बी" in replies[0].speech
    assert "दाम ₹250" in replies[3].text and "दो सौ पचास रुपये" in replies[3].speech
    assert replies[-1].done and call.result()["outcome"] == "done"
    assert call.result()["values"] == {"in_stock": True, "price_inr": 250, "delivery": True, "alternative": None}


def test_the_bot_switches_to_marathi():
    call, replies = talk("shop_stock", ["हो, आहे", "दोनशे वीस रुपये", "नाही, डिलिव्हरी नाही", "हो बरोबर"])
    assert call.lang == "mr" and "बरोबर आहे का?" in replies[3].text and call.result()["outcome"] == "done"


def test_a_correction_at_the_read_back():
    call, _ = talk("shop_stock", ["है", "दो सौ बीस", "हाँ", "नहीं, दाम दो सौ चालीस है", "हाँ"])
    assert call.result()["values"]["price_inr"] == 240 and call.outcome == "done"


def test_a_slot_is_asked_twice_at_most():
    call, replies = talk("shop_stock", ["हाँ है", "हम्म", "क्या पता भाई", "नहीं", "हाँ"])
    asked = sum("दाम कितना है" in r.text for r in replies)
    assert asked == 2 and call.status["price_inr"] == "unknown" and call.outcome == "partial"


def test_the_clinic_keeps_its_rules():
    call, replies = talk("clinic_booking", ["राहुल", "रविवार", "सोमवार", "रात नौ बजे", "शाम पाँच बजे", "9876543210",
                                            "बुखार", "हाँ"])
    assert "रविवार को क्लिनिक बंद" in replies[2].text and "उस समय क्लिनिक बंद" in replies[4].text
    assert call.result()["values"]["day"] == "monday" and call.result()["values"]["time"] == "17:00"


def test_a_question_gets_the_fact_then_the_question_again():
    model = FakeModel({"intent": "question", "fact": "fee", "patient_name": None, "day": None, "time": None,
                       "phone": None, "reason": None})
    _, replies = talk("clinic_booking", ["डॉक्टर की फीस कितनी है?"], llm=model)
    assert "₹500" in replies[1].text and "मरीज़ का नाम" in replies[1].text and "पाँच सौ रुपये" in replies[1].speech


def test_ends_politely():
    for lines, outcome in ((["अभी बिज़ी हूँ, बाद में कॉल करना"], "callback"), (["कॉल मत करो"], "do_not_call"),
                           (["रॉन्ग नंबर है"], "wrong_number"), (["किसी इंसान से बात कराओ"], "handoff"),
                           (["", "", ""], "no_answer")):
        call, replies = talk("shop_stock", lines)
        assert replies[-1].done and call.outcome == outcome and replies[-1].text


def test_hold_then_come_back_is_not_a_yes():
    call, replies = talk("shop_stock", ["एक मिनट रुकिए", "हाँ बोलिए"])
    assert "रुकती" in replies[1].text and "in_stock" not in call.values and "16A MCB है?" in replies[2].text


def test_the_rules_only_baseline_passes_most_test_calls():
    results = [evaluate.play(s) for s in evaluate.scenarios()]
    passed = sum(not r["wrong"] for r in results)
    assert passed >= len(results) - 3, evaluate.summary_lines(results)


def test_the_spoken_test_goes_through_say_and_hear():
    heard = []

    bot = []

    def say_and_hear(line, call):
        heard.append((line, call.lang))
        return line, 0.5

    row = next(s for s in evaluate.scenarios() if s["id"] == "shop-hi-basic")
    result = evaluate.play(row, say_and_hear=say_and_hear, on_bot=bot.append)
    assert not result["wrong"] and heard[0] == ("हाँ जी, है", "hi")
    assert len(bot) == len(heard) + 1 and bot[-1].done


# --- the Kaggle notebook and its runner ------------------------------------------------------------------------
from voicebot import build_voicebot, run_voicebot


def nb_cell(nb: dict, title: str) -> str:
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(f"#@title {title}"))["source"])


def test_notebook_is_up_to_date():
    saved = (HERE / build_voicebot.NOTEBOOK).read_text(encoding="utf-8")
    assert saved == build_voicebot.text(build_voicebot.notebook()), "run: python voicebot/build_voicebot.py"


def test_notebook_carries_the_whole_package():
    files = build_voicebot.package_files()
    assert {"engine.py", "understand.py", "numerals.py", "audio.py", "live.py", "tasks/shop_stock.json",
            "scenarios/shop_stock.jsonl"} <= set(files)
    for name, body in files.items():
        assert body == (HERE / name).read_text(encoding="utf-8")
    assert not any(n.startswith(("test_", "build_", "run_")) for n in files)


def test_every_code_cell_is_python():
    import ast

    nb = json.loads((HERE / build_voicebot.NOTEBOOK).read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))


def test_runner_sets_the_settings():
    nb = run_voicebot.notebook_with({"MODE": "live", "BRAINS": ["gemma3:4b"], "LIVE_MINUTES": 30,
                                     "LIVE_PASSWORD": 'pa"ss', "NOTIFY_TOPIC": "wimb-abc"})
    settings = nb_cell(nb, build_voicebot.SETTINGS_TITLE)
    space: dict = {}
    exec(compile(settings, "settings", "exec"), space)  # noqa: S102 - our own notebook cell
    assert space["MODE"] == "live" and space["BRAINS"] == ["gemma3:4b"] and space["LIVE_PASSWORD"] == 'pa"ss'
    assert space["LIVE_MINUTES"] == 30 and space["NOTIFY_TOPIC"] == "wimb-abc"
    with pytest.raises(KeyError):
        run_voicebot.notebook_with({"NOPE": 1})


def test_brain_names_are_checked():
    assert run_voicebot.brains_from("gemma3:4b, qwen3:4b-instruct-2507-q4_K_M") == [
        "gemma3:4b", "qwen3:4b-instruct-2507-q4_K_M"]
    assert run_voicebot.brains_from("  ") is None
    with pytest.raises(ValueError):
        run_voicebot.brains_from("gemma3:4b; rm -rf /")


def test_the_link_comes_back_through_ntfy(monkeypatch):
    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body.encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    bodies = ['{"event":"open"}\n',
              ('{"event":"message","message":"https://evil.example.com"}\n'
               '{"event":"message","message":"https://abc123def.gradio.live"}\n')]
    monkeypatch.setattr(run_voicebot.urllib.request, "urlopen", lambda url, timeout: Resp(bodies.pop(0)))
    monkeypatch.setattr(run_voicebot.time, "sleep", lambda s: None)
    assert run_voicebot.wait_for_link("wimb-x", minutes=1) == "https://abc123def.gradio.live"


def test_live_needs_a_password(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_voicebot.py", "--live", "30"])
    monkeypatch.delenv("VOICEBOT_PASSWORD", raising=False)
    with pytest.raises(SystemExit) as stop:
        run_voicebot.main()
    assert "VOICEBOT_PASSWORD" in str(stop.value)


def test_notebook_glue_runs_with_a_fake_brain(tmp_path, monkeypatch):
    """The helper cell writes and imports the package; the typed test cell plays every test call with a brain."""
    nb = json.loads((HERE / build_voicebot.NOTEBOOK).read_text(encoding="utf-8"))
    space: dict = {}
    settings = nb_cell(nb, build_voicebot.SETTINGS_TITLE).replace('"/kaggle/working/voicebot"', repr(str(tmp_path)))
    exec(compile(settings, "settings", "exec"), space)  # noqa: S102
    helpers = nb_cell(nb, "2. Helpers").replace('"/tmp/voicebot_src"', repr(str(tmp_path / "src")))
    saved_modules = {k: v for k, v in sys.modules.items() if k == "voicebot" or k.startswith("voicebot.")}
    try:
        for k in saved_modules:
            del sys.modules[k]
        exec(compile(helpers, "helpers", "exec"), space)  # noqa: S102
        assert (tmp_path / "src" / "voicebot" / "engine.py").exists()

        def make_llm(model):
            space["LLM_SECONDS"].setdefault(model, [])

            def llm(prompt_text, form):
                space["LLM_SECONDS"][model].append(0.5)
                return json.dumps({k: None for k in form["properties"]} | {"intent": "other"})

            return llm

        space.update(READY=["fake:1b"], LLM_SECONDS={}, make_llm=make_llm)
        source = nb_cell(nb, "5. Test calls, typed")
        source = source[source.index("\n") + 1:]  # without the title line
        source = source.replace("def text_test", "def _unused_text_test", 0)
        exec(compile(source.replace("def make_llm", "def _x"), "text test", "exec"), space)  # noqa: S102
        r = space["RESULTS"]
        assert set(r["text_test"]) == {"rules only", "fake:1b"} and r["best_brain"] == "fake:1b"
        assert r["text_test"]["fake:1b"]["model_seconds_avg"] == 0.5
        assert (tmp_path / build_voicebot.RESULTS_FILE).exists()
    finally:
        for k in [k for k in sys.modules if k == "voicebot" or k.startswith("voicebot.")]:
            del sys.modules[k]
        sys.modules.update(saved_modules)


@pytest.mark.parametrize("script", ["run_voicebot.py", "evaluate.py", "build_voicebot.py"])
def test_scripts_start_as_scripts(script, tmp_path):
    """Run as `python voicebot/<script>`, the folder voicebot/ is first on the path: nothing in it may hide a
    standard module (a voicebot/numbers.py once broke tqdm's `from numbers import Number`)."""
    import subprocess

    args = ["--help"]
    done = subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True, timeout=120,
                          cwd=tmp_path, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    clash = [p.name for p in HERE.glob("*.py") if p.stem in sys.stdlib_module_names]
    assert not clash, f"rename: {clash} hide standard modules"
