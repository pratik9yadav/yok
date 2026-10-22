"""Make the training data: what Pune shopkeepers say on a call, in Hindi, Marathi and English.

Version 2. The replies are still written from templates, but they are built on real data
(see data/real/NOTICE.md and data/TRAIN_NOTICE.md):
  - product names with their real MRP and selling price (Flipkart, Amazon India),
  - how each price is said in Hindi and Marathi words (data/real/price_words.json),
  - how people really spell Hindi and Marathi words in Latin letters
    (data/real/romanization_variants.json).
Version 2.1 (after an audit) takes away shortcuts a model could learn instead of reading: the product's
noun ("battery") is said in "in stock" replies as often as in "not in stock" ones; the asked price is
said first or last about equally often; the language does not depend on the answer; a real product's
price moves by up to 10% between rows. "Tomorrow" replies always say the item is not there now, and a
delivery fee is said with item-only wording ("maal ka 400, delivery alag").
The held-out sets data/test.jsonl and data/hard_test.jsonl are NOT used to make the data; they are
read only at the end, for the leakage check printed with the stats.
data/real/xpqa_hi_examples.jsonl is not included here (it is added separately at training time).

Row format (one JSON object per line):
    {"product": "...", "said": "...", "answer": {"in_stock": ..., "price_inr": ...,
     "delivery": ..., "alternative": ...}}

Run:  python make_data.py            (writes data/train.jsonl and prints stats; deterministic)
"""

import itertools
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
REAL = HERE / "data" / "real"
N_ROWS = 1300
SEED = 2026

# (product asked for, alternatives a shop might offer, typical price range in rupees)
PRODUCTS = [
    ("10 mm nut bolt (pack of 10)", ["12 mm bolt", "8 mm bolt"], (30, 120)),
    ("1.5 sq mm copper wire, 90 m roll", ["2.5 sq mm wire", "1 sq mm wire"], (1200, 2200)),
    ("Fevicol SH 1 kg", ["Fevicol 500 g", "local glue"], (300, 450)),
    ("1 inch PVC elbow", ["3/4 inch elbow", "CPVC elbow"], (10, 40)),
    ("Splendor brake shoe", ["local brake shoe", "Passion brake shoe"], (150, 400)),
    ("Activa clutch plate", ["local clutch plate", "Jupiter clutch plate"], (400, 900)),
    ("12V 7Ah battery", ["12V 9Ah battery", "12V 5Ah battery"], (900, 1600)),
    ("9W LED bulb", ["12W LED bulb", "7W LED bulb"], (60, 150)),
    ("16A MCB", ["20A MCB", "32A MCB"], (150, 350)),
    ("teflon tape", ["thread seal tape", "big teflon roll"], (10, 30)),
    ("fan capacitor 2.5 mfd", ["3.15 mfd capacitor", "2.25 mfd capacitor"], (40, 120)),
    ("Pulsar 150 spark plug", ["local spark plug", "Pulsar 180 spark plug"], (100, 250)),
    ("wall putty 20 kg", ["wall putty 40 kg", "local putty"], (600, 1000)),
    ("8 mm drill bit", ["10 mm drill bit", "6 mm drill bit"], (40, 150)),
    ("tank float valve", ["ball valve", "plastic float valve"], (150, 400)),
    ("4 inch door hinge", ["3 inch hinge", "steel hinge"], (30, 120)),
    ("50 mm padlock", ["40 mm padlock", "60 mm padlock"], (150, 500)),
    ("2 inch paint brush", ["3 inch brush", "paint roller"], (30, 90)),
    ("silicone sealant", ["M-seal", "white cement"], (150, 350)),
    ("Activa air filter", ["local air filter", "Dio air filter"], (150, 350)),
    ("10W-30 engine oil 1 L", ["20W-40 engine oil", "10W-30 engine oil 900 ml"], (300, 550)),
    ("bike headlight bulb", ["LED headlight bulb", "halogen bulb"], (60, 250)),
    ("4 socket extension board", ["6 socket extension board", "3 socket board"], (250, 600)),
    ("water pump seal kit", ["local seal kit", "mechanical seal"], (120, 400)),
]

# More about the products above: the short name of the asked variant ("16A nahi hai, 32A hai"),
# a short way to say each alternative ("9Ah wala"), the unit they are counted in, the noun used in
# "battery khatam ho gayi", whether a unit-vs-box price makes sense, and pack size.
OLD_EXTRA = {
    "10 mm nut bolt (pack of 10)": {
        "ask": "10 mm", "short": {"12 mm bolt": "12 mm", "8 mm bolt": "8 mm"}, "unit": "packet", "pack": 10},
    "1.5 sq mm copper wire, 90 m roll": {
        "ask": "1.5 sq mm", "short": {"2.5 sq mm wire": "2.5 sq mm", "1 sq mm wire": "1 sq mm"}, "unit": "roll",
        "noun": "wire"},
    "Fevicol SH 1 kg": {"ask": "1 kg", "short": {"Fevicol 500 g": "500 g"}, "unit": "dabba"},
    "1 inch PVC elbow": {
        "ask": "1 inch", "short": {"3/4 inch elbow": "3/4 inch", "CPVC elbow": "CPVC"}, "countable": True},
    "Splendor brake shoe": {
        "ask": "Splendor", "short": {"local brake shoe": "local", "Passion brake shoe": "Passion"}},
    "Activa clutch plate": {
        "ask": "Activa", "short": {"local clutch plate": "local", "Jupiter clutch plate": "Jupiter"}, "noun": "plate"},
    "12V 7Ah battery": {"ask": "7Ah", "short": {"12V 9Ah battery": "9Ah", "12V 5Ah battery": "5Ah"}, "noun": "battery"},
    "9W LED bulb": {
        "ask": "9W", "short": {"12W LED bulb": "12W", "7W LED bulb": "7W"}, "noun": "bulb", "countable": True},
    "16A MCB": {"ask": "16A", "short": {"20A MCB": "20A", "32A MCB": "32A"}, "countable": True},
    "teflon tape": {"countable": True, "noun": "tape"},
    "fan capacitor 2.5 mfd": {
        "ask": "2.5 mfd", "short": {"3.15 mfd capacitor": "3.15 mfd", "2.25 mfd capacitor": "2.25 mfd"},
        "countable": True},
    "Pulsar 150 spark plug": {
        "ask": "Pulsar 150", "short": {"local spark plug": "local", "Pulsar 180 spark plug": "Pulsar 180"},
        "noun": "plug", "countable": True},
    "wall putty 20 kg": {
        "ask": "20 kg", "short": {"wall putty 40 kg": "40 kg", "local putty": "local"}, "unit": "bag", "noun": "putty"},
    "8 mm drill bit": {
        "ask": "8 mm", "short": {"10 mm drill bit": "10 mm", "6 mm drill bit": "6 mm"}, "countable": True},
    "tank float valve": {},
    "4 inch door hinge": {"ask": "4 inch", "short": {"3 inch hinge": "3 inch"}, "countable": True, "noun": "hinge"},
    "50 mm padlock": {
        "ask": "50 mm", "short": {"40 mm padlock": "40 mm", "60 mm padlock": "60 mm"}, "noun": "padlock"},
    "2 inch paint brush": {"ask": "2 inch", "short": {"3 inch brush": "3 inch"}, "countable": True},
    "silicone sealant": {},
    "Activa air filter": {"ask": "Activa", "short": {"local air filter": "local", "Dio air filter": "Dio"}},
    "10W-30 engine oil 1 L": {
        "ask": "10W-30", "short": {"20W-40 engine oil": "20W-40", "10W-30 engine oil 900 ml": "900 ml"},
        "unit": "dabba", "noun": "oil"},
    "bike headlight bulb": {"short": {"LED headlight bulb": "LED"}, "noun": "bulb", "countable": True},
    "4 socket extension board": {
        "ask": "4 socket", "short": {"6 socket extension board": "6 socket", "3 socket board": "3 socket"}},
    "water pump seal kit": {"short": {"local seal kit": "local"}},
}

# Nouns for "battery khatam ho gayi" / "बॅटरी संपली": (Hindi roman, Hindi Devanagari, Marathi roman,
# Marathi Devanagari), each (word, gender).
NOUNS = {
    "battery": (("battery", "f"), ("बैटरी", "f"), ("battery", "f"), ("बॅटरी", "f")),
    "bulb": (("bulb", "m"), ("बल्ब", "m"), ("bulb", "m"), ("बल्ब", "m")),
    "tyre": (("tyre", "m"), ("टायर", "m"), ("tyre", "m"), ("टायर", "m")),
    "oil": (("oil", "m"), ("ऑयल", "m"), ("oil", "n"), ("ऑइल", "n")),
    "helmet": (("helmet", "m"), ("हेलमेट", "m"), ("helmet", "n"), ("हेल्मेट", "n")),
    "fan": (("pankha", "m"), ("पंखा", "m"), ("pankha", "m"), ("पंखा", "m")),
    "wire": (("wire", "f"), ("वायर", "f"), ("wire", "f"), ("वायर", "f")),
    "plate": (("plate", "f"), ("प्लेट", "f"), ("plate", "f"), ("प्लेट", "f")),
    "padlock": (("tala", "m"), ("ताला", "m"), ("kulup", "n"), ("कुलूप", "n")),
    "pump": (("motor", "f"), ("मोटर", "f"), ("motor", "f"), ("मोटर", "f")),
    "geyser": (("geyser", "m"), ("गीज़र", "m"), ("geyser", "m"), ("गिझर", "m")),
    "cover": (("cover", "m"), ("कवर", "m"), ("cover", "n"), ("कव्हर", "n")),
    "plug": (("plug", "m"), ("प्लग", "m"), ("plug", "m"), ("प्लग", "m")),
    "tape": (("tape", "f"), ("टेप", "f"), ("tape", "f"), ("टेप", "f")),
    "tube": (("tube", "f"), ("ट्यूब", "f"), ("tube", "f"), ("ट्यूब", "f")),
    "machine": (("machine", "f"), ("मशीन", "f"), ("machine", "f"), ("मशीन", "f")),
    "putty": (("putty", "f"), ("पुट्टी", "f"), ("putty", "f"), ("पुट्टी", "f")),
    "hinge": (("kabza", "m"), ("कब्ज़ा", "m"), ("bijagri", "f"), ("बिजागरी", "f")),
    "light": (("light", "f"), ("लाइट", "f"), ("light", "f"), ("लाईट", "f")),
    "plier": (("plas", "m"), ("प्लास", "m"), ("pakkad", "f"), ("पक्कड", "f")),
    "tap": (("nal", "m"), ("नल", "m"), ("nal", "m"), ("नळ", "m")),
}

# ---------------------------------------------------------------------------------------------
# Real products (data/real/products_*.jsonl): short names as a caller would ask for them.

# type, English noun, short noun for "Havells fan", noun key, regex on the source name
TYPES = [
    ("engine_oil", "engine oil", "oil", "oil", r"इंजन (ऑयल|तेल)"),
    ("helmet_lock", "helmet lock", "lock", None, r"Helmet Lock"),
    ("helmet", "helmet", "helmet", "helmet", r"हेलमेट|Motorbike Helmet"),
    ("car_mat", "car mat", "mat", None, r"Car Mat"),
    ("car_cover", "car cover", "cover", "cover", r"Car Cover|कार (बॉडी )?कवर"),
    ("bike_cover", "bike cover", "cover", "cover", r"Two [Ww]heeler Cover|बाइक बॉडी कवर"),
    ("steering_cover", "steering cover", "cover", "cover", r"Steering Cover"),
    ("handle_grip", "handle grip", "grip", None, r"Bike Handle Grip"),
    ("horn", "horn", "horn", None, r"Vehicle Horn|हॉर्न"),
    ("tyre", "tyre", "tyre", "tyre", r"Tube Tyre|Tube Less Tyre|टायर"),
    ("wiper", "wiper blade", "wiper", None, r"वाइपर ब्लेड"),
    ("chain_lube", "chain lube", "lube", None, r"चेन (लुब्रिकेंट|चिकनाई|ल्यूब)"),
    ("battery", "battery", "battery", "battery", r"\d+\s?AH .*बैटरी"),
    ("switch", "switch", "switch", None, r"Electrical Switch|Way Electrical|Switch \d+ (One|Two) Way"),
    ("mcb", "MCB", "MCB", None, r"Tripper .*MCB"),
    ("wire", "wire", "wire", "wire", r"sq/mm .* Wire$"),
    ("ceiling_fan", "ceiling fan", "fan", "fan", r"सीलिंग फैन"),
    ("exhaust_fan", "exhaust fan", "fan", "fan", r"एग्जॉस्ट फैन"),
    ("pedestal_fan", "pedestal fan", "fan", "fan", r"पेडस्टल फैन"),
    ("immersion_rod", "immersion rod", "rod", None, r"इमर्शन|इमर्सन|इमरेशन"),
    ("geyser", "geyser", "geyser", "geyser", r"गीजर|गीज़र"),
    ("room_heater", "room heater", "heater", None, r"रूम हीटर"),
    ("led_bulb", "LED bulb", "bulb", "bulb", r"LED बल्ब|एलईडी बल्ब"),
    ("batten", "LED batten", "batten", "tube", r"बैटन"),
    ("torch", "LED torch", "torch", None, r"LED टॉर्च"),
    ("multimeter", "multimeter", "multimeter", None, r"मल्टीमीटर"),
    ("soldering_iron", "soldering iron", "iron", None, r"सोल्डरिंग आयरन"),
    ("angle_grinder", "angle grinder", "grinder", "machine", r"एंगल ग्राइंडर"),
    ("drill", "drill machine", "drill", "machine", r"(हैमर|इम्पैक्ट) ड्रिल|ड्रिल मशीन"),
    ("plier", "plier", "plier", "plier", r"प्लायर"),
    ("sealant", "silicone sealant", "sealant", None, r"सिलिकॉन सीलेंट"),
    ("welding", "welding machine", "machine", "machine", r"वेल्डिंग मशीन"),
    ("water_pump", "water pump", "pump", "pump", r"(वॉटर|वाटर) (मोटर )?पंप|मोनोब्लॉक पंप|प्रेशर पंप"),
    ("cutter", "cutter", "cutter", None, r"(Glass|Bolt|Pipe) Cutter"),
    ("bit_set", "screwdriver bit set", "bit set", None, r"Screwdriver Bit Set"),
    ("tap", "tap", "tap", "tap", r"Faucet Set"),
    ("headlight", "LED headlight", "headlight", "light", r"LED Headlight For|हेडलाइट बल्ब"),
]
# Things a hardware, electrical or auto-parts shop would not sell, or names that are not one product.
NOT_SHOP = (r"(?i)Lock For Helmet|हेलमेट (लाइट|कैप|होल्डर|कवर)|क्रिकेट|स्पार्टन|टैक्टिकल|इन्फ्लेटर|इंफ्लेटर|इंफ़लेटर"
            r"|क्लीनिंग|फाउंटेन|एक्वारियम|कूलर|फ्यूल|ईंधन|साइफन|Combo|हैंगर|Holder|फ्लश|Mini Trip|बच्च|Exchange"
            r"|Amazon|स्मार्ट|Wi-Fi|किट|\bKit\b|रिमोट|\b50 ?ml\b|सेल्फ प्राइमिंग .*मिनी|Mini")
# Brands that are not a brand ("Generic"), or a brand matched to the wrong kind of product.
NOT_BRAND = {"generic", "easy", "auto", "care", "silicone", "scrub", "activa", "amazon", "electronic", "robotbanao",
             "art", "lovely", "tusa", "no"}
NOT_BRAND_FOR = {("tyre", "stanley"), ("helmet", "allextreme")}
# Online-only prices that a shop would not quote for this kind of product: below MIN_PRICE it is an accessory or
# a spare part, above MAX_PRICE a pack or an import. MAX_PRICE is for one piece; a "(pack of n)" may cost n times it.
MAX_PRICE = {"helmet": 6000, "led_bulb": 400, "headlight": 5000, "wiper": 2000, "switch": 2000, "car_cover": 8000,
             "multimeter": 10000, "room_heater": 10000, "tap": 5000, "car_mat": 6000, "angle_grinder": 8000,
             "batten": 1200, "plier": 1500, "engine_oil": 2500, "drill": 10000}
MIN_PRICE = {"water_pump": 1000, "helmet": 400, "angle_grinder": 900, "tyre": 700, "pedestal_fan": 1000, "drill": 900}
PER_TYPE = 9
GENERIC = {
    "by", "panasonic", "electric", "isi", "certified", "full", "face", "open", "half", "helmet", "with", "for", "the",
    "and", "new", "premium", "heavy", "duty", "universal", "professional", "original", "high", "speed", "bike",
    "car", "led", "stylish", "graphic", "in", "matt", "of", "set", "pack", "black", "white", "red", "blue", "&",
    "-", "tyres", "tyre", "india", "home", "air", "technology", "tools", "power", "water", "motor", "pump", "fan",
    "lcd", "device", "kit", "mini",
}
# Amazon brands of more than one word (the first Latin words of the title)
MULTI_WORD_BRANDS = ["Pick Ur Needs", "Golden Bullet", "Liqui Moly", "Crompton Greaves", "Morphy Richards",
                     "Grand Pitstop", "Royal Enfield", "Euro Care", "Sage Square", "Duro Gold", "AO Smith", "V-Guard"]
BRAND_FIX = {"USHA": "Usha", "Rk": "RK", "amiciTools": "AmiciTools", "amiciVision": "AmiciVision"}
VEHICLES = [
    "Datsun Go", "Grand i10", "Pulsar 150", "Classic 350", "XUV 500", "Santro Xing", "WagonR", "Alto", "Swift",
    "Dzire", "Baleno", "Ciaz", "Ertiga", "i10", "i20", "Creta", "Verna", "Santro", "Xcent", "City", "Amaze", "Jazz",
    "Innova", "Etios", "Fortuner", "Polo", "Vento", "Scorpio", "Bolero", "Ecosport", "Endeavour", "Nexon", "Tiago",
    "Zest", "Indica", "Kwid", "Duster", "Beat", "Spark", "Sail", "Cruze", "Micra", "Ritz", "Activa", "Splendor",
    "Pulsar", "Apache", "Access", "Jupiter", "Bullet", "Duke", "R15", "Shine", "Passion", "Dio", "Unicorn",
    "Avenger", "Platina", "Discover", "CT100", "YBR", "Optra", "Tavera", "Innova Crysta", "Sumo", "Safari",
    "Hero", "Honda", "Bajaj", "TVS", "Yamaha", "Mahindra",
]
SPECS = [  # (regex, how it is written in the product name)
    (r"\b(\d{1,2})W-?(\d{2})\b", lambda m: f"{m[1]}W-{m[2]}"),
    (r"\b(\d{2,3}/\d{2,3}-\d{2}|\d\.\d{2}-\d{2})\b", lambda m: m[1]),
    (r"(\d+(?:\.\d+)?) ?sq ?/?mm", lambda m: f"{m[1]} sq mm"),
    (r"(\d+(?:\.\d+)?) ?(?:HP|hp)\b", lambda m: f"{m[1]} HP"),
    (r"\b(\d{1,3}) ?(?:AH|Ah)\b", lambda m: f"{m[1]}Ah"),
    (r"\b(\d{3,4}) ?(?:mm|MM|मिमी)", lambda m: f"{m[1]} mm"),
    (r"\b(\d{1,4}) ?-? ?(?:W|w|Watt|Watts|वॉट|वाट)(?![A-Za-z0-9])", lambda m: f"{m[1]}W"),
    (r"\b(\d{1,2}(?:\.\d)?) ?-? ?(?:L|लीटर)(?![A-Za-z0-9])", lambda m: f"{m[1]} L"),
    (r"\b(\d{2,4}) ?(?:ml|ML)\b", lambda m: f"{m[1]} ml"),
    (r"\b(\d{1,2}) ?-? ?(?:इंच)", lambda m: f"{m[1]} inch"),
    (r"\b(\d{1,3}) ?A\b", lambda m: f"{m[1]}A"),
]
SPEC_FOR = {  # which specs go into the name, by type
    "engine_oil": (0, 7, 8), "tyre": (1,), "wire": (2,), "water_pump": (3,), "battery": (4,), "ceiling_fan": (5,),
    "exhaust_fan": (5,), "pedestal_fan": (5,), "immersion_rod": (6,), "geyser": (7,), "room_heater": (6,),
    "led_bulb": (6,), "batten": (6,), "soldering_iron": (6,), "drill": (6,), "chain_lube": (8,), "plier": (9,),
    "switch": (10,), "welding": (10,),
}
HELMET_KIND = [(r"फ़?फुल फ़?फेस|फ़ुल फ़ेस|Full Face", "full face"), (r"ओपन फ़?फेस|Open Face", "open face"),
               (r"हाफ फ़?फेस|Half Face", "half face"), (r"फ्लिप", "flip-up")]
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
DEV_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def latin_head(name: str) -> list[str]:
    """The Latin words a title starts with, before the first Hindi word (Amazon titles)."""
    out = []
    for tok in re.split(r"[\s|,()]+", name):
        if not tok:
            continue
        if DEVANAGARI.search(tok):
            break
        out.append(tok)
    return out


def tidy_brand(word: str) -> str:
    word = BRAND_FIX.get(word, word).replace("®", "").replace("™", "")
    if word.isupper() and len(word) > 4 and word.isalpha():
        return word.title()
    if word[:1].islower() and word[1:2].isupper():  # iBELL
        return word
    return word[:1].upper() + word[1:]


def model_words(tokens: list[str], brand: str, max_words: int = 2) -> list[str]:
    out = []
    for tok in tokens:
        tok = tok.replace("®", "").replace("™", "")
        if (tok.lower() in GENERIC or len(tok) > 8 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]*", tok)
                or not re.search(r"[A-Za-z]", tok)):
            break
        if re.search(r"\d", tok) and re.search(r"[A-Za-z]", tok) and len(tok) >= 6:
            break  # a part number such as F002H10028, not something a caller says
        if re.fullmatch(r"\d+(\.\d+)?", tok) or re.search(r"(?i)\d+-?(w|watt|mm|l|ml|hp|ah|a|v)$", tok):
            break  # a bare number or a spec, not a model name
        if any(re.fullmatch(rx, tok) for rx, _ in SPECS) or tok.lower() in brand.lower().split():
            continue
        out.append(tok.title() if tok.isupper() and len(tok) > 4 else tok)
        if len(out) == max_words:
            break
    return out


def find_vehicle(name: str) -> str | None:
    for v in sorted(VEHICLES, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(v)}(?![A-Za-z0-9])", name, re.IGNORECASE):
            return v
    return None


def find_pack(name: str) -> int | None:
    m = re.search(r"([0-9०-९]+) का (?:पैक|सेट)|[Pp]ack [Oo]f (\d+)|[Ss]et [Oo]f (\d+)", name)
    if not m:
        return None
    digits = next(g for g in m.groups() if g)
    n = int(digits.translate(str.maketrans("०१२३४५६७८९", "0123456789")))
    return n if n > 1 else None


def real_product_name(kind: str, noun: str, brand: str, rest: list[str], full: str) -> str | None:
    specs = []
    for i in SPEC_FOR.get(kind, ()):
        rx, fmt = SPECS[i]
        m = re.search(rx, full)
        if m and fmt(m) not in specs:
            specs.append(fmt(m))
    words = [brand]
    if kind not in ("car_mat", "car_cover", "bike_cover", "steering_cover", "handle_grip", "headlight", "cutter",
                    "helmet_lock", "tap", "bit_set"):
        words += model_words(rest, brand, 1 if kind == "tyre" else 2)
    if kind == "helmet":
        kinds = [k for rx, k in HELMET_KIND if re.search(rx, full)]
        words += kinds[:1]
    if kind == "cutter":
        noun = re.search(r"(Glass|Bolt|Pipe) Cutter", full)[0].lower()
    words += specs + [noun]
    if kind in ("car_mat", "car_cover", "bike_cover", "steering_cover", "handle_grip", "headlight"):
        vehicle = find_vehicle(full)
        if vehicle:
            words += ["for", vehicle]
    name = " ".join(words)
    pack = find_pack(full)
    if pack:
        name += f" (pack of {pack})"
    if len(name.split()) > 8 or not re.match(r"[A-Za-z0-9]", name):
        return None
    return name


def load_real_products() -> list[dict]:
    """Shop products from the Flipkart and Amazon India lists, at most PER_TYPE of each kind."""
    by_type: dict[str, list[dict]] = {}
    seen = set()
    for fname in ("products_flipkart.jsonl", "products_amazon_in.jsonl"):
        for line in (REAL / fname).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            full = nfc(r["name"])
            if re.search(NOT_SHOP, full) or not (30 <= r["price"] <= 20000):
                continue
            hit = next((t for t in TYPES if re.search(t[4], full)), None)
            if hit is None:
                continue
            kind, noun, short_noun, noun_key, _ = hit
            if fname.startswith("products_flipkart"):
                brand_words = r["brand"].split()
                tokens = full.split()
                while tokens and brand_words and tokens[0].lower() == brand_words[0].lower():
                    tokens = tokens[1:]
                    if len(brand_words) > 1 and tokens and tokens[0].lower() == brand_words[1].lower():
                        tokens = tokens[1:]
                    if tokens and tokens[0].lower() == brand_words[0].lower():
                        continue
                    break
                brand = " ".join(tidy_brand(w) for w in brand_words[:2])
                rest = [t for t in tokens if t != "-"]
            else:
                head = latin_head(full)
                if not head or head[0].lower() in GENERIC or not head[0][0].isalpha():
                    continue
                size = next((len(b.split()) for b in MULTI_WORD_BRANDS
                             if " ".join(head[:len(b.split())]).lower() == b.lower()), 1)
                brand = " ".join(tidy_brand(w) for w in head[:size])
                rest = head[size:]
                if rest and rest[0] in ("Electric", "by"):
                    rest = rest[3:] if rest[0] == "by" else rest[1:]
            first = brand.split()[0]
            pack = find_pack(full)
            if (first.lower() in NOT_BRAND or sum(c.isalpha() for c in first) < 2 or sum(c.isdigit() for c in first) > 2
                    or (len(first) <= 2 and not first.isupper()) or (kind, brand.split()[0].lower()) in NOT_BRAND_FOR
                    or not MIN_PRICE.get(kind, 30) <= r["price"] <= MAX_PRICE.get(kind, 20000) * (pack or 1)):
                continue
            name = real_product_name(kind, noun, brand, rest, full)
            if name is None or name.lower() in seen:
                continue
            seen.add(name.lower())
            if kind == "cutter":  # a glass cutter is no alternative to a bolt cutter: the subtype is the kind
                sub = re.search(r"(Glass|Bolt|Pipe) Cutter", full)[1].lower()
                kind, short_noun = f"{sub}_cutter", f"{sub} cutter"
            mrp = r.get("mrp")
            pair = mrp is not None and mrp > r["price"] and 0.55 <= r["price"] / mrp <= 0.95
            by_type.setdefault(kind, []).append({
                "name": name, "brand": brand, "kind": kind, "short_noun": short_noun, "noun": noun_key,
                "price": r["price"],
                "mrp": mrp if pair else None, "source": fname.split("_")[1].split(".")[0], "pack": pack,
            })
    out = []
    for kind in sorted(by_type):
        rows = by_type[kind]
        # Spread over brands: take one of each brand in turn.
        brands: dict[str, list[dict]] = {}
        for r in rows:
            brands.setdefault(r["brand"], []).append(r)
        picked, i = [], 0
        while len(picked) < PER_TYPE and any(len(v) > i for v in brands.values()):
            for v in brands.values():
                if len(v) > i and len(picked) < PER_TYPE:
                    picked.append(v[i])
            i += 1
        out += picked
    return out


# ---------------------------------------------------------------------------------------------
# Numbers said in words (data/real/price_words.json) and Latin spellings
# (data/real/romanization_variants.json).


def hundreds_prices() -> list[int]:
    """Prices from 2100 to 4550 that are said as "N hundred": 25 सौ, 25 सौ 50, but not 30 सौ."""
    return [h * 100 + r for h in range(21, 46) if h % 10 for r in (0, 50)]


def on_grid(n: int) -> int:
    """The nearest price that price_words.json has words for."""
    if n <= 100:
        return max(n, 1)
    step = 5 if n <= 1000 else 50 if n <= 5000 else 100
    return min(20000, round(n / step) * step)


# Romanized number spellings in price_words.json that are not words ("nauu" for नऊ).
BAD_ROMAN = re.compile(r"\bnauu\b")
# Hundreds style above 2000 (पच्चीस सौ, पंचवीसशे, "pachis sau pachas"): price_words.json has it only for
# 1100-1999, so it is built here by the same rule from the formal words for 21-45 (not 30 or 40: those are
# said in thousands). Higher prices are said in thousands (साढ़े पाँच हज़ार).
HUNDREDS_JOIN = {"hi": (" सौ", " sau"), "mr": ("शे", "she")}


class Numbers:
    def __init__(self, data: dict) -> None:
        self.forms: dict[int, dict] = {}
        for entry in data["prices"]:
            forms = {}
            for lang in ("hi", "mr"):
                for kind in ("formal", "hundreds_style", "colloquial"):
                    for f in entry[lang].get(kind, []):
                        dev = [nfc(f["text"])] + [nfc(t) for t in f.get("spacing_variants", [])]
                        roman = [f["roman"]] + [r for r in f.get("roman_variants", []) if not BAD_ROMAN.search(r)]
                        forms.setdefault((lang, kind), []).append(
                            {"dev": dev, "roman": roman, "primary": f.get("primary")})
            self.forms[entry["price"]] = forms
        for n in hundreds_prices():
            h, r = divmod(n, 100)
            for lang, (dev_join, roman_join) in HUNDREDS_JOIN.items():
                head = self.primary(h, lang)
                tail = self.primary(r, lang) if r else {"dev": [""], "roman": [""]}
                dev = (head["dev"][0] + dev_join + " " + tail["dev"][0]).strip()
                roman = [(x + roman_join + " " + tail["roman"][0]).strip() for x in head["roman"]]
                self.forms[n].setdefault((lang, "hundreds_style"), []).append(
                    {"dev": [dev], "roman": roman, "primary": True})
        self.tokens = {t for forms in self.forms.values() for options in forms.values() for o in options
                       for text in o["dev"] + o["roman"] for t in text.split()}

    def primary(self, n: int, lang: str) -> dict:
        return next(o for o in self.forms[n][(lang, "formal")] if o["primary"])

    def has(self, n: int, lang: str, kind: str) -> bool:
        return (lang, kind) in self.forms.get(n, {})

    def words(self, rng: random.Random, n: int, lang: str, kind: str, script: str) -> str | None:
        options = self.forms.get(n, {}).get((lang, kind))
        if not options:
            return None
        primary = [o for o in options if o["primary"]]
        f = primary[0] if primary and (rng.random() < 0.8 or len(options) == 1) else rng.choice(options)
        if script == "dev":
            return f["dev"][0] if len(f["dev"]) == 1 or rng.random() < 0.6 else rng.choice(f["dev"][1:])
        if len(f["roman"]) == 1 or rng.random() < 0.6:
            return f["roman"][0]
        return rng.choice(f["roman"][1:])


# Spellings that are also common English words, or another Hindi word's spelling: never replaced.
NOT_KEYS = {"to", "me", "per", "a", "bus", "do", "men", "main", "pr", "p", "ha", "h", "hn", "n", "b", "bi", "so",
            "bar", "dar", "tas", "mal", "ya", "ja", "le", "aa", "pe", "par", "ka", "se", "ji", "ho", "na", "bas"}


# Spellings that are never written into a reply. The replies stand for speech-to-text output, so chat
# shorthand is left out ("h" for hai, "hlo", "bs"), and so are spellings that are not words ("yeiel", "deuu").
BAD_SPELLINGS = {"hlo", "bus", "dege", "duga", "sast", "jald", "subh", "aakr", "deuu", "yeiel", "yeila", "yeiil",
                 "gheuoon", "dusrn", "dusaram", "adicha", "bhauu", "hoya", "dekhiae", "aaiae", "aaeye", "packeet",
                 "pakeet", "pyackeet", "pyacket", "pyakeet", "taaee", "taiee", "taii", "tae", "taee"}
VOWELS = re.compile(r"[aeiou]")


def chat_spelling(word: str, others: list[str]) -> bool:
    """Chat shorthand that leaves out a spoken vowel: no vowel at all ("h", "kl", "khtm", "ghr"), or the vowel
    of the first syllable ("nhi" for nahi, "gya" for gaya, "btaiye"). Dropping an unspoken vowel later in the
    word is normal spelling and stays ("sampla" / "sampala", "bhejta" / "bhejata")."""
    if not VOWELS.search(word):
        return True
    onset = len(VOWELS.split(word)[0])
    return any(VOWELS.sub("", o) == VOWELS.sub("", word) and len(VOWELS.split(o)[0]) < onset for o in others)


def spoken(spelling: str, choices: list[str]) -> bool:
    """A spelling a speech-to-text system could write: a word, not chat shorthand or a non-word."""
    words = spelling.split()
    others = [o.split() for o in choices if o != spelling and len(o.split()) == len(words)]
    return not any(w in BAD_SPELLINGS or chat_spelling(w, [o[i] for o in others]) for i, w in enumerate(words))


class Spellings:
    """Replace a word with another attested Latin spelling of it, by its weight."""

    def __init__(self, table: dict) -> None:
        self.maps: dict[str, dict] = {"hi": {}, "mr": {}}
        best: dict[tuple, float] = {}
        for entry in table.values():
            attested = [(w["roman"], w["weight"]) for w in entry["weights"] if w["weight"] >= 0.05]
            choices = [(r, w) for r, w in attested if spoken(r, [c for c, _ in attested])]
            langs = ["hi", "mr"] if entry["lang"] == "hi+mr" else [entry["lang"]]
            # A spelling belongs to the word it is most often a spelling of ("hai" to है, not to हैं), even when
            # that word has only one spoken spelling left and so is never changed.
            for spelling, weight in attested:
                if weight < 0.12 or spelling in NOT_KEYS:
                    continue
                for lang in langs:
                    if weight > best.get((lang, spelling), 0):
                        best[(lang, spelling)] = weight
                        self.maps[lang][spelling] = choices if len(choices) >= 2 else None
        self.maps = {lang: {k: v for k, v in m.items() if v} for lang, m in self.maps.items()}
        self.longest = {lang: max(len(k.split()) for k in m) for lang, m in self.maps.items()}

    def vary(self, rng: random.Random, text: str, lang: str, level: str) -> str:
        if level == "clean" or not text.strip():
            return text
        table = self.maps[lang]
        words = list(re.finditer(r"[A-Za-z]+", text))
        out, pos, i = [], 0, 0
        while i < len(words):
            done = False
            for size in range(self.longest[lang], 0, -1):
                group = words[i:i + size]
                if len(group) < size or any(text[a.end():b.start()] != " " for a, b in itertools.pairwise(group)):
                    continue
                key = " ".join(m[0] for m in group)
                if key.lower() not in table or (len(key) > 1 and key[1].isupper()):
                    continue
                choices = table[key.lower()]
                weights = [w if level == "normal" else w ** 0.5 for _, w in choices]
                new = rng.choices([s for s, _ in choices], weights=weights)[0]
                if key[0].isupper():
                    new = new[0].upper() + new[1:]
                out.append(text[pos:group[0].start()] + new)
                pos, i, done = group[-1].end(), i + size, True
                break
            if not done:
                i += 1
        out.append(text[pos:])
        return "".join(out)


# ---------------------------------------------------------------------------------------------
# What shopkeepers say. {p} = price of the asked product; {m} MRP; {old} old price; {bulk} box price;
# {bn} pieces in the box; {bu} per-piece price when buying many; {unit} price of one loose piece (when the
# asked product is a pack); {fee} delivery fee; {q} quantity; {u} unit word; {h} hour; {alt} the
# alternative; {ap} its price; {ask} the asked variant; {w} a wrong price that is corrected; {x} a price
# that does not count (not in stock / not sure); {noun} the product's noun. "{p}_ला" joins ला to a price
# said in words. A slot ending in "_rev" says the same numbers in the other order (the price of the asked
# product first), so the model cannot learn the price from where it stands. yes_m / yes_f / yes_n say the
# noun in an "in stock" reply as often as no_m / no_f / no_n say it in a "not in stock" one. Every "tomorrow"
# reply says it is not there now.

BANKS: dict[str, dict[str, list[str]]] = {}

BANKS["hi_rom"] = {
    "open": ["haan bolo", "haan ji, boliye", "hello, haan bolo", "haan bhai, bolo", "ji boliye", "haan haan, bolo",
             "hello? haan, awaaz aa rahi hai, bolo", "ek minute... haan bolo", "haan bhaiya, bataiye",
             "namaste ji, boliye", "haan, kaun? acha, bolo", "haan sir, boliye"],
    "yes": ["haan hai", "haan ji, mil jayega", "hai bhai", "available hai", "haan, stock mein hai",
            "ji haan, rakha hai", "haan, hai apne paas", "hai, bilkul hai", "haan haan, hai", "mil jayega, hai",
            "ji, available hai", "haan, wahi wala hai", "hai na, kitne chahiye?", "haan, maal pada hai",
            "haan, mil jayega bhai", "hai ji, mil jayega"],
    "yes_m": ["{noun} hai, mil jayega", "haan, {noun} hai", "{noun} pada hai", "{noun} rakha hai, mil jayega",
              "{noun} khatam nahi hua, hai", "haan ji, {noun} available hai"],
    "yes_f": ["{noun} hai, mil jayegi", "haan, {noun} hai", "{noun} padi hai", "{noun} rakhi hai, mil jayegi",
              "{noun} khatam nahi hui, hai", "haan ji, {noun} available hai"],
    "yes_qty": ["haan, {q} {u} pade hain", "hai, abhi {q} {u} bache hain", "bas {q} {u} bache hain",
                "haan, stock mein {q} {u} hai", "hai, {q} {u} rakhe hain"],
    "no": ["nahi hai", "khatam ho gaya", "abhi stock mein nahi hai", "nahi milega bhai", "woh hum nahi rakhte",
           "abhi nahi hai", "maal khatam hai", "nahi hai, sorry", "stock khatam ho gaya hai",
           "wo toh kal hi khatam ho gaya", "nahi bhai, yahan nahi milega", "nahi hai ji, khatam",
           "wo item nahi rakhte hum", "abhi khatam hai bhai"],
    "no_m": ["{noun} khatam ho gaya", "{noun} nahi hai, khatam ho gaya"],
    "no_f": ["{noun} khatam ho gayi", "{noun} nahi hai, khatam ho gayi"],
    "tomorrow": ["aaj nahi hai, kal aayega", "abhi nahi hai, parso tak aa jayega", "maal kal aayega, abhi nahi hai",
                 "abhi nahi hai, order diya hai, do din mein aa jayega", "abhi khatam hai, shaam tak aa jayega",
                 "kal subah tak aa jayega, abhi nahi hai", "abhi stock mein nahi hai, naya maal Monday ko aayega"],
    "order": ["nahi hai, mangwa dunga, do din lagenge", "abhi nahi, order pe mangwa denge",
              "nahi hai, bolo toh mangwa deta hun", "humare paas nahi hota, mangwana padega"],
    "unclear": ["ruko, dekh ke batata hun", "malik abhi nahi hain, baad mein phone karo",
                "pata nahi, godown mein dekhna padega", "thodi der baad call karo",
                "abhi busy hun, aadhe ghante baad phone karna", "seth ji bahar gaye hain, wahi batayenge",
                "check karke batata hun", "dekhna padega bhai, yaad nahi hai",
                "hello? kuch sunai nahi de raha, baad mein karo",
                "abhi dukaan band kar raha hun, kal subah phone karo", "ladka dekh ke batayega, ruko zara"],
    "unclear_n": ["{noun} hai ki nahi, dekh ke batata hun", "{noun} ka pata nahi, godown mein dekhna padega",
                  "{noun} ka stock check karke batata hun"],
    "price": ["{p} ka hai", "{p} ka padega", "rate {p} hai", "{p} lagega", "{p} rupaye", "{p} rupaye ka hai",
              "{p} mein de dunga", "{p} final", "{p} ka milega", "aapke liye {p}", "{p} last rate hai",
              "{p} ka hai, isse kam nahi hoga", "{p} hoga", "{p} rupaye lagenge"],
    "mrp": ["MRP {m} hai, aapko {p} mein de denge", "MRP {m} likha hai, {p} laga denge",
            "{m} MRP hai, {p} mein ho jayega", "MRP toh {m} hai, apna rate {p}", "upar {m} likha hai, aapse {p} lenge",
            "MRP {m} hai, par {p} de dena"],
    "mrp_rev": ["apna rate {p}, MRP toh {m} hai", "aapko {p} mein de denge, MRP {m} hai",
                "{p} laga denge, waise MRP {m} likha hai", "{p} mein ho jayega, upar MRP {m} likha hai"],
    "old_new": ["pehle {old} tha, ab {p} ho gaya", "rate badh gaya, {old} se {p} ho gaya",
                "last time {old} tha, abhi {p} hai", "pichle mahine {old} tha, ab {p}"],
    "old_down": ["pehle {old} tha, ab {p} ho gaya, sasta hua hai", "rate gira hai, {old} tha, ab {p}"],
    "old_rev": ["ab {p} hai, pehle {old} tha", "company ne rate badla, ab {p} hai, pehle {old} tha",
                "abhi rate {p} ho gaya, pichle mahine {old} tha", "{p} lagega ab, pehle {old} ka tha"],
    "unit_bulk": ["ek ka {p}, {bn} ka dabba {bulk} ka", "{p} ka ek, {bn} loge toh {bu} ke hisaab se laga dunga",
                  "ek piece {p}, {bn} wala box loge toh {bulk}", "khula {p} ka, {bn} ka poora box {bulk} mein"],
    "unit_bulk_rev": ["{bn} ka dabba {bulk} ka, ek piece {p}", "{bn} loge toh {bu} ke hisaab se, ek ka {p}",
                      "{bn} piece wala box {bulk} mein, khula ek {p} ka"],
    "pack": ["packet {p} ka hai, khula ek {unit} ka", "poora packet {p}, alag se ek piece {unit} ka",
             "{p} ka packet hai, khule bhi hain {unit} wale"],
    "pack_rev": ["khula ek {unit} ka, poora packet {p}", "ek piece {unit} ka milta hai, packet {p} ka",
                 "single {unit} ka, packet lo toh {p}"],
    "gst": ["{p} plus GST", "{p} hai, GST alag", "{p} plus 18 percent GST", "GST mila ke {p}",
            "GST extra lagega, rate {p}"],
    "fee": ["bhej denge, {fee} rupaye delivery charge lagega", "delivery ka {fee} extra",
            "ghar pe bhej dunga, {fee} lagega bhejne ka", "ladka de aayega, {fee} rupaye auto ka"],
    "item_price": ["maal ka {p}, delivery alag", "item {p} ka hai, delivery charge alag", "saman ka {p} lagega",
                   "{p} ka hai, plus delivery", "uska rate {p}, bhejne ka alag"],
    "timing": ["dukaan {h} baje tak khuli hai", "{h} baje tak main yahin hun", "{h} baje dukaan band hoti hai"],
    "alt": ["{alt} hai, chalega?", "uski jagah {alt} de sakte hain", "{alt} rakha hai", "{alt} hai, wo le lo",
            "{alt} chalega toh bolo", "{alt} hai mere paas", "chahiye toh {alt} de dunga"],
    "alt_neg": ["{ask} nahi hai, {alt} hai", "{ask} wala khatam hai, {alt} hai, chalega?"],
    "alt_price": ["{alt} hai, {ap} ka", "{alt} de sakta hun, wo {ap} ka padega", "{alt} chalega? {ap} ka hai",
                  "{alt} hai, uska rate {ap}"],
    "other": ["{alt} bhi hai, wo {ap} ka", "{alt} chahiye toh wo bhi hai, {ap} wala"],
    "other_first": ["{alt} bhi hai, uska {ap}, aur aap wala {p} ka", "{alt} ka rate {ap} hai, ye wala {p}",
                    "{alt} wala {ap} mein aata hai, aapka wala {p}"],
    "del_yes": ["ghar pe bhej denge", "delivery ho jayegi", "ladka bhej deta hun", "home delivery hai",
                "bhijwa dunga", "shaam tak pahuncha denge", "address bhej do, bhej dete hain", "delivery free hai"],
    "del_no": ["delivery nahi karte, aake le jao", "dukaan pe aake lena padega", "delivery nahi hai",
               "bhejne wala koi nahi hai, aapko aana padega", "home delivery nahi hai bhai", "khud aake le jaiye"],
    "close": ["aur kuch chahiye?", "bolo kya karna hai", "thik hai?", "bata dena", "aur kuch?"],
    "noprice": ["rate abhi pata nahi, malik se puchna padega", "rate seth ji batayenge", "rate puch ke batata hun"],
    "wrong": ["{w}... nahi nahi, {p} hai", "{w} hai, sorry sorry, {p}",
              "{w}... ek minute, rate list dekhta hun... haan {p}", "{w} bola kya? galti se bola, {p} hai"],
    "wrong_rev": ["{p} hai, {w} nahi", "{p} lagega, {w} wala rate purana hai, wo nahi", "{p}, {w} nahi bola maine"],
    "corr_yes": ["nahi hai... ruko ruko, hai, ek pada hai", "khatam ho gaya tha... nahi, ek bacha hai",
                 "lagta hai nahi hai... haan mil gaya, hai"],
    "corr_no": ["haan hai... ek minute, nahi, wo kal khatam ho gaya",
                "hai... arre nahi, wo dusra wala hai, ye wala nahi hai",
                "hai hai... nahi yaar, last piece subah bik gaya"],
    "trap_yes": ["nahi nahi, hai na", "aisa nahi hai ki nahi hai, hai", "khatam nahi hua, abhi bhi hai",
                 "mana kisne kiya? hai na", "kam nahi hai, bahut pada hai", "kyun nahi milega, hai"],
    "trap_no": ["haan haan, samajh gaya, lekin wo abhi nahi hai", "haan bolo... nahi, wo nahi hai",
                "haan rakhte hain, par abhi khatam hai"],
    "trap_del_yes": ["delivery ki tension mat lo, bhej denge", "bhejne mein koi problem nahi",
                     "delivery nahi hoti aisa nahi hai, bhej dunga"],
    "trap_del_no": ["bhej deta, par aaj ladka nahi aaya, aake le jao", "delivery? nahi bhai, wo nahi hota"],
    "nostock_price": ["rate {x} hai, par abhi stock nahi hai", "{x} ka aata hai, lekin abhi khatam hai"],
    "unclear_price": ["pehle {x} ka tha, abhi hai ki nahi dekhna padega",
                      "rate {x} ke aas paas hoga, stock malik se puchna padega"],
    "chatter": ["arre kya bataun, subah se bahut rush hai", "ladka chai lene gaya hai", "aaj subah se light nahi hai",
                "market mein maal ki kami chal rahi hai", "company ne is mahine phir rate badhaya",
                "Diwali ka season hai na, sab yahi maang rahe hain", "ek customer khada hai, ek minute",
                "awaaz kat rahi hai, zor se bolo", "haan haan, sun raha hun", "kal Sunday tha toh dukaan band thi",
                "aapko number kisne diya? acha thik hai", "pichle hafte bhi kisi ne yahi poocha tha",
                "bhaiya aap ruko do minute, phone pe hun"],
    "filler": ["matlab", "dekho", "acha", "haan toh", "kya hai na", "umm", "arre"],
    "address": ["bhai", "ji", "sir", "bhaiya", "boss"],
}

BANKS["mr_rom"] = {
    "open": ["haan bola", "ho bola", "hello, bola na", "bola dada", "haan bola, kay pahije?", "namaskar, bola",
             "ek minute ha... haan bola", "hello? aawaj yetoy ka? haan bola", "bola saheb"],
    "yes": ["ho aahe", "aahe na", "ho, milel", "stock madhe aahe", "haan aahe", "aahe aahe", "ho, aahe aamchyakade",
            "milel na, aahe", "ho dada, aahe", "aahe ki, milel", "ho, thevto aamhi, aahe"],
    "yes_m": ["{noun} aahe, milel", "ho, {noun} aahe", "{noun} padlay", "{noun} sampla nahi, aahe", "ho, {noun} milel"],
    "yes_f": ["{noun} aahe, milel", "ho, {noun} aahe", "{noun} padliye", "{noun} sampli nahi, aahe",
              "ho, {noun} milel"],
    "yes_n": ["{noun} aahe, milel", "ho, {noun} aahe", "{noun} padlay", "{noun} sampla nahi, aahe", "ho, {noun} milel"],
    "yes_qty": ["ho, {q} {u} aahet", "aahe, {q}ch urlet", "{q} nag shillak aahet", "aahe, ajun {q} {u} padlet"],
    "no": ["nahi aahe", "sampla", "sadhya nahi", "aamhi te thevat nahi", "nahiye", "stock samplay", "maal sampla",
           "te kalach sampla", "nahi milnar", "shillak nahi", "aatta nahiye", "nahi ho, sampla te",
           "aamchyakade nasta te", "samplay, udya dupari yeil"],
    "no_m": ["{noun} sampla", "{noun} samplay"],
    "no_f": ["{noun} sampli", "{noun} sampliye", "ti sampli ho"],
    "no_n": ["{noun} sampla", "{noun} nahiye, sampla"],
    "tomorrow": ["aaj nahi, udya yeil", "udya sakali maal yeil, aatta nahiye", "aatta nahiye, parva paryant yeil",
                 "sadhya nahiye, order dilay, don divsat yeil", "aatta nahiye, sandhyakali yeil"],
    "order": ["nahiye, magvun deto, don divas lagtil", "sadhya nahi, magvava lagel",
              "aamchyakade nasta, order karun magvto"],
    "unclear": ["thamba, baghun sangto", "malak nahit, nantar phone kara", "mahit nahi, godown madhe baghava lagel",
                "jara velane phone kara", "aatta gadbadit aahe, ardhya tasane phone kara",
                "sheth baher gelet, te aale ki sangto", "baghto aani kalavto",
                "hello? aawaj yet nahiye, nantar kara", "aatta sangta yenar nahi, sandhyakali phone kara"],
    "unclear_n": ["{noun} aahe ka baghava lagel, nantar sangto", "{noun} cha stock baghun sangto",
                  "{noun} aahe ki nahi mahit nahi, godown madhe baghto"],
    "price": ["{p} rupaye", "{p} la padel", "rate {p} aahe", "{p} rupayanna padel", "{p} la deto", "{p} fix",
              "tumhala {p} la deto", "{p} hotil", "{p} rupaye hotil", "{p} rupaye, kami nahi honar"],
    "mrp": ["MRP {m} aahe, tumhala {p} la deto", "{m} MRP aahe, pan {p} la deu",
            "chhapil kimmat {m}, aamhi {p} la deto", "MRP {m}, aapla rate {p}"],
    "mrp_rev": ["aapla rate {p}, MRP {m} aahe", "tumhala {p} la deto, MRP {m} aahe",
                "{p} la deu, chhapil kimmat {m} aahe"],
    "old_new": ["aadhi {old} hota, aata {p} zala", "magchya mahinyat {old} hota, aata {p} aahe"],
    "old_down": ["aadhi {old} hota, aata {p} la detoy", "rate kami zalay, {old} hota, aata {p}"],
    "old_rev": ["aata {p} aahe, aadhi {old} hota", "rate badalla, aata {p}, aadhi {old} hota",
                "{p} padel aata, magchya mahinyat {old} la hota"],
    "unit_bulk": ["eka nagache {p}, {bn} nagancha box {bulk} la", "ek piece {p}, {bn} ghetle tar {bu} pramane lavto",
                  "sutta {p} la, {bn} nagancha box ghetla tar {bulk}"],
    "unit_bulk_rev": ["{bn} nagancha box {bulk} la, sutta ek {p}", "{bn} ghetle tar {bu} pramane, ek piece {p}"],
    "pack": ["packet {p} la aahe, sutta ek {unit} la", "purna packet {p}, sutte pan aahet {unit} la ek"],
    "pack_rev": ["sutta ek {unit} la, purna packet {p}", "ek piece {unit} la milto, packet {p} la"],
    "gst": ["{p} plus GST", "{p}, GST vegla", "GST dharun {p}", "{p} aani 18 takke GST vegla",
            "bill pahije asel tar GST vegla, rate {p}"],
    "fee": ["pathavto, {fee} rupaye delivery charge lagel", "ghari pathavto, {fee} vegle",
            "porga pathavto, {fee} rupaye rikshache"],
    "item_price": ["maalache {p}, delivery vegli", "vastuche {p} rupaye, delivery vegli",
                   "{p} la aahe, delivery charge vegla", "item {p} la, pathavnyache vegle"],
    "timing": ["dukan {h} vajeparyant ughda aahe", "{h} vajeparyant mi ithech aahe", "{h} vajta dukan band hota"],
    "alt": ["{alt} aahe, chalel ka?", "tyachya aivaji {alt} deu shakto", "{alt} aahe aamchyakade",
            "{alt} ghya, te chalel", "{alt} chalnar asel tar sanga"],
    "alt_neg": ["{ask} nahiye, {alt} aahe", "{ask} sampla, {alt} aahe, chalel?"],
    "alt_price": ["{alt} aahe, {ap} la", "{alt} deu shakto, {ap} padel", "{alt} aahe, tyache {ap} rupaye"],
    "other": ["{alt} pan aahe, te {ap} la", "{alt} pahije asel tar te pan aahe, {ap}"],
    "other_first": ["{alt} pan aahe, tyache {ap}, aani tumhi vicharlela {p} la", "{alt} cha rate {ap}, ha {p} la"],
    "del_yes": ["ghari pathavto", "delivery deto", "porga pathavto", "gharpoch milel", "sandhyakali pathavun deto",
                "address pathva, pathavto"],
    "del_no": ["delivery nahi, dukanat yeun ghya", "yeun nyava lagel", "delivery nahi karat",
               "pathavayla manus nahi, ithe yeun ghya", "gharpoch nahi, tumhala yava lagel"],
    "close": ["ajun kahi?", "sanga kay karaycha", "theek aahe?", "kalva mag"],
    "noprice": ["rate malakanna vicharun sangto", "kimmat nantar sangto, aatta list nahi", "rate sheth sangtil"],
    "wrong": ["{w}... nahi nahi, {p}", "{w} sangitla ka? chuklo, {p} aahe", "{w}... thamba, list baghto... {p}"],
    "wrong_rev": ["{p} aahe, {w} nahi", "{p} padel, {w} cha rate juna aahe, to nahi"],
    "corr_yes": ["nahi vatta... thamba, aahe, ek urlay", "sampla hota... nahi, ek aahe ajun",
                 "nahiye bahutek... ho ho, sapadla, aahe"],
    "corr_no": ["aahe... ek minute, nahi, te kal sampla", "ho aahe... nahi nahi, to dusra aahe, ha nahiye"],
    "trap_yes": ["nahi kasa, aahe na", "nahi asa nahi, aahe", "sampla nahi ajun, aahe", "kami nahi, bharpur aahe",
                 "nahi nahi, aahe"],
    "trap_no": ["ho ho, samajla, pan te nahiye", "haan bola... nahi, te sadhya nahi",
                "ho, thevto aamhi, pan aatta samplay"],
    "trap_del_yes": ["delivery chi kalji karu naka, pathavto", "pathavayla kahi problem nahi"],
    "trap_del_no": ["pathavla asta, pan aaj porga nahi, yeun ghya", "delivery? nahi ho, te nahi karat"],
    "nostock_price": ["rate {x} aahe, pan aatta stock nahi", "{x} la yeto, pan sadhya samplay"],
    "unclear_price": ["aadhi {x} la hota, aata aahe ki nahi baghava lagel"],
    "chatter": ["kay sangu, sakal pasun gardi aahe", "porga chaha aanayla gelay", "aaj sakali light navhti",
                "bazarat maal kami aahe sadhya", "company ne parat rate vadhavle", "Diwali chi gardi aahe na",
                "ek giraik aahe, ek minute", "aawaj tutatoy, motyane bola", "kal Ravivar hota mhanun dukan band hota",
                "haan haan, aiktoy"],
    "filler": ["mhanje", "bagha", "bara", "haan tar", "kay aahe na"],
    "address": ["dada", "bhau", "saheb", "kaka"],
}

BANKS["en"] = {
    "open": ["Hello?", "Yes, tell me", "Hello, who is this?", "Yes sir, tell me", "Hello, good morning",
             "Yes, go ahead", "Hello, yes, I can hear you", "Yes, speak"],
    "yes": ["Yes, we have it", "Yes, available", "It's in stock", "Yes sir, we have", "Available, sir",
            "We have it, yes", "Yes, it's there", "Yes, we keep that", "Yes yes, available"],
    "yes_qty": ["Yes, we have {q} {u}", "Only {q} {u} left", "We have {q} {u} in stock right now"],
    "no": ["No, we don't have it", "Out of stock", "Sorry, finished", "We don't keep that",
           "Not available right now", "Sold out, sir", "No sir, not there", "It's finished, sorry",
           "That we don't have"],
    "tomorrow": ["Not today, it comes tomorrow", "Not in stock now, it will come tomorrow evening",
                 "Not now, new stock in two days", "It's not there right now, it's ordered, should come by Monday"],
    "order": ["We don't have it, but I can order it, takes two days", "Not in stock, we can get it on order"],
    "unclear": ["Hold on, let me check", "Owner is not here, call later", "Not sure, I'll have to check the godown",
                "Can you call after some time?", "I'm busy right now, call in half an hour",
                "Let me check and call you back", "Hello? I can't hear you, call again later"],
    "price": ["It's {p} rupees", "Price is {p}", "{p} rupees", "That will be Rs {p}", "Rs {p} only", "₹{p}",
              "{p}, final price", "For you, {p}", "{p}/-", "It comes to {p}"],
    "mrp": ["MRP is {m}, we'll give it for {p}", "{m} is the MRP, for you {p}", "MRP {m}, but our price is {p}",
            "Printed price is {m}, I'll give it at {p}"],
    "mrp_rev": ["Our price is {p}, MRP is {m}", "I'll give it for {p}, the MRP is {m}", "{p} for you, printed price is {m}"],
    "old_new": ["Earlier it was {old}, now it's {p}", "It was {old} before, now {p}"],
    "old_down": ["It was {old}, now reduced to {p}"],
    "old_rev": ["It's {p} now, earlier it was {old}", "Price changed, it's {p} now, was {old} last month",
                "Now {p}, before it was {old}"],
    "unit_bulk": ["{p} for one, the box of {bn} is {bulk}", "One piece {p}, if you take {bn} I'll do {bu} each"],
    "unit_bulk_rev": ["The box of {bn} is {bulk}, single piece {p}", "If you take {bn} it's {bu} each, one piece is {p}"],
    "pack": ["The packet is {p}, loose ones are {unit} each", "{p} for the full packet, single piece {unit}"],
    "pack_rev": ["Loose ones are {unit} each, the full packet is {p}", "Single piece {unit}, full packet {p}"],
    "gst": ["{p} plus GST", "{p}, GST extra", "{p} including GST", "{p} + 18% GST"],
    "fee": ["We can deliver, {fee} rupees delivery charge", "Delivery is {fee} extra",
            "We'll send it, {fee} for delivery"],
    "item_price": ["The item is {p}, delivery separate", "{p} for the item, plus delivery",
                   "Item price is {p}, delivery is extra"],
    "timing": ["We are open till {h} pm", "Shop closes at {h}", "I'm here till {h} o'clock"],
    "alt": ["We have {alt}, will that work?", "I can give {alt} instead", "{alt} is there, you can take that",
            "We have {alt}, same thing"],
    "alt_neg": ["{ask} is not there, we have {alt}", "No {ask}, only {alt}"],
    "alt_price": ["We have {alt}, that's {ap}", "{alt} is available, {ap} rupees"],
    "other": ["{alt} is also there, that one is {ap}", "We also have {alt} at {ap}"],
    "other_first": ["{alt} is also there at {ap}, and this one is {p}", "The {alt} costs {ap}, the one you asked for is {p}"],
    "del_yes": ["We can deliver", "Home delivery is available", "I'll send it with our boy", "We'll deliver by evening",
                "Delivery is free"],
    "del_no": ["No delivery, you'll have to come to the shop", "Pickup only", "We don't deliver",
               "You'll have to collect it from the shop"],
    "close": ["Anything else?", "Let me know", "Tell me what to do", "OK?"],
    "noprice": ["I don't know the rate, the owner will tell you", "Rate I'll confirm and tell you"],
    "wrong": ["{w}... sorry, no, {p}", "{w}, wait, I'm checking the list... it's {p}", "{w}, sorry, I mean {p}"],
    "wrong_rev": ["It's {p}, not {w}", "{p}, the {w} rate is old, not that"],
    "corr_yes": ["No... wait, yes, we have one piece left", "I thought it was finished, but there's one left"],
    "corr_no": ["Yes... one minute, no, that got finished yesterday",
                "Yes it's... no sorry, that's the other one, this one is not there"],
    "trap_yes": ["No no, it's there", "It's not finished, we have plenty", "Not out of stock, we have it",
                 "Why not, it's available"],
    "trap_no": ["Yes yes, I understood, but we don't have it", "Yes, tell me... no, that's not there",
                "Yes we keep it, but right now it's finished"],
    "trap_del_yes": ["Delivery is not a problem, we'll send it", "No issue with delivery, we'll send"],
    "trap_del_no": ["I would send it, but my boy is on leave today, come and take it",
                    "Delivery? No, we don't do that"],
    "nostock_price": ["It's {x} rupees, but right now no stock", "Normally {x}, but it's finished"],
    "unclear_price": ["It used to be {x}, I'll have to check if it's there"],
    "chatter": ["It's very busy since morning", "My boy has gone for tea", "There was no power since morning",
                "Market is short of stock these days", "Company increased rates again",
                "It's festival season, everyone is asking", "One customer is here, one minute",
                "Your voice is breaking, speak loudly", "We were closed yesterday, Sunday"],
    "filler": ["uh", "see", "actually", "basically", "okay so"],
    "address": ["sir", "madam", "boss"],
}

BANKS["hi_dev"] = {
    "open": ["हाँ बोलिए", "हैलो, कौन?", "जी बोलिए", "हाँ भाई, बोलो", "हाँ जी, बताइए", "हैलो? हाँ, आवाज़ आ रही है",
             "नमस्ते, बोलिए", "एक मिनट... हाँ बोलिए"],
    "yes": ["हाँ, है", "हाँ जी, मिल जाएगा", "स्टॉक में है", "है भाई", "हाँ, रखा है", "जी, उपलब्ध है", "हाँ हाँ, है",
            "मिल जाएगा, है अपने पास", "है ना, कितने चाहिए?", "हाँ, माल पड़ा है"],
    "yes_m": ["{noun} है, मिल जाएगा", "हाँ, {noun} है", "{noun} पड़ा है", "{noun} ख़त्म नहीं हुआ, है", "हाँ जी, {noun} रखा है"],
    "yes_f": ["{noun} है, मिल जाएगी", "हाँ, {noun} है", "{noun} पड़ी है", "{noun} ख़त्म नहीं हुई, है", "हाँ जी, {noun} रखी है"],
    "yes_qty": ["हाँ, {q} {u} पड़े हैं", "अभी {q} {u} ही बचे हैं", "{q} {u} हैं स्टॉक में"],
    "no": ["नहीं है", "ख़त्म हो गया", "अभी नहीं है", "वो हम नहीं रखते", "माल खत्म है", "स्टॉक में नहीं है",
           "नहीं मिलेगा भाई", "वो तो कल ही ख़त्म हो गया", "नहीं है जी, ख़त्म"],
    "no_m": ["{noun} ख़त्म हो गया", "{noun} नहीं है, खत्म हो गया"],
    "no_f": ["{noun} ख़त्म हो गई", "{noun} नहीं है, खत्म हो गई"],
    "tomorrow": ["आज नहीं है, कल आएगा", "अभी नहीं है, परसों तक आ जाएगा", "अभी नहीं है, ऑर्डर दिया है, दो दिन में आएगा",
                 "शाम तक माल आ जाएगा, अभी नहीं है", "कल सुबह आ जाएगा, अभी नहीं"],
    "order": ["नहीं है, मंगवा देंगे, दो दिन लगेंगे", "अभी नहीं है, ऑर्डर पर मंगवा देंगे"],
    "unclear": ["रुकिए, देखकर बताता हूँ", "मालिक नहीं हैं, बाद में फ़ोन कीजिए", "पता नहीं, गोदाम में देखना पड़ेगा",
                "थोड़ी देर बाद कॉल करो", "अभी बिज़ी हूँ, आधे घंटे बाद फ़ोन करना", "सेठ जी बाहर गए हैं, वही बताएँगे",
                "चेक करके बताता हूँ", "हैलो? आवाज़ नहीं आ रही, बाद में करना"],
    "unclear_n": ["{noun} है कि नहीं, देखकर बताता हूँ", "{noun} का स्टॉक चेक करके बताता हूँ",
                  "{noun} का पता नहीं, गोदाम में देखना पड़ेगा"],
    "price": ["{p} रुपये का है", "{p} लगेगा", "रेट {p} है", "{p} का पड़ेगा", "{p} रुपये", "{p} में दे देंगे",
              "{p} फ़ाइनल", "आपके लिए {p}", "{p} का है, इससे कम नहीं होगा", "{p} रुपये लगेंगे"],
    "mrp": ["MRP {m} है, आपको {p} में दे देंगे", "{m} MRP लिखा है, {p} लगा देंगे",
            "एमआरपी {m} है, पर हम {p} में देते हैं"],
    "mrp_rev": ["आपको {p} में दे देंगे, MRP {m} है", "हमारा रेट {p}, MRP तो {m} है", "{p} लगा देंगे, वैसे MRP {m} लिखा है"],
    "old_new": ["पहले {old} था, अब {p} हो गया", "पिछले महीने {old} था, अभी {p}"],
    "old_down": ["पहले {old} था, अब {p} हो गया, सस्ता हुआ है"],
    "old_rev": ["अब {p} है, पहले {old} था", "रेट बदल गया है, अब {p} है, पहले {old} था",
                "अभी रेट {p} हो गया, पिछले महीने {old} था"],
    "unit_bulk": ["एक का {p}, {bn} का डिब्बा {bulk} का", "एक पीस {p}, {bn} लोगे तो {bu} के हिसाब से"],
    "unit_bulk_rev": ["{bn} का डिब्बा {bulk} का, एक पीस {p}", "{bn} लोगे तो {bu} के हिसाब से, एक का {p}"],
    "pack": ["पैकेट {p} का है, खुला एक {unit} का", "पूरा पैकेट {p}, अलग से एक पीस {unit} का"],
    "pack_rev": ["खुला एक {unit} का, पूरा पैकेट {p}", "एक पीस {unit} का मिलता है, पैकेट {p} का"],
    "gst": ["{p} प्लस GST", "{p} है, GST अलग से", "GST मिलाकर {p}", "{p} और 18 परसेंट GST अलग"],
    "fee": ["भेज देंगे, {fee} रुपये डिलीवरी चार्ज लगेगा", "घर पर भेज देंगे, {fee} अलग से",
            "डिलीवरी का {fee} एक्स्ट्रा"],
    "item_price": ["माल का {p}, डिलीवरी अलग", "सामान {p} का है, डिलीवरी अलग से", "{p} का है, प्लस डिलीवरी"],
    "timing": ["दुकान {h} बजे तक खुली है", "{h} बजे तक मैं यहीं हूँ"],
    "alt": ["{alt} है, चलेगा?", "उसकी जगह {alt} दे सकते हैं", "{alt} रखा है", "{alt} है, वो ले लीजिए"],
    "alt_neg": ["{ask} नहीं है, {alt} है", "{ask} वाला ख़त्म है, {alt} है, चलेगा?"],
    "alt_price": ["{alt} है, {ap} का", "{alt} दे सकता हूँ, {ap} रुपये का", "{alt} चलेगा? उसका {ap} है"],
    "other": ["{alt} भी है, वो {ap} का", "{alt} चाहिए तो वो भी है, {ap} वाला"],
    "other_first": ["{alt} भी है, उसका {ap}, और आपवाला {p} का", "{alt} का रेट {ap} है, ये वाला {p}"],
    "del_yes": ["घर पर भेज देंगे", "होम डिलीवरी हो जाएगी", "लड़का भेज देता हूँ", "शाम तक पहुँचा देंगे",
                "डिलीवरी फ़्री है"],
    "del_no": ["डिलीवरी नहीं करते, दुकान पर आकर ले जाइए", "डिलीवरी नहीं है", "आपको आकर लेना पड़ेगा",
               "भेजने वाला कोई नहीं है, आ के ले जाओ"],
    "close": ["और कुछ?", "बताइए क्या करना है", "ठीक है?"],
    "noprice": ["रेट मालिक बताएँगे", "रेट पूछ कर बताता हूँ"],
    "wrong": ["{w}... नहीं नहीं, {p} है", "{w}... रुकिए, लिस्ट देखता हूँ... {p}", "{w} बोला क्या? माफ़ करना, {p} है"],
    "wrong_rev": ["{p} है, {w} नहीं", "{p} लगेगा, {w} वाला रेट पुराना है, वो नहीं"],
    "corr_yes": ["नहीं है... रुको रुको, है, एक पड़ा है", "ख़त्म हो गया था... नहीं, एक बचा है"],
    "corr_no": ["हाँ है... एक मिनट, नहीं, वो कल ख़त्म हो गया", "है... अरे नहीं, वो दूसरा वाला है, ये वाला नहीं है"],
    "trap_yes": ["नहीं नहीं, है ना", "ऐसा नहीं है कि नहीं है, है", "ख़त्म नहीं हुआ, अभी भी है",
                 "कम नहीं है, बहुत पड़ा है"],
    "trap_no": ["हाँ हाँ, समझ गया, लेकिन वो अभी नहीं है", "हाँ रखते हैं, पर अभी ख़त्म है"],
    "trap_del_yes": ["डिलीवरी की टेंशन मत लो, भेज देंगे", "भेजने में कोई दिक्कत नहीं"],
    "trap_del_no": ["भेज देता, पर आज लड़का नहीं आया, आकर ले जाओ"],
    "nostock_price": ["रेट {x} है, पर अभी स्टॉक नहीं है", "{x} का आता है, लेकिन अभी ख़त्म है"],
    "unclear_price": ["पहले {x} का था, अभी है कि नहीं देखना पड़ेगा"],
    "chatter": ["अरे क्या बताऊँ, सुबह से बहुत भीड़ है", "लड़का चाय लेने गया है", "आज सुबह से लाइट नहीं है",
                "बाज़ार में माल की कमी चल रही है", "कंपनी ने फिर रेट बढ़ा दिया", "त्योहार का सीज़न है ना",
                "एक ग्राहक खड़ा है, एक मिनट", "आवाज़ कट रही है, ज़ोर से बोलिए"],
    "filler": ["मतलब", "देखिए", "अच्छा", "हाँ तो"],
    "address": ["भाई", "जी", "सर"],
}

BANKS["mr_dev"] = {
    "open": ["हां बोला", "हॅलो, कोण बोलतंय?", "बोला ना", "हो, बोला दादा", "नमस्कार, बोला",
             "हॅलो? आवाज येतोय का? हां, बोला", "एक मिनिट हां... बोला", "हां बोला साहेब", "बोला, काय पाहिजे?"],
    "yes": ["हो, आहे", "आहे ना", "स्टॉकमध्ये आहे", "आहे आहे", "हो, मिळेल", "हो दादा, आहे", "आमच्याकडे आहे",
            "आहे की, मिळेल", "हो, ठेवतो आम्ही, आहे", "हो, आहे शिल्लक"],
    "yes_m": ["{noun} आहे, मिळेल", "हो, {noun} आहे", "{noun} पडलाय", "{noun} संपला नाही, आहे", "हो, {noun} शिल्लक आहे"],
    "yes_f": ["{noun} आहे, मिळेल", "हो, {noun} आहे", "{noun} पडलीये", "{noun} संपली नाही, आहे", "हो, {noun} शिल्लक आहे"],
    "yes_n": ["{noun} आहे, मिळेल", "हो, {noun} आहे", "{noun} पडलंय", "{noun} संपलं नाही, आहे", "हो, {noun} शिल्लक आहे"],
    "yes_qty": ["हो, {q} {u} आहेत", "{q}च {u} उरलेत", "{q} नग शिल्लक आहेत", "आहे, अजून {q} {u} पडलेत"],
    "no": ["नाही आहे", "संपलं", "सध्या नाही", "आम्ही ते ठेवत नाही", "नाहीये", "संपलंय", "माल संपला", "स्टॉक संपलाय",
           "आत्ता नाहीये", "शिल्लक नाही", "ते कालच संपलं", "नाय हो, नाहीये", "ते मिळणार नाही इथे", "आत्ता मिळत नाही",
           "सगळं संपलं", "नाही हो, संपलं ते", "ते आमच्याकडे नसतं", "संपले सगळे", "माल नाहीये", "स्टॉकमध्ये नाहीये",
           "नाही ना, संपलंय", "संपलंय, उद्या दुपारपर्यंत येईल"],
    "no_m": ["{noun} संपला", "{noun} संपलाय"],
    "no_f": ["{noun} संपली", "{noun} संपलीये", "{noun} नाहीये, संपली", "ती संपली हो", "{noun} आत्ता नाहीये, संपली"],
    "no_n": ["{noun} संपलं", "{noun} संपलंय"],
    "tomorrow": ["आज नाही, उद्या येईल", "उद्या सकाळी माल येईल, आत्ता नाहीये", "आत्ता नाहीये, परवापर्यंत येईल",
                 "सध्या नाहीये, ऑर्डर दिलीय, दोन दिवसांत येईल", "आत्ता नाहीये, संध्याकाळी येईल"],
    "order": ["नाहीये, मागवून देतो, दोन दिवस लागतील", "सध्या नाही, मागवावं लागेल", "आमच्याकडे नसतं, ऑर्डर करून मागवतो"],
    "unclear": ["थांबा, बघून सांगतो", "मालक नाहीत, नंतर फोन करा", "माहीत नाही, गोडाऊनमध्ये बघावं लागेल",
                "जरा वेळाने फोन करा", "आत्ता गडबडीत आहे, अर्ध्या तासाने फोन करा", "शेठ बाहेर गेलेत, ते आले की सांगतो",
                "बघतो आणि कळवतो", "हॅलो? आवाज येत नाहीये, नंतर करा", "आत्ता सांगता येणार नाही, संध्याकाळी फोन करा"],
    "unclear_n": ["{noun} आहे का बघावं लागेल, नंतर सांगतो", "{noun}चा स्टॉक बघून सांगतो",
                  "{noun} आहे की नाही माहीत नाही, गोडाऊनमध्ये बघतो"],
    "price": ["{p} रुपये", "{p}_ला पडेल", "रेट {p} आहे", "{p} रुपयांना पडेल", "{p}_ला देतो", "{p} फिक्स",
              "तुम्हाला {p}_ला देतो", "{p} होतील", "{p} रुपये होतील", "{p} रुपये, कमी नाही होणार"],
    "mrp": ["MRP {m} आहे, तुम्हाला {p}_ला देतो", "{m} MRP आहे, पण {p}_ला देऊ", "छापील किंमत {m} आहे, आम्ही {p}_ला देतो",
            "एमआरपी {m}, आपला रेट {p}"],
    "mrp_rev": ["आमचा रेट {p}, MRP {m}", "तुम्हाला {p}_ला देतो, MRP {m} आहे", "{p}_ला देऊ, छापील किंमत {m} आहे"],
    "old_new": ["आधी {old} होता, आता {p} झाला", "मागच्या महिन्यात {old} होता, आता {p} आहे"],
    "old_down": ["आधी {old} होता, आता {p}_ला देतोय", "रेट कमी झालाय, {old} होता, आता {p}"],
    "old_rev": ["आता {p} आहे, आधी {old} होता", "रेट बदललाय, आता {p}, आधी {old} होता",
                "आता रेट {p} झाला, मागच्या महिन्यात {old} होता"],
    "unit_bulk": ["एका नगाचे {p}, {bn} नगांचा बॉक्स {bulk}_ला", "एक पीस {p}, {bn} घेतले तर {bu} प्रमाणे लावतो",
                  "सुटा {p}_ला, {bn} नगांचा बॉक्स घेतला तर {bulk}"],
    "unit_bulk_rev": ["{bn} नगांचा बॉक्स {bulk}_ला, सुटा एक {p}", "{bn} घेतले तर {bu} प्रमाणे, एक पीस {p}"],
    "pack": ["पॅकेट {p}_ला आहे, सुटा एक {unit}_ला", "पूर्ण पॅकेट {p}, सुटे पण आहेत {unit}_ला एक"],
    "pack_rev": ["सुटा एक {unit}_ला, पूर्ण पॅकेट {p}", "एक पीस {unit}_ला मिळतो, पॅकेट {p}_ला"],
    "gst": ["{p} प्लस GST", "{p}, GST वेगळा", "GST धरून {p}", "{p} आणि 18 टक्के GST वेगळा",
            "बिल हवं असेल तर GST वेगळा, रेट {p}"],
    "fee": ["पाठवतो, {fee} रुपये डिलिव्हरी चार्ज लागेल", "घरी पाठवतो, {fee} वेगळे",
            "पोरगा पाठवतो, {fee} रुपये रिक्षाचे"],
    "item_price": ["मालाचे {p}, डिलिव्हरी वेगळी", "वस्तू {p}_ला आहे, डिलिव्हरी वेगळी", "आयटमचे {p} रुपये, डिलिव्हरी चार्ज वेगळा"],
    "timing": ["दुकान {h} वाजेपर्यंत उघडं आहे", "{h} वाजेपर्यंत मी इथेच आहे", "{h} वाजता दुकान बंद होतं"],
    "alt": ["{alt} आहे, चालेल का?", "त्याच्याऐवजी {alt} देऊ शकतो", "{alt} आहे आमच्याकडे", "{alt} घ्या, ते चालेल",
            "{alt} चालणार असेल तर सांगा"],
    "alt_neg": ["{ask} नाहीये, {alt} आहे", "{ask} संपलं, {alt} आहे, चालेल?"],
    "alt_price": ["{alt} आहे, {ap}_ला", "{alt} देऊ शकतो, {ap} पडेल", "{alt} आहे, त्याचे {ap} रुपये"],
    "other": ["{alt} पण आहे, ते {ap}_ला", "{alt} हवं असेल तर ते पण आहे, {ap}"],
    "other_first": ["{alt} पण आहे, त्याचे {ap}, आणि तुम्ही विचारलेला {p}_ला", "{alt} आहे {ap}_ला, पण हा {p}_ला मिळेल"],
    "del_yes": ["घरी पाठवतो", "घरपोच देतो", "पोरगा पाठवतो", "संध्याकाळपर्यंत पाठवून देतो", "डिलिव्हरी फ्री आहे",
                "पत्ता पाठवा, पाठवून देतो"],
    "del_no": ["डिलिव्हरी नाही, दुकानात येऊन घ्या", "येऊन न्यावं लागेल", "डिलिव्हरी करत नाही",
               "पाठवायला माणूस नाही, इथे येऊन घ्या", "घरपोच नाही, तुम्हाला यावं लागेल"],
    "close": ["अजून काही?", "सांगा काय करायचं", "ठीक आहे?", "कळवा मग"],
    "noprice": ["रेट मालकांना विचारून सांगतो", "किंमत नंतर सांगतो, आत्ता लिस्ट नाही", "रेट शेठ सांगतील"],
    "wrong": ["{w}... नाही नाही, {p}", "{w} सांगितलं का? चुकलो, {p} आहे", "{w}... थांबा, लिस्ट बघतो... {p}"],
    "wrong_rev": ["{p} आहे, {w} नाही", "{p} पडेल, {w}चा रेट जुना आहे, तो नाही"],
    "corr_yes": ["नाही वाटतं... थांबा, आहे, एक उरलाय", "संपलं होतं... नाही, एक आहे अजून",
                 "नाहीये बहुतेक... हो हो, सापडलं, आहे"],
    "corr_no": ["आहे... एक मिनिट, नाही, ते काल संपलं", "हो आहे... नाही नाही, तो दुसरा आहे, हा नाहीये"],
    "trap_yes": ["नाही कसं, आहे ना", "नाही असं नाही, आहे", "संपलं नाही अजून, आहे", "कमी नाही, भरपूर आहे",
                 "नाही नाही, आहे", "नाही म्हणालो का? आहे हो"],
    "trap_no": ["हो हो, समजलं, पण ते नाहीये", "हां बोला... नाही, ते सध्या नाही", "हो, ठेवतो आम्ही, पण आत्ता संपलंय"],
    "trap_del_yes": ["डिलिव्हरीची काळजी करू नका, पाठवतो", "पाठवायला काही प्रॉब्लेम नाही",
                     "डिलिव्हरी नाही असं नाही, पाठवतो की"],
    "trap_del_no": ["पाठवलं असतं, पण आज पोरगा नाही, येऊन घ्या", "डिलिव्हरी? नाही हो, ते नाही करत"],
    "nostock_price": ["रेट {x} आहे, पण आत्ता स्टॉक नाही", "{x}_ला येतो, पण सध्या संपलाय"],
    "unclear_price": ["आधी {x}_ला होता, आता आहे की नाही बघावं लागेल", "रेट {x}च्या आसपास असेल, स्टॉक बघून सांगतो"],
    "chatter": ["काय सांगू, सकाळपासून गर्दी आहे", "पोरगा चहा आणायला गेलाय", "आज सकाळपासून लाईट नाही",
                "बाजारात माल कमी आहे सध्या", "कंपनीने परत रेट वाढवले", "दिवाळीची गर्दी आहे ना",
                "एक गिऱ्हाईक आहे, एक मिनिट", "आवाज तुटतोय, मोठ्याने बोला", "हां हां, ऐकतोय"],
    "filler": ["म्हणजे", "बघा", "बरं", "हां तर"],
    "address": ["दादा", "भाऊ", "साहेब", "काका"],
}

# Code-mixed: Hinglish and Marathi-English as Pune shops talk. Missing slots come from hi_rom / mr_rom.
BANKS["hien"] = {
    "open": ["haan sir, bataiye", "yes sir, boliye", "hello, haan boliye", "ji sir, tell me"],
    "yes": ["haan sir, available hai", "stock mein hai sir", "yes, available hai", "haan, ready stock hai",
            "available hai, koi problem nahi", "haan sir, wo model available hai"],
    "yes_qty": ["{q} {u} available hai abhi", "stock mein sirf {q} {u} hai"],
    "no": ["sorry sir, abhi out of stock hai", "stock nahi hai abhi", "wo model available nahi hai",
           "abhi stock finish ho gaya", "not available sir", "wo item hum keep nahi karte"],
    "tomorrow": ["abhi stock nahi hai, kal tak aa jayega", "abhi out of stock hai, new stock kal aa raha hai",
                 "abhi available nahi hai, order place kiya hai, Monday tak aayega"],
    "unclear": ["sir ek minute, check karke batata hun", "owner abhi available nahi hain, baad mein call kijiye",
                "abhi confirm nahi kar sakta, godown check karna padega", "please thodi der baad call karo"],
    "price": ["price {p} hai", "{p} rupees hai", "rate {p} hai sir", "{p} final price hai", "aapke liye {p}",
              "best price {p}"],
    "mrp": ["MRP {m} hai, but aapko {p} mein de denge", "MRP {m} hai, discount ke baad {p} padega",
            "printed MRP {m}, our price {p}"],
    "mrp_rev": ["our price {p}, MRP {m} hai", "aapko {p} mein milega, MRP {m} hai"],
    "old_new": ["price increase ho gaya, pehle {old} tha, now {p}", "last month {old} tha, ab {p} hai"],
    "old_rev": ["now price {p} hai, pehle {old} tha", "ab {p} hai, last month {old} tha"],
    "unit_bulk": ["single piece {p}, {bn} ka full box {bulk}", "per piece {p}, {bn} lo toh {bu} lagega"],
    "unit_bulk_rev": ["{bn} ka full box {bulk}, single piece {p}"],
    "pack": ["full packet {p}, loose piece {unit} ka"],
    "pack_rev": ["loose piece {unit} ka, full packet {p}"],
    "gst": ["{p} plus GST", "{p} + GST extra", "{p} inclusive of GST"],
    "fee": ["delivery ho jayegi, {fee} rupees charge extra", "home delivery available hai, {fee} charge"],
    "item_price": ["item price {p}, delivery charge alag", "product ka {p}, delivery extra"],
    "timing": ["shop {h} baje tak open hai"],
    "alt": ["{alt} available hai, chalega?", "{alt} hai, same quality", "uski jagah {alt} le sakte ho"],
    "alt_price": ["{alt} available hai, uska price {ap}", "{alt} hai, {ap} ka"],
    "del_yes": ["home delivery available hai", "delivery ho jayegi sir", "same day delivery kar denge",
                "free delivery hai"],
    "del_no": ["delivery nahi hoti, pickup karna padega", "sorry, delivery service nahi hai",
               "aapko shop se pickup karna hoga"],
    "trap_yes": ["no no, available hai", "out of stock nahi hai, available hai"],
    "chatter": ["sir aaj bahut busy day hai", "network issue hai, aawaz break ho rahi hai",
                "season time hai na, demand zyada hai"],
}
BANKS["mren"] = {
    "open": ["ho sir, bola", "hello, haan bola", "yes, bola"],
    "yes": ["ho sir, available aahe", "stock madhe aahe", "aahe sir, ready stock aahe"],
    "no": ["sorry, out of stock aahe", "stock nahi aahe sadhya", "te model available nahi"],
    "tomorrow": ["aatta stock nahi, udya yeil", "sadhya out of stock aahe, new stock parva yenar aahe"],
    "unclear": ["ek minute, check karun sangto", "owner nahit, nantar call kara"],
    "price": ["price {p} aahe", "rate {p} aahe sir", "{p} final", "tumhala {p} la deu"],
    "mrp": ["MRP {m} aahe, pan tumhala {p} la deu", "MRP {m}, discount nantar {p}"],
    "mrp_rev": ["tumhala {p} la deu, MRP {m} aahe"],
    "old_new": ["aadhi {old} hota, aata price {p} zala"],
    "gst": ["{p} plus GST", "{p}, GST extra"],
    "fee": ["delivery deto, {fee} charge lagel"],
    "item_price": ["item cha price {p}, delivery vegli"],
    "alt": ["{alt} available aahe, chalel ka?", "{alt} aahe, same quality"],
    "alt_price": ["{alt} aahe, tyacha price {ap}"],
    "del_yes": ["home delivery aahe", "delivery deto sir"],
    "del_no": ["delivery nahi, shop varun pickup kara", "pickup karava lagel"],
}

# unit words: {u}
UNITS = {
    "hi_rom": {"piece": "piece", "dabba": "dabbe", "roll": "roll", "bag": "bag", "packet": "packet"},
    "mr_rom": {"piece": "piece", "dabba": "dabe", "roll": "roll", "bag": "bag", "packet": "packet"},
    "en": {"piece": "pieces", "dabba": "cans", "roll": "rolls", "bag": "bags", "packet": "packets"},
    "hi_dev": {"piece": "पीस", "dabba": "डिब्बे", "roll": "रोल", "bag": "बैग", "packet": "पैकेट"},
    "mr_dev": {"piece": "पीस", "dabba": "डबे", "roll": "रोल", "bag": "बॅगा", "packet": "पॅकेट"},
}
BASE = {"hi_rom": "hi_rom", "mr_rom": "mr_rom", "en": "en", "hi_dev": "hi_dev", "mr_dev": "mr_dev",
        "hien": "hi_rom", "mren": "mr_rom"}
FAMILY = {"hi_rom": "hi", "mr_rom": "mr", "en": "en", "hi_dev": "hi", "mr_dev": "mr"}
SCRIPT = {"hi_rom": "rom", "mr_rom": "rom", "en": "en", "hi_dev": "dev", "mr_dev": "dev"}
LANG_WEIGHTS = {"hi_rom": 0.20, "mr_rom": 0.16, "en": 0.12, "hi_dev": 0.14, "mr_dev": 0.20, "hien": 0.08,
                "mren": 0.05, "mrhi": 0.05}
LANG_NAMES = {"hi_rom": "romanized Hindi", "mr_rom": "romanized Marathi", "en": "Indian English",
              "hi_dev": "Devanagari Hindi", "mr_dev": "Devanagari Marathi", "hien": "code-mixed",
              "mren": "code-mixed", "mrhi": "code-mixed"}
SEP = {"rom": [", "] * 6 + [". ", " "], "en": [". "] * 3 + [", "], "dev": [", "] * 3 + ["। ", ". "]}
# Code-mixed Marathi-Hindi ("mrhi"): a row is in one main language and borrows only these from the other.
BORROW = ("open", "close", "filler", "address", "chatter")

# Devanagari spellings (Hindi, Marathi) of the words in alternative names, so that a Devanagari reply can name
# the alternative in Devanagari ("हॅवेल्स फॅन आहे"); the label is then written the same way. Names with other
# words ("12W", most brands) stay in Latin letters.
DEV_WORDS = {
    "cover": ("कवर", "कव्हर"), "fan": ("फैन", "फॅन"), "bulb": ("बल्ब", "बल्ब"), "local": ("लोकल", "लोकल"),
    "oil": ("ऑयल", "ऑइल"), "drill": ("ड्रिल", "ड्रिल"), "headlight": ("हेडलाइट", "हेडलाईट"), "pump": ("पंप", "पंप"),
    "grinder": ("ग्राइंडर", "ग्राइंडर"), "wiper": ("वाइपर", "वायपर"), "batten": ("बैटन", "बॅटन"),
    "machine": ("मशीन", "मशीन"), "mat": ("मैट", "मॅट"), "geyser": ("गीज़र", "गिझर"), "iron": ("आयरन", "आयर्न"),
    "switch": ("स्विच", "स्विच"), "tyre": ("टायर", "टायर"), "horn": ("हॉर्न", "हॉर्न"), "rod": ("रॉड", "रॉड"),
    "torch": ("टॉर्च", "टॉर्च"), "multimeter": ("मल्टीमीटर", "मल्टीमीटर"), "helmet": ("हेलमेट", "हेल्मेट"),
    "lock": ("लॉक", "लॉक"), "heater": ("हीटर", "हीटर"), "sealant": ("सीलेंट", "सीलंट"), "led": ("एलईडी", "एलईडी"),
    "socket": ("सॉकेट", "सॉकेट"), "bit": ("बिट", "बिट"), "plier": ("प्लायर", "प्लायर"), "bolt": ("बोल्ट", "बोल्ट"),
    "lube": ("ल्यूब", "ल्यूब"), "seal": ("सील", "सील"), "grip": ("ग्रिप", "ग्रिप"),
    "capacitor": ("कैपेसिटर", "कॅपॅसिटर"), "engine": ("इंजन", "इंजिन"), "air": ("एयर", "एअर"),
    "filter": ("फ़िल्टर", "फिल्टर"), "set": ("सेट", "सेट"), "battery": ("बैटरी", "बॅटरी"), "clutch": ("क्लच", "क्लच"),
    "plate": ("प्लेट", "प्लेट"), "mcb": ("एमसीबी", "एमसीबी"), "board": ("बोर्ड", "बोर्ड"), "brake": ("ब्रेक", "ब्रेक"),
    "shoe": ("शू", "शू"), "padlock": ("ताला", "कुलूप"), "putty": ("पुट्टी", "पुट्टी"), "spark": ("स्पार्क", "स्पार्क"),
    "plug": ("प्लग", "प्लग"), "hinge": ("कब्ज़ा", "बिजागरी"), "cutter": ("कटर", "कटर"), "wire": ("वायर", "वायर"),
    "elbow": ("एल्बो", "एल्बो"), "brush": ("ब्रश", "ब्रश"), "m-seal": ("एम-सील", "एम-सील"), "paint": ("पेंट", "पेंट"),
    "roller": ("रोलर", "रोलर"), "teflon": ("टेफ्लॉन", "टेफ्लॉन"), "roll": ("रोल", "रोल"), "thread": ("थ्रेड", "थ्रेड"),
    "tape": ("टेप", "टेप"), "halogen": ("हैलोजन", "हॅलोजन"), "glue": ("ग्लू", "ग्लू"), "kit": ("किट", "किट"),
    "wall": ("वॉल", "वॉल"), "white": ("व्हाइट", "व्हाईट"), "cement": ("सीमेंट", "सिमेंट"), "steel": ("स्टील", "स्टील"),
    "extension": ("एक्सटेंशन", "एक्स्टेंशन"), "mechanical": ("मैकेनिकल", "मेकॅनिकल"), "ball": ("बॉल", "बॉल"),
    "plastic": ("प्लास्टिक", "प्लास्टिक"), "float": ("फ्लोट", "फ्लोट"), "glass": ("ग्लास", "ग्लास"),
    "pipe": ("पाइप", "पाईप"), "mm": ("एमएम", "एमएम"), "inch": ("इंच", "इंच"), "kg": ("किलो", "किलो"),
    "g": ("ग्राम", "ग्राम"), "ml": ("एमएल", "एमएल"), "cpvc": ("सीपीवीसी", "सीपीव्हीसी"),
    "havells": ("हैवेल्स", "हॅवेल्स"), "crompton": ("क्रॉम्पटन", "क्रॉम्प्टन"), "bajaj": ("बजाज", "बजाज"),
    "bosch": ("बॉश", "बॉश"), "stanley": ("स्टैनली", "स्टॅनली"), "philips": ("फिलिप्स", "फिलिप्स"),
    "orient": ("ओरिएंट", "ओरिएंट"), "usha": ("उषा", "उषा"), "anchor": ("एंकर", "अँकर"), "wipro": ("विप्रो", "विप्रो"),
    "michelin": ("मिशलिन", "मिशलिन"), "goodyear": ("गुडईयर", "गुडइयर"), "kirloskar": ("किर्लोस्कर", "किर्लोस्कर"),
    "racold": ("रैकोल्ड", "रॅकोल्ड"), "shell": ("शेल", "शेल"), "castrol": ("कैस्ट्रॉल", "कॅस्ट्रॉल"),
    "syska": ("सिस्का", "सिस्का"), "legrand": ("लेग्रैंड", "लेग्रँड"), "schneider": ("श्नाइडर", "श्नायडर"),
    "steelbird": ("स्टीलबर्ड", "स्टीलबर्ड"), "studds": ("स्टड्स", "स्टड्स"), "vega": ("वेगा", "वेगा"),
    "motul": ("मोतुल", "मोतुल"), "mobil": ("मोबिल", "मोबिल"), "luminous": ("ल्यूमिनस", "ल्युमिनस"),
    "osram": ("ओसराम", "ओस्राम"), "ceat": ("सिएट", "सिएट"), "jk": ("जेके", "जेके"), "mk": ("एमके", "एमके"),
    "jcb": ("जेसीबी", "जेसीबी"), "dewalt": ("डीवॉल्ट", "डीवॉल्ट"), "shakti": ("शक्ति", "शक्ती"),
    "lakshmi": ("लक्ष्मी", "लक्ष्मी"), "khaitan": ("खेतान", "खेतान"), "galaxy": ("गैलेक्सी", "गॅलेक्सी"),
    "fevicol": ("फेविकोल", "फेविकॉल"), "passion": ("पैशन", "पॅशन"), "jupiter": ("ज्यूपिटर", "ज्युपिटर"),
    "dio": ("डियो", "डिओ"), "pulsar": ("पल्सर", "पल्सर"), "v-guard": ("वी-गार्ड", "व्ही-गार्ड"),
    "atomberg": ("एटमबर्ग", "ॲटमबर्ग"), "candes": ("कैंडेस", "कँडेस"), "rico": ("रीको", "रिको"),
    "vista": ("विस्टा", "विस्टा"), "ao": ("एओ", "एओ"), "smith": ("स्मिथ", "स्मिथ"), "royal": ("रॉयल", "रॉयल"),
    "enfield": ("एनफील्ड", "एनफिल्ड"), "golden": ("गोल्डन", "गोल्डन"), "bullet": ("बुलेट", "बुलेट"),
}


def to_devanagari(name: str, family: str) -> str | None:
    """"Havells fan" -> "हॅवेल्स फॅन" (Marathi); None when a word is not in DEV_WORDS. Digits stay."""
    out = []
    for word in name.split():
        if re.fullmatch(r"[\d./]+", word):
            out.append(word)
        elif word.lower() in DEV_WORDS:
            out.append(DEV_WORDS[word.lower()][0 if family == "hi" else 1])
        else:
            return None
    return " ".join(out)

# How many rows of each kind (they add up to N_ROWS).
PLAN = {
    "clear_price": 250, "in_stock_no_price": 70,
    "mrp_vs_offered": 70, "old_vs_new": 45, "unit_vs_bulk": 40, "pack_price": 20, "gst": 30, "delivery_fee": 35,
    "quantity": 25, "timing": 20, "other_variant_price": 15,
    "no_stock": 75, "no_stock_alt": 70, "no_stock_alt_price": 40, "comes_later": 45, "no_stock_price_said": 15,
    "can_order": 15,
    "unclear": 90, "unclear_price_said": 20,
    "price_correction": 40, "stock_correction_yes": 25, "stock_correction_no": 25, "negation_trap_yes": 30,
    "negation_trap_no": 20, "delivery_trap": 20,
    "rambling": 110, "short": 40,
}
GROUPS = {
    "single clear price": ["clear_price"],
    "in stock, no price": ["in_stock_no_price"],
    "several numbers": ["mrp_vs_offered", "old_vs_new", "unit_vs_bulk", "pack_price", "gst", "delivery_fee",
                        "quantity", "timing", "other_variant_price"],
    "not in stock": ["no_stock", "no_stock_alt", "no_stock_alt_price", "comes_later", "no_stock_price_said",
                     "can_order"],
    "unclear": ["unclear", "unclear_price_said"],
    "self-corrections and negation traps": ["price_correction", "stock_correction_yes", "stock_correction_no",
                                            "negation_trap_yes", "negation_trap_no", "delivery_trap"],
    "rambling": ["rambling"],
    "short": ["short"],
}


def make_price(rng: random.Random, low: int, high: int) -> int:
    step = 5 if high <= 100 else 10 if high <= 1000 else 50
    return rng.randrange(low // step * step, high // step * step + step, step)


class Product:
    def __init__(self, name, low, high, alts, ask=None, unit="piece", noun=None, countable=False, pack=None,
                 price=None, mrp=None, source="old"):
        self.name, self.low, self.high, self.alts, self.ask = name, low, high, alts, ask
        self.unit, self.noun, self.countable, self.pack = unit, noun, countable, pack
        self.price, self.mrp, self.source = price, mrp, source


# Words that name the kind of product, not who makes it ("LED", "ceiling fan", "flip-up helmet").
KIND_WORDS = {w for t in TYPES for w in t[1].lower().split()} | GENERIC | {"glass", "bolt", "pipe", "flip-up", "for"}


def maker_words(name: str) -> set[str]:
    """The brand and model words of a product name ("Tata Agrico plier" -> {"tata", "agrico"})."""
    words = set()
    for tok in name.lower().replace("(", " ").split():
        if tok in KIND_WORDS:
            break
        if not re.search(r"\d", tok) and tok not in ("mm", "sq", "hp", "l", "ml", "w", "a", "inch"):
            words.add(tok)
    return words


def all_products() -> list[Product]:
    out = []
    for name, alts, (low, high) in PRODUCTS:
        extra = OLD_EXTRA[name]
        alt_pairs = [(a, extra.get("short", {}).get(a)) for a in alts]
        out.append(Product(name, low, high, alt_pairs, ask=extra.get("ask"), unit=extra.get("unit", "piece"),
                           noun=extra.get("noun"), countable=extra.get("countable", False), pack=extra.get("pack")))
    real = load_real_products()
    by_kind: dict[str, list[dict]] = {}
    for r in real:
        by_kind.setdefault(r["kind"], []).append(r)
    for r in real:
        # Other companies only: "Tata Agrico plier" is no alternative to "Agrico TATA plier", nor "Crompton Greaves"
        # to "Crompton". Two products are from one company when their names share a brand or model word.
        mine, others = maker_words(r["name"]), []
        for o in by_kind[r["kind"]]:
            theirs = maker_words(o["name"])
            if not theirs & mine and not any(theirs & maker_words(x["name"]) for x in others):
                others.append(o)
        alts = [(f"{o['brand']} {r['short_noun']}", o["brand"]) for o in others[:4]]
        if not alts:
            continue
        price = r["price"]
        out.append(Product(r["name"], price, price, alts, ask=r["brand"], noun=r["noun"],
                           countable=r["kind"] in ("switch", "led_bulb", "mcb", "horn", "handle_grip", "helmet_lock"),
                           pack=r["pack"], price=price, mrp=r["mrp"], source=r["source"]))
    return out


# ---------------------------------------------------------------------------------------------
# Building one row.

PLACEHOLDER = re.compile(r"\{(\w+)\}(_?)")
NUMERIC = ("p", "m", "old", "bulk", "bn", "bu", "unit", "fee", "q", "h", "ap", "w", "x")
COUNTS = ("q", "h", "bn")  # numbers that are not prices: said as they are, never moved to the price grid


def lower_first(text: str) -> str:
    """"Sorry, finished" -> "sorry, finished" after a comma; names, "I", "Rs" and "MRP" stay as they are."""
    if not text[:1].isupper() or text[1:2].isupper() or text.startswith(("I ", "I'")):
        return text
    first = text.split()[0].strip(",.?!").lower()
    return text[0].lower() + text[1:] if first in ENGLISH_STARTS else text


ENGLISH_STARTS = {"yes", "no", "it's", "it", "we", "sorry", "out", "not", "sold", "that", "price", "for", "only",
                  "earlier", "one", "the", "hold", "owner", "can", "let", "hello", "stock", "anything", "tell", "uh",
                  "see", "actually", "basically", "okay", "why", "normally", "available", "pickup", "shop", "printed",
                  "delivery", "home", "you'll", "your", "my", "market", "company", "there", "speak", "go", "we'll",
                  "we're", "that's", "it'll", "rate", "this", "new", "our", "now", "if", "item", "loose", "single",
                  "full"}


def upper_first(text: str) -> str:
    """A sentence starts with a capital: "paint roller is there" -> "Paint roller is there"; not "iBELL"."""
    return text[0].upper() + text[1:] if text[:1].islower() and not text[1:2].isupper() else text


class Row:
    def __init__(self, gen: "Generator", lang: str, product: Product, kind: str) -> None:
        self.gen, self.rng, self.lang, self.product, self.kind = gen, gen.rng, lang, product, kind
        self.answer = {"in_stock": None, "price_inr": None, "delivery": None, "alternative": None}
        self.parts: list[str] = []
        self.nums: dict[str, int] = {}
        self.main = "p"  # the number whose way of saying was chosen with the price (pick_price)
        self.styles: dict[str, str] = {}
        self.level = self.rng.choices(["clean", "normal", "heavy"], [0.3, 0.62, 0.08])[0]
        if kind == "short":
            self.level = "clean"  # a two-word reply in a rare spelling is too cryptic
        self.price_style = "digits"
        self.flags: set[str] = set()
        self.order: list[str] = []  # the numbers in the order they are said ("m", "p")
        # Marathi-Hindi rows: one main language; openings, closings and fillers may come from the other
        self.home, self.away = ("mr_rom", "hi_rom") if self.rng.random() < 0.5 else ("hi_rom", "mr_rom")
        if lang != "mrhi":
            self.home = self.away = BASE[lang]
        self.base = self.home

    # --- languages
    def bank_lang(self, slot: str) -> str:
        if self.lang == "mrhi":
            if slot in BORROW and self.rng.random() < 0.5:
                self.flags.add("borrowed")
                return self.away
            return self.home
        if self.lang in ("hien", "mren"):
            own = BANKS[self.lang]
            if slot in own and (self.rng.random() < 0.7 or slot not in BANKS[BASE[self.lang]]):
                return self.lang
            return BASE[self.lang]
        return self.lang

    def has(self, slot: str) -> bool:
        return any(slot in BANKS[b] for b in (self.lang if self.lang != "mrhi" else self.home, self.base))

    # --- numbers
    def render(self, key: str, family: str, script: str) -> tuple[str, bool]:
        style = self.price_style if key == self.main else self.gen.other_style(self, key)
        n = self.nums[key]
        if script == "en" or style == "digits" or family == "en":
            if script == "en" and n >= 1000 and key in ("p", "m", "old", "bulk", "ap") and self.rng.random() < 0.3:
                return f"{n:,}", False
            return str(n), False
        if style == "dev_digits":
            return (str(n).translate(DEV_DIGITS), False) if script == "dev" else (str(n), False)
        kinds = {"colloquial": ["colloquial", "formal"], "hundreds": ["hundreds_style", "formal"],
                 "words": ["formal"]}[style]
        for kind in kinds:
            said = self.gen.numbers.words(self.rng, n, family, kind, script)
            if said:
                return said, True
        return (str(n).translate(DEV_DIGITS), False) if script == "dev" and self.price_style == "dev_digits" else (
            str(n), False)

    # --- text
    def say(self, slot: str, **text) -> str:
        blang = self.bank_lang(slot)
        return self.fill(self.rng.choice(BANKS[blang][slot]), blang, **text)

    def fill(self, template: str, blang: str, **text) -> str:
        base = BASE[blang]
        family, script = FAMILY[base], SCRIPT[base]
        vary_lang = "mr" if base == "mr_rom" else "hi"
        out, pos = [], 0
        for m in PLACEHOLDER.finditer(template):
            chunk = template[pos:m.start()]
            out.append(self.gen.spellings.vary(self.rng, chunk, vary_lang, self.level) if script == "rom" else chunk)
            key, join = m[1], m[2]
            if key in NUMERIC:
                value, words = self.render(key, family, script)
                self.order.append(key)
            elif key == "u":
                value, words = UNITS[base][self.product.unit if self.product.unit in UNITS[base] else "piece"], False
            else:
                value, words = text[key], False
            out.append(value)
            if join and not (words and script == "dev"):
                out.append(" ")
            pos = m.end()
        tail = template[pos:]
        out.append(self.gen.spellings.vary(self.rng, tail, vary_lang, self.level) if script == "rom" else tail)
        return "".join(out)

    def add(self, slot: str, **text) -> None:
        self.parts.append(self.say(slot, **text))

    # --- answer helpers
    def set_price(self, style: str | None = None) -> int:
        p = self.gen.pick_price(self, style)
        self.nums["p"] = p
        return p

    def delivery(self, prob: float, value: bool | None = None) -> None:
        if value is None and self.rng.random() >= prob:
            return
        if value is None:
            value = self.rng.random() < 0.55
        self.answer["delivery"] = value
        self.add("del_yes" if value else "del_no")

    def noun_word(self) -> tuple[str, str] | None:
        if not self.product.noun:
            return None
        index = {"hi_rom": 0, "hi_dev": 1, "mr_rom": 2, "mr_dev": 3}.get(self.base)
        return None if index is None else NOUNS[self.product.noun][index]

    def noun_prob(self) -> float:
        """How often the product's noun is said ("बॅटरी संपली" / "बॅटरी आहे"): the same for yes and for no."""
        return 0.65 if self.base in ("mr_dev", "mr_rom") else 0.3

    def say_with_noun(self, prefix: str, fallback: str) -> None:
        """yes_f / no_f ... with the noun ("battery khatam ho gayi", "battery padi hai"), else the plain slot."""
        noun = self.noun_word()
        if noun and self.rng.random() < self.noun_prob() and self.has(f"{prefix}_{noun[1]}"):
            self.add(f"{prefix}_{noun[1]}", noun=noun[0])
            self.flags.add("noun")
        else:
            self.add(fallback)

    def say_no(self) -> None:
        self.say_with_noun("no", "no")

    def alt_text(self, allow_short: bool = True) -> str:
        full, short = self.rng.choice(self.product.alts)
        use_short = bool(short and allow_short and self.rng.random() < 0.45)
        name = short if use_short else full
        if SCRIPT[self.base] == "dev" and self.rng.random() < 0.35:
            name = to_devanagari(name, FAMILY[self.base]) or name
        self.answer["alternative"] = name
        if not use_short:
            return name
        suffix = {"hi_rom": " wala", "mr_rom": " wala", "en": " one", "hi_dev": " वाला", "mr_dev": " वाला"}[self.base]
        return ("the " + name + suffix) if self.base == "en" else name + suffix

    def finish(self) -> str:
        script = SCRIPT[self.base]
        rng = self.rng
        if self.lang == "mrhi" and "borrowed" not in self.flags:
            # code-mixed: at least one word from the other language (a filler or a closing)
            away = BANKS[self.away]
            if "closed" not in self.flags and rng.random() < 0.5:
                self.parts.append(self.fill(rng.choice(away["close"]), self.away))
            else:
                i = rng.randrange(len(self.parts))
                self.parts[i] = rng.choice(away["filler"]) + " " + self.parts[i]
            self.flags.add("borrowed")
        if rng.random() < 0.3 and self.kind not in ("short",):
            self.flags.add("filler")
            fillers = BANKS[self.bank_lang("filler")]["filler"]
            first = 1 if "opened" in self.flags and len(self.parts) > 1 else 0  # not before "hello"
            i = rng.randrange(first, len(self.parts))
            if script == "en":
                self.parts[i] = rng.choice(fillers).capitalize() + ", " + lower_first(self.parts[i])
            else:
                self.parts[i] = rng.choice(fillers) + " " + self.parts[i]
        sep = rng.choice(SEP[script])
        no_punct = script in ("rom", "dev") and rng.random() < 0.12
        parts = list(self.parts)
        said = upper_first(parts[0]) if script == "en" else parts[0]
        for part in parts[1:]:
            if no_punct:
                said += ", " + part  # the commas go below
                continue
            new_sentence = said.endswith(("?", ".", "!", "।"))
            joiner = " " if new_sentence else sep
            if joiner == " " and not new_sentence and self.gen.ends_with_number(said) and \
                    self.gen.starts_with_number(part):
                joiner = ", "
            if script == "en":
                part = upper_first(part) if joiner != ", " else lower_first(part)
            said += joiner + part
        if script == "rom" and rng.random() < 0.35:
            said = said[0].upper() + said[1:]
        if no_punct:
            said = self.gen.strip_punctuation(said)
            if script == "rom":
                said = said.lower()
                if self.answer["alternative"]:
                    self.answer["alternative"] = self.answer["alternative"].lower()
            self.flags.add("no punctuation")
        elif not said.endswith(("?", ".", "!", "।")) and rng.random() < 0.4:
            said += "।" if script == "dev" and sep == "। " else "."
        return said


class Generator:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.numbers = Numbers(json.loads((REAL / "price_words.json").read_text(encoding="utf-8")))
        self.spellings = Spellings(json.loads((REAL / "romanization_variants.json").read_text(encoding="utf-8")))
        self.products = all_products()
        self.old = [p for p in self.products if p.source == "old"]
        self.real = [p for p in self.products if p.source != "old"]
        self.with_mrp = [p for p in self.real if p.mrp]
        self.packs = [p for p in self.products if p.pack]
        # unit vs box: small things only, so that a box stays a shop-sized amount
        self.countable = [p for p in self.products if p.countable and not p.pack and p.high <= 300]

    # --- numbers in the text
    def ends_with_number(self, text: str) -> bool:
        words = re.findall(r"[\w\u0900-\u097F]+", text)
        return bool(words) and (words[-1].isdigit() or words[-1].lower() in self.numbers.tokens)

    def starts_with_number(self, text: str) -> bool:
        words = re.findall(r"[\w\u0900-\u097F]+", text)
        return bool(words) and (words[0].isdigit() or words[0].lower() in self.numbers.tokens)

    def strip_punctuation(self, text: str) -> str:
        """Speech-to-text without punctuation, but a comma stays between two numbers ("610, 9 baje tak")."""
        pieces = re.split(r"([,.।?!]+)(?=\s|$)", text)
        out = pieces[0]
        for i in range(1, len(pieces), 2):
            rest = pieces[i + 1]
            out += ("," if self.ends_with_number(out) and self.starts_with_number(rest) else "") + rest
        return re.sub(r"\s+", " ", out.replace("...", " ")).strip()

    # --- prices
    def family_of(self, row: Row) -> list[str]:
        return [] if row.base == "en" else [FAMILY[row.base]]

    def jitter(self, price: int) -> int:
        """A shop's price for a real product: within 10% of the online price, so a product is not one price."""
        if self.rng.random() < 0.2:
            return price
        p = price * (1 + self.rng.choice([-1, 1]) * self.rng.uniform(0.03, 0.10))
        step = 1 if p < 100 else 5 if p < 1000 else 10
        return max(1, int(round(p / step) * step))

    def choose_style(self, row: Row) -> str:
        base = row.base
        if base == "en":
            return "digits"
        r = self.rng.random()
        if SCRIPT[base] == "dev":
            return "digits" if r < 0.34 else "dev_digits" if r < 0.56 else "words" if r < 0.76 else (
                "colloquial" if r < 0.9 else "hundreds")
        return "digits" if r < 0.5 else "words" if r < 0.74 else "colloquial" if r < 0.9 else "hundreds"

    def pick_price(self, row: Row, style: str | None = None) -> int:
        product = row.product
        style = style or self.choose_style(row)
        if product.price is not None:
            p = self.jitter(product.price)
            if style == "digits" and self.rng.random() < 0.5:
                p = on_grid(p) if p > 100 else p
        elif style == "hundreds" and product.low <= 4550 and product.high >= 1100:
            p = make_price(self.rng, max(product.low, 1100), min(product.high, 4550))
        else:
            p = make_price(self.rng, product.low, product.high)
        fams = self.family_of(row)
        if style in ("words", "colloquial", "hundreds", "dev_digits") and fams:
            p = on_grid(p)
        if style == "colloquial" and fams:
            near = [c for c in range(max(1, int(p * 0.85)), int(p * 1.15) + 1)
                    if all(self.numbers.has(c, f, "colloquial") for f in fams)]
            if near:
                p = min(near, key=lambda c: (abs(c - p), c))
            else:
                style = "words"
        hundreds = bool(fams) and all(self.numbers.has(p, f, "hundreds_style") for f in fams)
        if style == "hundreds" and not hundreds:
            style = "words"
        elif style in ("words", "colloquial") and hundreds and self.rng.random() < 0.7:
            style = "hundreds"
        row.price_style = style if fams else "digits"
        return p

    def other_style(self, row: Row, key: str) -> str:
        cache = row.styles
        if key not in cache:
            if row.price_style in ("words", "colloquial", "hundreds") and self.rng.random() < 0.6:
                cache[key] = "words"
            elif row.price_style == "dev_digits":
                cache[key] = "dev_digits"
            elif key in COUNTS and self.rng.random() < 0.35:
                cache[key] = "words"
            else:
                cache[key] = "digits"
            n = row.nums[key]
            if cache[key] == "words" and key not in COUNTS:
                g = on_grid(n)
                if g == row.nums.get(row.main):
                    cache[key] = "digits"
                else:
                    row.nums[key] = g
        return cache[key]

    def near(self, p: int, lo: float, hi: float, avoid: set[int]) -> int:
        for _ in range(50):
            n = int(p * self.rng.uniform(lo, hi))
            n = n if n < 100 else int(round(n / 5) * 5) if n < 1000 else int(round(n / 10) * 10)
            if n > 0 and n not in avoid:
                return n
        return p + 7

    # --- one row
    def product_for(self, kind: str, lang: str) -> Product:
        if kind == "mrp_vs_offered":
            pool = self.with_mrp
        elif kind == "pack_price":
            pool = self.packs
        elif kind == "unit_vs_bulk":
            pool = self.countable
        else:
            pool = self.old if self.rng.random() < 0.42 else self.real
        if kind == "delivery_fee":
            pool = [p for p in pool if p.low >= 40]  # nobody pays a delivery fee for a 15-rupee tape
        if lang in ("mr_dev", "mr_rom") and self.rng.random() < 0.45:
            # Marathi nouns change with gender: बॅटरी संपली / पडलीये, पंखा संपला / पडलाय, ऑइल संपलं / पडलंय.
            # More products with a noun, feminine ones most, for every kind alike (so the noun says nothing
            # about stock).
            with_noun = [p for p in pool if p.noun]
            feminine = [p for p in with_noun if NOUNS[p.noun][3][1] == "f"]
            pool = (feminine if feminine and self.rng.random() < 0.5 else with_noun) or pool
        return self.rng.choice(pool)

    def row(self, kind: str) -> tuple[dict, Row]:
        weights = LANG_WEIGHTS  # the same for every kind: the language says nothing about the answer
        lang = self.rng.choices(list(weights), weights=list(weights.values()))[0]
        product = self.product_for(kind, lang)
        row = Row(self, lang, product, kind)
        build = getattr(self, "b_" + kind)
        build(row)
        said = row.finish()
        return {"product": product.name, "said": said, "answer": row.answer}, row

    def opening(self, row: Row, prob: float = 0.3) -> None:
        if self.rng.random() < prob:
            row.add("open")
            row.flags.add("opened")

    def closing(self, row: Row, prob: float = 0.12) -> None:
        if self.rng.random() < prob:
            row.add("close")
            row.flags.add("closed")

    def yes(self, row: Row) -> None:
        row.answer["in_stock"] = True
        row.say_with_noun("yes", "yes")

    # in stock, one price
    def b_clear_price(self, row: Row) -> None:
        self.opening(row)
        row.set_price()
        row.answer["price_inr"] = row.nums["p"]
        if self.rng.random() < 0.25:
            row.add("price")
            self.yes(row)
        else:
            self.yes(row)
            row.add("price")
        row.delivery(0.42)
        self.closing(row)

    def b_in_stock_no_price(self, row: Row) -> None:
        self.opening(row)
        if self.rng.random() < 0.35:
            row.answer["in_stock"] = True
            row.nums["q"] = self.rng.randint(2, 25)
            row.add("yes_qty")
        else:
            self.yes(row)
        if self.rng.random() < 0.5:
            row.add("noprice")
        row.delivery(0.5)
        self.closing(row, 0.2)

    # in stock, several numbers
    def either(self, row: Row, slot: str) -> str:
        """slot or slot_rev, half and half: the asked product's price is said first or last."""
        return slot + "_rev" if row.has(slot + "_rev") and self.rng.random() < 0.5 else slot

    def b_mrp_vs_offered(self, row: Row) -> None:
        self.opening(row, 0.2)
        product = row.product
        p = row.set_price()
        m = product.mrp * p / product.price  # the MRP moves with the (jittered) price
        m = int(round(m / 5) * 5) if m < 1000 else int(round(m / 10) * 10)
        row.nums["m"] = m if m > p * 1.03 else on_grid(int(p * 1.2))
        row.answer["price_inr"] = p
        self.yes(row)
        row.add(self.either(row, "mrp"))
        row.delivery(0.35)

    def b_old_vs_new(self, row: Row) -> None:
        self.opening(row, 0.2)
        p = row.set_price()
        down = self.rng.random() < 0.25 and row.has("old_down")
        row.nums["old"] = self.near(p, 1.08, 1.25, {p}) if down else self.near(p, 0.75, 0.93, {p})
        row.answer["price_inr"] = p
        self.yes(row)
        if row.has("old_rev") and self.rng.random() < 0.5:
            row.add("old_rev")  # the new price first; old_rev works for a rise and a fall
        else:
            row.add("old_down" if down else "old_new")
        row.delivery(0.35)

    def b_unit_vs_bulk(self, row: Row) -> None:
        p = row.set_price()
        n = self.rng.choice([n for n in (6, 10, 10, 12, 20, 25, 50) if p * n <= 5000] or [5])
        row.nums["bn"] = n  # the box size is said: "10 ka dabba 900"
        row.nums["bulk"] = self.near(p * n, 0.82, 0.93, {p, n})
        row.nums["bu"] = self.near(p, 0.8, 0.94, {p, n})
        row.answer["price_inr"] = p
        self.yes(row)
        row.add(self.either(row, "unit_bulk"))
        row.delivery(0.35)

    def b_pack_price(self, row: Row) -> None:
        p = row.set_price()
        row.nums["unit"] = max(1, self.near(p / row.product.pack, 1.1, 1.4, {p}))
        row.answer["price_inr"] = p
        self.yes(row)
        row.add(self.either(row, "pack"))
        row.delivery(0.35)

    def b_gst(self, row: Row) -> None:
        self.opening(row, 0.2)
        p = row.set_price()
        row.answer["price_inr"] = p
        self.yes(row)
        row.add("gst")
        row.delivery(0.4)

    def b_delivery_fee(self, row: Row) -> None:
        p = row.set_price()
        # at most half the price of the item ("teflon tape 30, delivery 50" does not happen)
        row.nums["fee"] = self.rng.choice([f for f in (20, 30, 40, 50, 50, 60, 80, 100) if f <= p // 2 and f != p]
                                          or [10])
        row.answer["price_inr"] = p
        self.yes(row)
        if self.rng.random() < 0.3:
            row.add("fee")
            row.add("item_price")  # "maal ka 400, delivery alag": not "It comes to 400", which sounds like a total
        else:
            row.add("price")
            row.add("fee")
        row.answer["delivery"] = True

    def b_quantity(self, row: Row) -> None:
        self.opening(row, 0.2)
        p = row.set_price()
        row.nums["q"] = self.rng.choice([x for x in range(2, 31) if x != p])
        row.answer["in_stock"] = True
        row.answer["price_inr"] = p
        if self.rng.random() < 0.6:
            row.add("yes_qty")
            row.add("price")
        else:
            self.yes(row)
            row.add("price")
            row.add("yes_qty")
        row.delivery(0.35)

    def b_timing(self, row: Row) -> None:
        p = row.set_price()
        row.nums["h"] = self.rng.choice([7, 8, 8, 9, 9, 10])
        row.answer["price_inr"] = p
        self.yes(row)
        row.add("price")
        if self.rng.random() < 0.5:
            row.delivery(0.4)
            row.add("timing")
        else:
            row.add("timing")
            row.delivery(0.4)

    def b_other_variant_price(self, row: Row) -> None:
        p = row.set_price()
        row.nums["ap"] = self.near(p, 0.7, 1.4, {p})
        row.answer["price_inr"] = p
        self.yes(row)
        alt = self.rng.choice(row.product.alts)[0]
        if row.has("other_first") and self.rng.random() < 0.5:
            row.add("other_first", alt=alt)  # the other product's price first, then the asked one's
        else:
            row.add("price")
            row.add("other", alt=alt)
        row.delivery(0.3)

    # not in stock
    def b_no_stock(self, row: Row) -> None:
        self.opening(row, 0.3)
        row.answer["in_stock"] = False
        row.say_no()
        if self.rng.random() < 0.15:
            row.add("chatter")
        self.closing(row, 0.15)

    def b_no_stock_alt(self, row: Row) -> None:
        self.opening(row, 0.25)
        row.answer["in_stock"] = False
        if row.product.ask and self.rng.random() < 0.3:
            alt = row.alt_text()
            row.add("alt_neg", ask=row.product.ask, alt=alt)
        else:
            row.say_no()
            row.add("alt", alt=row.alt_text())
        row.delivery(0.25)

    def b_no_stock_alt_price(self, row: Row) -> None:
        self.opening(row, 0.2)
        row.answer["in_stock"] = False
        row.say_no()
        row.main = "ap"  # a price is said, but it is the alternative's price: price_inr stays null
        row.nums["ap"] = self.pick_price(row)
        row.add("alt_price", alt=row.alt_text())
        row.delivery(0.2)

    def b_comes_later(self, row: Row) -> None:
        self.opening(row, 0.25)
        row.answer["in_stock"] = False
        if row.lang in ("mr_dev", "mr_rom") and row.noun_word() and self.rng.random() < 0.6:
            row.say_no()  # "बॅटरी संपली, उद्या येईल"
        row.add("tomorrow")
        if self.rng.random() < 0.25:
            row.set_price()
            row.add("price")  # a price is said, but the product is not there now: price_inr stays null
        row.delivery(0.3)

    def b_no_stock_price_said(self, row: Row) -> None:
        self.opening(row, 0.2)
        row.answer["in_stock"] = False
        row.main = "x"
        row.nums["x"] = self.pick_price(row)
        row.add("nostock_price")
        if self.rng.random() < 0.4:
            row.add("tomorrow")

    def b_can_order(self, row: Row) -> None:
        self.opening(row, 0.2)
        row.answer["in_stock"] = False
        row.add("order")
        row.delivery(0.2)

    # not sure
    def b_unclear(self, row: Row) -> None:
        self.opening(row, 0.35)
        noun = row.noun_word()
        if noun and self.rng.random() < row.noun_prob() / 2 and row.has("unclear_n"):
            row.add("unclear_n", noun=noun[0])  # "battery hai ki nahi, dekh ke batata hun"
            row.flags.add("noun")
        else:
            row.add("unclear")
        if self.rng.random() < 0.2:
            row.add("chatter")
        # no delivery: "owner is not here, call later. Home delivery is available" does not hang together

    def b_unclear_price_said(self, row: Row) -> None:
        self.opening(row, 0.25)
        row.main = "x"
        row.nums["x"] = self.pick_price(row)
        row.add("unclear_price")

    # self-corrections and negation traps
    def b_price_correction(self, row: Row) -> None:
        p = row.set_price()
        row.nums["w"] = self.near(p, 0.75, 1.3, {p})
        row.answer["price_inr"] = p
        self.yes(row)
        row.add("wrong_rev" if row.has("wrong_rev") and self.rng.random() < 0.35 else "wrong")
        row.delivery(0.3)

    def b_stock_correction_yes(self, row: Row) -> None:
        row.answer["in_stock"] = True
        row.add("corr_yes")
        if self.rng.random() < 0.55:
            row.answer["price_inr"] = row.set_price()
            row.add("price")
        row.delivery(0.3)

    def b_stock_correction_no(self, row: Row) -> None:
        row.answer["in_stock"] = False
        row.add("corr_no")
        if self.rng.random() < 0.25:
            row.add("alt", alt=row.alt_text())
            row.delivery(0.3)  # delivery only of the alternative that is offered

    def b_negation_trap_yes(self, row: Row) -> None:
        row.answer["in_stock"] = True
        row.add("trap_yes")
        if self.rng.random() < 0.6:
            row.answer["price_inr"] = row.set_price()
            row.add("price")
        row.delivery(0.3)

    def b_negation_trap_no(self, row: Row) -> None:
        row.answer["in_stock"] = False
        row.add("trap_no")
        if self.rng.random() < 0.25:
            row.add("alt", alt=row.alt_text())

    def b_delivery_trap(self, row: Row) -> None:
        self.yes(row)
        if self.rng.random() < 0.5:
            row.answer["price_inr"] = row.set_price()
            row.add("price")
        value = self.rng.random() < 0.6
        slot = "trap_del_yes" if value else "trap_del_no"
        if not row.has(slot):
            slot, value = "trap_del_yes", True
        row.answer["delivery"] = value
        row.add(slot)

    # long and short replies
    def b_rambling(self, row: Row) -> None:
        inner = self.rng.choice(["clear_price", "clear_price", "mrp_vs_offered", "old_vs_new", "delivery_fee",
                                 "no_stock", "no_stock_alt", "no_stock_alt_price", "comes_later", "unclear",
                                 "price_correction", "quantity"])
        if inner == "mrp_vs_offered" and not row.product.mrp:
            inner = "clear_price"
        getattr(self, "b_" + inner)(row)
        row.flags.add("inner:" + inner)
        blang = row.bank_lang("chatter")
        used = [t for t in BANKS[blang]["chatter"] if row.fill(t, blang) in row.parts]
        fresh = [t for t in BANKS[blang]["chatter"] if t not in used]
        # chatter goes after the greeting and before the closing ("aur kuch?")
        first = 1 if "opened" in row.flags else 0
        last = len(row.parts) - (1 if "closed" in row.flags else 0)
        for template in self.rng.sample(fresh, min(len(fresh), self.rng.randint(2, 4))):
            row.parts.insert(self.rng.randint(first, last), row.fill(template, blang))
            last += 1
        if self.rng.random() < 0.4 and "opened" not in row.flags:
            row.parts.insert(0, row.say("open"))
            row.flags.add("opened")
        if self.rng.random() < 0.3 and "closed" not in row.flags:
            row.add("close")
            row.flags.add("closed")

    def b_short(self, row: Row) -> None:
        r = self.rng.random()
        if r < 0.4:
            self.yes(row)
        elif r < 0.8:
            row.answer["in_stock"] = False
            row.say_no()
        else:
            row.add("unclear")
        if self.rng.random() < 0.5:
            blang = row.bank_lang("address")
            word = self.rng.choice(BANKS[blang]["address"])
            row.parts[-1] += ", " + word if row.base == "en" else " " + word


def make(n: int = N_ROWS, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    gen = Generator(seed)
    plan = [kind for kind, count in PLAN.items() for _ in range(count)]
    assert len(plan) == n, len(plan)
    gen.rng.shuffle(plan)
    rows, meta, seen = [], [], set()
    for kind in plan:
        for _ in range(50):
            ex, row = gen.row(kind)
            if ex["said"] not in seen:
                break
        seen.add(ex["said"])
        rows.append(ex)
        script = "romanized" if row.lang == "mrhi" else {"rom": "romanized", "dev": "Devanagari", "en": "English"}[
            SCRIPT[BASE[row.lang]]]
        style = None
        if ex["answer"]["price_inr"] is not None:
            style = {"dev_digits": "Devanagari digits", "hundreds": "words, hundreds style (बारह सौ / बाराशे)",
                     "colloquial": "words, colloquial (डेढ़ सौ / दीडशे)", "words": "words"}.get(row.price_style, "digits")
            style += f" in {script} text" if row.price_style != "dev_digits" else ""
        meta.append({"kind": kind, "lang": row.lang, "style": style, "flags": sorted(row.flags),
                     "source": row.product.source, "order": row.order, "nums": row.nums})
    return rows, meta


# ---------------------------------------------------------------------------------------------
# Stats and the leakage check against the held-out sets.

def words_of(text: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097F]+", nfc(text).lower())


def leakage(rows: list[dict], n: int = 8) -> dict:
    """Train rows whose "said" is identical to, or shares a run of n words with, a held-out row."""
    def grams(text):
        w = words_of(text)
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

    train_said = {r["said"] for r in rows}
    train_grams = set().union(*(grams(s) for s in train_said))
    out = {}
    for name in ("test.jsonl", "hard_test.jsonl"):
        held = [json.loads(line) for line in (HERE / "data" / name).read_text(encoding="utf-8").splitlines()]
        same = sum(1 for r in held if r["said"] in train_said)
        shared = [r["said"] for r in held if grams(r["said"]) & train_grams]
        out[name] = {"identical_said": same, f"rows_sharing_{n}_word_run": len(shared)}
    return out


def script_of(text: str) -> str:
    dev = bool(DEVANAGARI.search(text))
    lat = bool(re.search(r"[A-Za-z]", text))
    return "Devanagari + Latin" if dev and lat else "Devanagari" if dev else "Latin"


def stats(rows: list[dict], meta: list[dict]) -> dict:
    n = len(rows)

    def share(counter: Counter) -> dict:
        return {k: f"{v} ({v / n:.1%})" for k, v in counter.most_common()}

    def within(counter: Counter) -> dict:
        total = sum(counter.values())
        return {k: f"{v} ({v / total:.0%})" for k, v in counter.most_common()}

    group_of = {kind: g for g, kinds in GROUPS.items() for kind in kinds}
    first: dict[str, Counter] = {}  # several prices in one reply: is the asked product's price said first?
    for m in meta:
        order = [k for k in m["order"] if k not in COUNTS]
        if "p" in order and len(order) > 1:
            first.setdefault(m["kind"], Counter())["first" if order[0] == "p" else "later"] += 1
    prices = [r["answer"]["price_inr"] for r in rows if r["answer"]["price_inr"] is not None]
    bands = Counter("<100" if p < 100 else "100-999" if p < 1000 else "1000-4999" if p < 5000 else "5000+"
                    for p in prices)
    return {
        "rows": n,
        "categories": share(Counter(group_of[m["kind"]] for m in meta)),
        "kinds": share(Counter(m["kind"] for m in meta)),
        "in_stock": share(Counter(str(r["answer"]["in_stock"]) for r in rows)),
        "delivery": share(Counter(str(r["answer"]["delivery"]) for r in rows)),
        "alternative": share(Counter("null" if r["answer"]["alternative"] is None else "text" for r in rows)),
        "price_inr": {"null": share(Counter(r["answer"]["price_inr"] is None for r in rows)).get(True),
                      "bands": dict(bands),
                      "how_said": dict(Counter(m["style"] for m in meta if m["style"]).most_common())},
        "language": share(Counter(LANG_NAMES[m["lang"]] for m in meta)),
        "language_detail": share(Counter(m["lang"] for m in meta)),
        "script": share(Counter(script_of(r["said"]) for r in rows)),
        "marathi_not_in_stock_forms": {
            f: sum(1 for r in rows if f in r["said"] and r["answer"]["in_stock"] is False)
            for f in ("संपली", "संपलं", "संपला", "नाहीये", "शिल्लक नाही", "sampli", "sampla", "nahiye")},
        # shortcuts a model could learn instead of reading: they should look like the whole file
        "in_stock_when_the_noun_is_said": within(Counter(str(r["answer"]["in_stock"])
                                                         for r, m in zip(rows, meta, strict=True)
                                                         if "noun" in m["flags"])),
        "asked_price_said_first_or_later": {k: within(v) for k, v in sorted(first.items())},
        "with_filler_word": sum("filler" in m["flags"] for m in meta),
        "no_punctuation": sum("no punctuation" in m["flags"] for m in meta),
        "product_source": share(Counter(m["source"] for m in meta)),
        "distinct_products": len({r["product"] for r in rows}),
        "leakage": leakage(rows),
    }


# =============================================================================================
# Training set v4 (data/train_v4.jsonl): the same kind of calls, written the way a speech-to-text model
# (Whisper) writes them. Whisper writes most Hindi and Hinglish speech in Devanagari, English words included
# ("हाँ है, दो सौ बीस का, डिलीवरी नहीं"), with small errors. v2 (data/train.jsonl) is made by the code above and
# does not change; v4 reuses its products, prices, number words and spellings, and adds:
#   - the owner's language and script mix (MIX_V4): Hindi, Hinglish and Marathi, in Devanagari and in Latin letters;
#   - speech-to-text noise on NOISE_SHARE_V4 of the Devanagari rows, never on the words the answer depends on;
#   - spoken prices with सवा / साढ़े / पौने / डेढ़ / ढाई (सव्वा / साडे / पावणे / दीड / अडीच) + सौ / हज़ार, off the
#     price_words.json grid too (पौने सत्रह सौ = 1675), discounts, corrections, unit vs box;
#   - delivery said directly and indirectly ("ladka ghar pe de jayega", "पोरगा घरी आणून देईल", "aake le jao");
#   - one rule for `alternative`: brand + size/type + item, exactly as the shopkeeper said it, in the script used;
#   - unclear, call-back, wrong-number and "pata nahi" replies: all four fields null.

N_ROWS_V4 = 6000
SEED_V4 = 2604
NOISE_SHARE_V4 = 0.30  # of the Devanagari rows
# The owner's mix: class -> (share of all rows, the languages in it with their weights)
MIX_V4 = {
    "Hindi, Devanagari": (0.25, {"hi_dev": 1.0}),
    "Hinglish, Devanagari": (0.25, {"hx_dev": 1.0}),
    "Hinglish, Latin": (0.20, {"hi_rom": 0.65, "hien": 0.35}),
    "Marathi, Devanagari": (0.20, {"mr_dev": 0.8, "mx_dev": 0.2}),
    "Marathi, Latin": (0.10, {"mr_rom": 0.7, "mren": 0.3}),
}
LANG_NAMES_V4 = {**LANG_NAMES, "hx_dev": "Hinglish in Devanagari (as Whisper writes it)",
                 "mx_dev": "Marathi-English in Devanagari", "hi_dev": "Hindi in Devanagari",
                 "hi_rom": "romanized Hindi/Hinglish", "hien": "code-mixed Hinglish, Latin",
                 "mren": "code-mixed Marathi-English, Latin"}
BASE["hx_dev"] = "hi_dev"  # code-mixed banks fall back to their base bank (as hien -> hi_rom)
BASE["mx_dev"] = "mr_dev"
OWN_BANK_V4 = ("hien", "mren", "hx_dev", "mx_dev")

# v4's banks: a copy of v2's (which must not change, or train.jsonl would), with more phrasings.
BANKS4: dict[str, dict[str, list[str]]] = {lang: {slot: list(v) for slot, v in bank.items()}
                                           for lang, bank in BANKS.items()}


def extend_v4(lang: str, **slots: list[str]) -> None:
    bank = BANKS4.setdefault(lang, {})
    for slot, items in slots.items():
        bank[slot] = bank.get(slot, []) + items


extend_v4(
    "hi_rom",
    del_yes=["ladka ghar pe de jayega", "ladka ghar pe chhod dega", "ghar tak pahuncha denge", "courier kar denge",
             "apna aadmi de aayega", "address bhej do, ladka de aayega", "shaam tak ghar pe aa jayega",
             "aapke ghar tak bhijwa denge", "delivery boy hai, bhej denge", "pahuncha dunga, fikar mat karo"],
    del_no=["pick up karna padega", "aake le jao", "khud aake uthana padega", "dukaan se le jana padega",
            "bhejne ka intezaam nahi hai", "abhi delivery band hai", "counter pe aake le lo",
            "ladka nahi hai bhejne ko, aake le jao"],
    discount=["MRP {m} hai par {p} laga denge", "{m} ka hai, aapke liye {p}", "{m} likha hai, discount karke {p}",
              "company ka rate {m} hai, main {p} lunga", "list mein {m} hai, aapko {p} padega"],
    discount_rev=["{p} laga denge, waise {m} ka hai", "aapse {p} lenge, likha {m} hai",
                  "{p} de dena, MRP toh {m} hai"],
    wrong_number=["galat number hai", "ye dukaan nahi hai, ghar ka number hai", "kaun? yahan koi dukaan nahi hai",
                  "aapne galat number lagaya hai", "kisse baat karni hai? ye wo dukaan nahi hai"],
    unclear=["pata nahi bhai", "abhi bata nahi sakta, baad mein call karo", "kal phone karna",
             "main naya hun, mujhe nahi pata, seth se puchho", "ghante bhar baad call karo"],
    wrong=["{w} ka... nahi nahi, galat bola, {p} ka hai", "{w}... arre nahi, {p}"],
)
extend_v4(
    "hien",
    del_yes=["boy ghar pe de jayega", "courier kar denge sir", "ghar tak deliver kar denge"],
    del_no=["pick up karna padega", "self pickup hai, delivery nahi"],
    discount=["MRP {m} hai but {p} laga denge", "list price {m}, discount ke baad {p}"],
    discount_rev=["aapko {p}, MRP {m} hai"],
    wrong_number=["wrong number hai sir", "sorry, wrong number"],
)
extend_v4(
    "mr_rom",
    del_yes=["porga ghari aanun deil", "porga ghari sodun deil", "gharpoch pohochvun deu", "courier karto",
             "manus pathavto ghari"],
    del_no=["yeun gheun ja", "pick up karava lagel", "dukanatun nyava lagel", "pathavaychi soy nahi"],
    discount=["MRP {m} aahe, pan {p} la lavto", "{m} cha aahe, tumchyasathi {p}", "{m} lihila aahe, discount karun {p}"],
    discount_rev=["{p} la lavto, MRP {m} aahe"],
    wrong_number=["chukicha number aahe", "wrong number aahe", "he dukan nahi, gharcha number aahe"],
    unclear=["mahit nahi ho", "udya phone kara", "mi navin aahe, mala mahit nahi"],
)
extend_v4(
    "mren",
    del_yes=["boy ghari deliver karel", "courier karto sir"],
    del_no=["self pickup aahe"],
    discount=["MRP {m} aahe but {p} la deu"],
    wrong_number=["wrong number aahe sir"],
)
extend_v4(
    "hi_dev",
    del_yes=["लड़का घर पे दे जाएगा", "लड़का छोड़ देगा घर पर", "घर तक पहुँचा देंगे", "कूरियर कर देंगे",
             "अपना आदमी दे आएगा", "पता भेज दो, लड़का दे आएगा", "शाम तक घर पहुँच जाएगा", "आपके घर तक भिजवा देंगे"],
    del_no=["आकर ले जाओ", "खुद आकर उठाना पड़ेगा", "दुकान से ले जाना पड़ेगा", "भेजने का इंतज़ाम नहीं है",
            "अभी डिलीवरी बंद है", "लड़का नहीं है भेजने को, आकर ले जाइए"],
    discount=["MRP {m} है पर {p} लगा देंगे", "{m} का है, आपके लिए {p}", "{m} लिखा है, छूट देकर {p}",
              "कंपनी का रेट {m} है, मैं {p} लूँगा", "लिस्ट में {m} है, आपको {p} पड़ेगा"],
    discount_rev=["{p} लगा देंगे, वैसे {m} का है", "आपसे {p} लेंगे, लिखा {m} है"],
    wrong_number=["गलत नंबर है", "ये दुकान नहीं है, घर का नंबर है", "किससे बात करनी है? यहाँ कोई दुकान नहीं है",
                  "आपने गलत नंबर मिलाया है"],
    unclear=["पता नहीं भाई", "अभी नहीं बता सकता, बाद में फ़ोन करो", "कल फ़ोन करना", "मैं नया हूँ, मुझे नहीं पता",
             "घंटे भर बाद कॉल करो"],
    wrong=["{w} का... नहीं नहीं, गलत बोला, {p} का है"],
    filler=["अं", "हम्म", "देखो", "वो"],
)
extend_v4(
    "mr_dev",
    del_yes=["पोरगा घरी आणून देईल", "पोरगा घरी सोडून देईल", "घरपोच पोहोचवून देऊ", "कुरिअर करतो",
             "माणूस पाठवून देतो घरी", "संध्याकाळी घरी पोहोचेल"],
    del_no=["येऊन घेऊन जा", "पिकअप करावा लागेल", "दुकानातून न्यावं लागेल", "पाठवायची सोय नाही"],
    discount=["एमआरपी {m} आहे, पण {p}_ला लावतो", "{m}चं आहे, तुमच्यासाठी {p}", "छापील {m}, डिस्काउंट करून {p}"],
    discount_rev=["{p}_ला लावतो, एमआरपी {m} आहे", "तुम्हाला {p}, छापील किंमत {m}"],
    wrong_number=["चुकीचा नंबर आहे", "रॉंग नंबर", "हे दुकान नाही हो, घरचा नंबर आहे"],
    unclear=["माहीत नाही हो", "उद्या फोन करा", "मी नवीन आहे, मला माहीत नाही", "तासाभराने कॉल बॅक करतो"],
    filler=["अं", "हां", "तर"],
)
# Hinglish as Whisper writes it: in Devanagari, English words too (अवेलेबल, डिलीवरी, प्राइस); Latin MRP / GST.
BANKS4["hx_dev"] = {
    "open": ["हाँ बोलिए सर", "हेलो, हाँ बोलो", "यस सर, बोलिए", "हाँ जी सर, बताइए", "हेलो? हाँ, बोलो भाई"],
    "yes": ["हाँ सर, अवेलेबल है", "स्टॉक में है सर", "यस, अवेलेबल है", "हाँ, रेडी स्टॉक है", "हाँ है, कोई प्रॉब्लम नहीं",
            "हाँ सर, वो मॉडल अवेलेबल है", "हाँ है", "है भाई, मिल जाएगा", "हाँ जी, स्टॉक में पड़ा है"],
    "yes_qty": ["{q} {u} अवेलेबल है अभी", "स्टॉक में सिर्फ {q} {u} है", "हाँ, {q} {u} पड़े हैं"],
    "no": ["सॉरी सर, अभी आउट ऑफ स्टॉक है", "स्टॉक नहीं है अभी", "वो मॉडल अवेलेबल नहीं है", "अभी स्टॉक फिनिश हो गया",
           "नॉट अवेलेबल सर", "वो आइटम हम नहीं रखते", "नहीं है भाई, खत्म हो गया"],
    "tomorrow": ["अभी स्टॉक नहीं है, कल तक आ जाएगा", "अभी आउट ऑफ स्टॉक है, नया स्टॉक कल आ रहा है",
                 "अभी अवेलेबल नहीं है, ऑर्डर प्लेस किया है, मंडे तक आएगा"],
    "order": ["अभी नहीं है, ऑर्डर पे मंगवा देंगे", "नहीं है, बोलो तो मंगवा देता हूँ, दो दिन लगेंगे"],
    "unclear": ["सर एक मिनट, चेक करके बताता हूँ", "ओनर अभी नहीं हैं, बाद में कॉल कीजिए",
                "अभी कन्फर्म नहीं कर सकता, गोडाउन चेक करना पड़ेगा", "प्लीज़ थोड़ी देर बाद कॉल करो",
                "पता नहीं भाई, बाद में कॉल करो", "अभी बिज़ी हूँ, कॉल बैक करता हूँ"],
    "wrong_number": ["रॉन्ग नंबर है", "गलत नंबर लगाया है आपने", "ये दुकान का नंबर नहीं है", "सॉरी, रॉन्ग नंबर"],
    "price": ["प्राइस {p} है", "{p} रुपीज़ है", "रेट {p} है सर", "{p} फाइनल प्राइस है", "आपके लिए {p}",
              "बेस्ट प्राइस {p}", "{p} का पड़ेगा", "{p} लगेगा", "{p} का है"],
    "mrp": ["MRP {m} है, बट आपको {p} में दे देंगे", "एमआरपी {m} है, डिस्काउंट के बाद {p} पड़ेगा",
            "प्रिंटेड एमआरपी {m}, अपना प्राइस {p}"],
    "mrp_rev": ["अपना प्राइस {p}, एमआरपी {m} है", "आपको {p} में मिलेगा, MRP {m} है"],
    "discount": ["एमआरपी {m} है पर {p} लगा देंगे", "{m} का है, आपके लिए {p}", "{m} लिखा है, डिस्काउंट करके {p}",
                 "लिस्ट प्राइस {m}, आपको {p} पड़ेगा"],
    "discount_rev": ["{p} लगा देंगे, वैसे {m} का है", "आपको {p}, MRP तो {m} है"],
    "old_new": ["प्राइस इनक्रीज़ हो गया, पहले {old} था, अब {p}", "लास्ट मंथ {old} था, अब {p} है"],
    "old_rev": ["अब प्राइस {p} है, पहले {old} था", "अब {p} है, लास्ट मंथ {old} था"],
    "unit_bulk": ["सिंगल पीस {p}, {bn} का फुल बॉक्स {bulk}", "पर पीस {p}, {bn} लो तो {bu} लगेगा",
                  "{p} का एक, {bn} का डब्बा {bulk} का"],
    "unit_bulk_rev": ["{bn} का फुल बॉक्स {bulk}, सिंगल पीस {p}", "डब्बा {bulk} का है, {bn} पीस वाला, एक का {p}"],
    "pack": ["फुल पैकेट {p}, लूज़ पीस {unit} का"],
    "pack_rev": ["लूज़ पीस {unit} का, फुल पैकेट {p}"],
    "gst": ["{p} प्लस जीएसटी", "{p}, जीएसटी एक्स्ट्रा", "{p} इन्क्लूसिव ऑफ जीएसटी"],
    "fee": ["डिलीवरी हो जाएगी, {fee} रुपीज़ चार्ज एक्स्ट्रा", "होम डिलीवरी अवेलेबल है, {fee} चार्ज",
            "लड़का दे आएगा, {fee} रुपये लगेंगे"],
    "item_price": ["आइटम प्राइस {p}, डिलीवरी चार्ज अलग", "प्रोडक्ट का {p}, डिलीवरी एक्स्ट्रा"],
    "timing": ["शॉप {h} बजे तक ओपन है"],
    "del_yes": ["होम डिलीवरी अवेलेबल है", "डिलीवरी हो जाएगी सर", "सेम डे डिलीवरी कर देंगे", "फ्री डिलीवरी है",
                "लड़का घर पे दे जाएगा", "कूरियर कर देंगे", "बॉय भेज दूंगा", "घर तक पहुँचा देंगे",
                "अपना लड़का छोड़ देगा घर पे", "एड्रेस भेजो, डिलीवर कर देंगे"],
    "del_no": ["डिलीवरी नहीं होती, पिकअप करना पड़ेगा", "सॉरी, डिलीवरी सर्विस नहीं है", "आपको शॉप से पिकअप करना होगा",
               "आके ले जाओ", "खुद आके ले जाना पड़ेगा", "डिलीवरी बंद है अभी, दुकान से ले लो"],
    "trap_yes": ["नो नो, अवेलेबल है", "आउट ऑफ स्टॉक नहीं है, अवेलेबल है"],
    "trap_del_yes": ["डिलीवरी की टेंशन मत लो, भेज देंगे"],
    "trap_del_no": ["भेज देता, पर आज बॉय छुट्टी पे है, आके ले जाओ"],
    "alt": ["{alt} अवेलेबल है, चलेगा?", "{alt} है, सेम क्वालिटी", "उसकी जगह {alt} ले सकते हो", "{alt} पड़ा है, वो ले लो"],
    "alt_neg": ["{ask} नहीं है, {alt} है", "{ask} वाला फिनिश है, {alt} अवेलेबल है"],
    "alt_price": ["{alt} अवेलेबल है, उसका प्राइस {ap}", "{alt} है, {ap} का"],
    "chatter": ["सर आज बहुत बिज़ी डे है", "नेटवर्क इशू है, आवाज़ ब्रेक हो रही है", "सीज़न टाइम है ना, डिमांड ज़्यादा है",
                "एक कस्टमर खड़ा है, होल्ड करो"],
    "filler": ["मतलब", "देखो", "अं", "वो क्या है", "बेसिकली"],
    "close": ["और कुछ?", "बताओ क्या करना है", "ओके?", "और कुछ चाहिए सर?"],
    "wrong": ["{w}... नहीं नहीं, {p} है", "{w}, सॉरी सॉरी, {p}"],
    "wrong_rev": ["{p} है, {w} नहीं"],
    "noprice": ["रेट अभी पता नहीं, ओनर से पूछना पड़ेगा", "प्राइस कन्फर्म करके बताता हूँ"],
    "address": ["सर", "भाई", "बॉस", "जी"],
}
BANKS4["mx_dev"] = {
    "open": ["हो सर, बोला", "हॅलो, हां बोला", "येस, बोला"],
    "yes": ["हो सर, अवेलेबल आहे", "स्टॉकमध्ये आहे", "आहे सर, रेडी स्टॉक आहे"],
    "no": ["सॉरी, आउट ऑफ स्टॉक आहे", "स्टॉक नाही आहे सध्या", "ते मॉडेल अवेलेबल नाही"],
    "tomorrow": ["आत्ता स्टॉक नाही, उद्या येईल", "सध्या आउट ऑफ स्टॉक आहे, नवीन स्टॉक परवा येणार आहे"],
    "unclear": ["एक मिनिट, चेक करून सांगतो", "ओनर नाहीत, नंतर कॉल करा", "माहीत नाही, नंतर कॉल करा"],
    "wrong_number": ["रॉंग नंबर आहे", "चुकीचा नंबर लागलाय", "हे दुकान नाही, चुकीचा नंबर"],
    "price": ["प्राइस {p} आहे", "रेट {p} आहे सर", "{p} फायनल", "तुम्हाला {p}_ला देऊ"],
    "mrp": ["MRP {m} आहे, पण तुम्हाला {p}_ला देऊ", "एमआरपी {m}, डिस्काउंट नंतर {p}"],
    "discount": ["एमआरपी {m} आहे पण {p}_ला लावतो", "{m}चं आहे, तुम्हाला {p}_ला देतो"],
    "gst": ["{p} प्लस जीएसटी", "{p}, जीएसटी एक्स्ट्रा"],
    "fee": ["डिलिव्हरी देतो, {fee} चार्ज लागेल", "पोरगा घरी आणून देईल, {fee} रुपये लागतील"],
    "item_price": ["आयटमचा प्राइस {p}, डिलिव्हरी वेगळी"],
    "old_new": ["आधी {old} होता, आता प्राइस {p} झाला"],
    "del_yes": ["होम डिलिव्हरी आहे", "डिलिव्हरी देतो सर", "पोरगा घरी आणून देईल", "कुरिअर करतो", "घरपोच करून देतो",
                "बॉय पाठवतो"],
    "del_no": ["डिलिव्हरी नाही, शॉपवरून पिकअप करा", "पिकअप करावा लागेल", "येऊन घेऊन जा"],
    "alt": ["{alt} अवेलेबल आहे, चालेल का?", "{alt} आहे, सेम क्वालिटी"],
    "alt_price": ["{alt} आहे, त्याचा प्राइस {ap}"],
    "chatter": ["आज खूप बिझी आहे सर", "नेटवर्क प्रॉब्लेम आहे, आवाज कट होतोय"],
    "filler": ["म्हणजे", "बघा", "अं"],
}

# Words in alternative names, in Devanagari (Hindi, Marathi), besides v2's DEV_WORDS.
DEV_WORDS_V4 = {
    **DEV_WORDS,
    "exhaust": ("एग्जॉस्ट", "एक्झॉस्ट"), "ceiling": ("सीलिंग", "सीलिंग"), "pedestal": ("पेडस्टल", "पेडेस्टल"),
    "immersion": ("इमर्शन", "इमर्शन"), "room": ("रूम", "रूम"), "soldering": ("सोल्डरिंग", "सोल्डरिंग"),
    "angle": ("एंगल", "अँगल"), "welding": ("वेल्डिंग", "वेल्डिंग"), "water": ("वॉटर", "वॉटर"), "chain": ("चेन", "चेन"),
    "blade": ("ब्लेड", "ब्लेड"), "steering": ("स्टीयरिंग", "स्टिअरिंग"), "handle": ("हैंडल", "हँडल"),
    "car": ("कार", "कार"), "bike": ("बाइक", "बाईक"), "screwdriver": ("स्क्रूड्राइवर", "स्क्रूड्रायव्हर"),
    "amp": ("एम्पियर", "अँपिअर"), "watt": ("वॉट", "वॉट"), "polycab": ("पॉलीकैब", "पॉलीकॅब"),
    "finolex": ("फिनोलेक्स", "फिनोलेक्स"), "supreme": ("सुप्रीम", "सुप्रीम"), "astral": ("एस्ट्रल", "ॲस्ट्रल"),
    "exide": ("एक्साइड", "एक्साइड"), "amaron": ("अमरोन", "अमरॉन"), "ngk": ("एनजीके", "एनजीके"),
    "godrej": ("गोदरेज", "गोदरेज"), "harrison": ("हैरिसन", "हॅरिसन"), "taparia": ("तपारिया", "तपारिया"),
    "birla": ("बिड़ला", "बिर्ला"), "makita": ("मकिता", "मकिता"), "sq": ("स्क्वेयर", "स्क्वेअर"),
    "lock": ("लॉक", "लॉक"), "cover": ("कवर", "कव्हर"),
}
# Brands a shop may offer instead of the hand-made list's products (v2's PRODUCTS has no brands).
OLD_BRANDS_V4 = {
    "1.5 sq mm copper wire, 90 m roll": ["Polycab", "Finolex", "Havells"], "1 inch PVC elbow": ["Supreme", "Astral"],
    "Activa clutch plate": ["Bosch"], "12V 7Ah battery": ["Exide", "Amaron"], "9W LED bulb": ["Philips", "Syska", "Wipro"],
    "16A MCB": ["Havells", "Legrand", "Schneider", "Anchor"], "fan capacitor 2.5 mfd": ["Havells", "Crompton"],
    "Pulsar 150 spark plug": ["Bosch", "NGK"], "wall putty 20 kg": ["Birla"], "8 mm drill bit": ["Bosch", "Taparia"],
    "50 mm padlock": ["Godrej", "Harrison"], "Activa air filter": ["Bosch"],
    "10W-30 engine oil 1 L": ["Castrol", "Shell", "Motul"], "bike headlight bulb": ["Philips", "Osram"],
    "4 socket extension board": ["Anchor", "Havells"],
}
KIND_NOUN = {t[0]: t[1] for t in TYPES} | {"glass_cutter": "glass cutter", "bolt_cutter": "bolt cutter",
                                           "pipe_cutter": "pipe cutter"}
SPEC_V4 = re.compile(r"\b\d+(?:\.\d+)?(?: ?(?:mm|sq mm|L|ml|HP|inch)|W|A|Ah|V)?\b")


def respell_spec(rng: random.Random, word: str) -> str:
    """"32A" -> "32 amp", "25W" -> "25 watt" (as people say them), sometimes."""
    m = re.fullmatch(r"(\d+)(A|W)", word)
    if m and rng.random() < 0.45:
        return f"{m[1]} {'amp' if m[2] == 'A' else 'watt'}"
    return word


def to_devanagari_v4(rng: random.Random, name: str, family: str) -> str:
    """Each word that has a Devanagari spelling is written in Devanagari (most of the time); the rest stays."""
    out = []
    for word in name.split():
        dev = DEV_WORDS_V4.get(word.lower())
        out.append(dev[0 if family == "hi" else 1] if dev and rng.random() < 0.85 else word)
    return " ".join(out)


def alternatives_v4(products: list[Product]) -> dict[str, list[dict]]:
    """For each product: the products a shop may offer instead, as {brand, spec, noun, short}."""
    real = load_real_products()
    by_kind: dict[str, list[dict]] = {}
    for r in real:
        by_kind.setdefault(r["kind"], []).append(r)
    by_name = {r["name"]: r for r in real}
    out = {}
    for p in products:
        if p.source == "old":
            brands = OLD_BRANDS_V4.get(p.name, [])
            alts = []
            for full, short in p.alts:
                maker = brands if brands and not full.lower().startswith(("local", "fevicol", "m-seal")) else [None]
                alts += [{"brand": b, "spec": None, "noun": full, "short": short} for b in maker]
            out[p.name] = alts
            continue
        r = by_name[p.name]
        mine, others = maker_words(r["name"]), []
        for o in by_kind[r["kind"]]:  # other companies only, as in all_products()
            theirs = maker_words(o["name"])
            if not theirs & mine and not any(theirs & maker_words(x["name"]) for x in others):
                others.append(o)
        alts = []
        for o in others[:4]:
            core = re.sub(r" \(pack of \d+\)| for .*$", "", o["name"])
            specs = [s for s in SPEC_V4.findall(core[len(o["brand"]):]) if re.search(r"\d", s)]
            alts.append({"brand": o["brand"], "spec": " ".join(specs[:1]) or None,
                         "noun": KIND_NOUN.get(o["kind"], o["short_noun"]), "short": o["short_noun"]})
        out[p.name] = alts
    return out


# Prices said with a fraction: सवा (+1/4), साढ़े (+1/2), पौने (-1/4) before N सौ or N हज़ार (Marathi सव्वा, साडे,
# पावणे; N + शे). price_words.json has these only on its grid (and डेढ़ / ढाई, दीड / अडीच); here they are built by
# the same rule for every N, so prices off the grid are said too (पौने सत्रह सौ = 1675, सवा पाँच हज़ार = 5250).
FRACTION_PREFIX = {
    ("hi", "dev"): {"sava": ["सवा"], "saadhe": ["साढ़े", "साढे"], "paune": ["पौने"]},
    ("hi", "roman"): {"sava": ["sava", "sawa", "savaa"], "saadhe": ["sadhe", "saadhe", "sade"],
                      "paune": ["paune", "pone", "pauna"]},
    ("mr", "dev"): {"sava": ["सव्वा"], "saadhe": ["साडे"], "paune": ["पावणे"]},
    ("mr", "roman"): {"sava": ["savva", "sawwa"], "saadhe": ["sade", "saade"], "paune": ["pavane", "pavne"]},
}
FRACTION_DELTA = {"sava": 0.25, "saadhe": 0.5, "paune": -0.25}
FRACTION_MIN = {"sava": 1, "saadhe": 3, "paune": 2}  # 150 / 250 are डेढ़ / ढाई, not "साढ़े एक / दो"
THOUSAND = {("hi", "dev"): ["हज़ार", "हजार"], ("hi", "roman"): ["hazar", "hajar", "hazaar"],
            ("mr", "dev"): ["हजार"], ("mr", "roman"): ["hazar", "hajar"]}
SAU_N = [n for n in range(1, 46) if n % 10]  # N सौ: 1-9, 11-19, 21-29, ... (10, 20, 30, 40 are said in thousands)


class Fractions:
    def __init__(self, numbers: Numbers) -> None:
        self.forms: dict[tuple[int, str, str], list[list[str]]] = {}  # (price, lang, script) -> options
        for price, forms in numbers.forms.items():  # price_words.json's own colloquial forms (डेढ़ सौ, ढाई हज़ार)
            for lang in ("hi", "mr"):
                for o in forms.get((lang, "colloquial"), []):
                    self.add(price, lang, "dev", o["dev"])
                    self.add(price, lang, "roman", o["roman"])
        for lang in ("hi", "mr"):
            for script in ("dev", "roman"):
                pre = FRACTION_PREFIX[(lang, script)]
                for name, delta in FRACTION_DELTA.items():
                    for n in SAU_N:
                        if n < FRACTION_MIN[name]:
                            continue
                        hundreds = self.hundreds(numbers, n, lang, script)
                        joins = [""] if lang == "mr" and n == 1 else ["", " "] if lang == "mr" else [" "]
                        self.add(int(n * 100 + delta * 100), lang, script,
                                 [p + j + h for p in pre[name] for h in hundreds for j in joins])
                    for n in range(1, 20):
                        if n < FRACTION_MIN[name] or n * 1000 + delta * 1000 > 20000:
                            continue
                        word = numbers.primary(n, lang)
                        words = [""] if n == 1 else (word["dev"][:1] if script == "dev" else word["roman"])
                        joins = [" "] if lang == "hi" or n == 1 else ["", " "]
                        self.add(int(n * 1000 + delta * 1000), lang, script,
                                 [(p + j + w + " " + t).replace("  ", " ") for p in pre[name] for j in joins
                                  for w in words for t in THOUSAND[(lang, script)]])
        self.tokens = {t for options in self.forms.values() for o in options for text in o for t in text.split()}
        self.prices = {lang: sorted({p for (p, la, _) in self.forms if la == lang}) for lang in ("hi", "mr")}

    @staticmethod
    def hundreds(numbers: Numbers, n: int, lang: str, script: str) -> list[str]:
        """N सौ / Nशे, the way price_words.json (and make_data.py above it) writes N x 100."""
        if n == 1:
            return ["सौ"] if (lang, script) == ("hi", "dev") else ["sau"] if lang == "hi" else (
                ["शे"] if script == "dev" else ["she"])
        kind = "formal" if n < 10 else "hundreds_style"
        o = next(o for o in numbers.forms[n * 100][(lang, kind)] if o["primary"])
        return o["dev"][:1] if script == "dev" else o["roman"][:2]

    def add(self, price: int, lang: str, script: str, texts: list[str]) -> None:
        texts = [nfc(t) for t in texts if t]
        if texts:
            self.forms.setdefault((price, lang, script), []).append(texts)

    def words(self, rng: random.Random, price: int, lang: str, script: str) -> str | None:
        options = self.forms.get((price, lang, "dev" if script == "dev" else "roman"))
        if not options:
            return None
        texts = rng.choice(options)
        return texts[0] if len(texts) == 1 or rng.random() < 0.55 else rng.choice(texts[1:])


# Speech-to-text noise (Devanagari rows only). Parts of these kinds may lose or repeat a word: nothing the answer
# depends on is in them except a price, and number words, digits and the alternative's name are never touched.
NOISE_SAFE_SLOTS = {"open", "close", "filler", "address", "chatter", "timing", "price", "noprice"}
NOISE_OPS = {"drop a word": 2, "merge words": 2, "split a word": 1, "repeat a word": 2, "filler word": 3,
             "nukta/chandrabindu": 3, "digits": 1, "no punctuation": 3}
KNOWN_MERGES = ["दे देंगे", "भेज देंगे", "कर देंगे", "मिल जाएगा", "आ जाएगा", "दे दूंगा", "ले लो", "दे जाएगा",
                "लगा देंगे", "देऊ शकतो", "मिल जाएगी", "हो जाएगी", "पाठवून देतो", "बघून सांगतो", "चेक करके"]
KNOWN_SPLITS = {"देखकर": "देख कर", "करके": "कर के", "आकर": "आ कर", "स्टॉकमध्ये": "स्टॉक मध्ये",
                "आमच्याकडे": "आमच्या कडे", "गोडाऊनमध्ये": "गोडाऊन मध्ये", "सकाळपासून": "सकाळ पासून", "इससे": "इस से",
                "उसकी": "उस की", "आपवाला": "आप वाला", "वाजेपर्यंत": "वाजे पर्यंत", "परवापर्यंत": "परवा पर्यंत",
                "नाहीये": "नाही ये", "मिलाकर": "मिला कर", "तुमच्यासाठी": "तुमच्या साठी", "दुकानातून": "दुकाना तून"}
REPEATABLE = {"हाँ", "हां", "है", "अरे", "जी", "हो", "आहे", "तो", "वो", "अच्छा", "बोलो", "बोला", "हेलो", "हॅलो"}
ASR_FILLERS = {"hi": ["अं", "हम्म", "देखो", "मतलब", "तो", "अच्छा", "वो"], "mr": ["अं", "हां", "बघा", "म्हणजे", "तर"]}
ASR_START_FILLERS = {"hi": ["हाँ जी", "हां जी", "हेलो"], "mr": ["हां", "हॅलो"]}
KNOWN_MERGES = [nfc(m) for m in KNOWN_MERGES]
KNOWN_SPLITS = {nfc(k): nfc(v) for k, v in KNOWN_SPLITS.items()}
REPEATABLE = {nfc(w) for w in REPEATABLE}
# English words that Whisper writes in Devanagari or in Latin letters, either way (Hinglish rows).
LATIN_LOANS = {"डिलीवरी": "delivery", "डिलिव्हरी": "delivery", "स्टॉक": "stock", "रेट": "rate", "प्राइस": "price",
               "एमआरपी": "MRP", "जीएसटी": "GST", "ऑर्डर": "order", "अवेलेबल": "available", "फाइनल": "final",
               "डिस्काउंट": "discount", "पिकअप": "pickup", "होम": "home", "सर": "sir", "सॉरी": "sorry",
               "कूरियर": "courier", "कॉल": "call", "पीस": "piece", "बॉक्स": "box", "ओनर": "owner", "शॉप": "shop"}


class Row4(Row):
    """A v4 row: v4's banks, protected facts, and speech-to-text noise."""

    def __init__(self, gen: "Generator4", lang: str, product: Product, kind: str, noisy: bool) -> None:
        super().__init__(gen, lang, product, kind)
        self.slots: list[str] = []
        self.protect: list[str] = []  # numbers as said, the alternative's name: never changed by the noise
        self.noisy = noisy
        self.noise: list[str] = []
        self.alt_names: list[str] = []

    def bank_lang(self, slot: str) -> str:
        if self.lang in OWN_BANK_V4:
            own = BANKS4[self.lang]
            if slot in own and (self.rng.random() < 0.7 or slot not in BANKS4[BASE[self.lang]]):
                return self.lang
            return BASE[self.lang]
        return self.lang

    def has(self, slot: str) -> bool:
        return slot in BANKS4[self.lang] or slot in BANKS4[self.base]

    def say(self, slot: str, **text) -> str:
        blang = self.bank_lang(slot)
        return self.fill(self.rng.choice(BANKS4[blang][slot]), blang, **text)

    def add(self, slot: str, **text) -> None:
        self.parts.append(self.say(slot, **text))
        self.slots.append(slot)

    def render(self, key: str, family: str, script: str) -> tuple[str, bool]:
        if key == self.main and self.price_style == "fraction":
            said = self.gen.fractions.words(self.rng, self.nums[key], family, script) if family != "en" else None
            if said:
                self.protect.append(nfc(said))
                return said, True
            self.price_style = "colloquial"
        value, words = super().render(key, family, script)
        self.protect.append(nfc(value))
        return value, words

    def alt_text(self, allow_short: bool = True, label: bool = True) -> str:
        said, name = self.gen.alt_phrase(self, allow_short)
        if label:
            self.answer["alternative"] = name
        self.protect.append(name)
        self.alt_names.append(name)
        return said

    # --- the noise
    def fact_word(self, word: str) -> bool:
        w = word.strip(",.।?!")
        if not w or any(c.isdigit() for c in w) or w in self.gen.tokens4:
            return True
        return any(w in t or t in w for p in self.protect for t in p.split())

    def noise_parts(self, ops: list[str]) -> None:
        rng, family = self.rng, FAMILY[self.base]
        if len(self.slots) != len(self.parts):
            self.slots = ["?"] * len(self.parts)  # unknown parts are not touched
        safe = [i for i, s in enumerate(self.slots) if s in NOISE_SAFE_SLOTS]
        for op in ops:
            done = False
            if op == "drop a word":
                for i in rng.sample(safe, len(safe)):
                    words = self.parts[i].split()
                    can = [j for j, w in enumerate(words) if not self.fact_word(w)
                           and not (0 < j < len(words) - 1 and self.fact_word(words[j - 1])
                                    and self.fact_word(words[j + 1]))]
                    if len(words) >= 3 and can:
                        j = rng.choice(can)
                        if j > 0 and words[j][-1] in ",.।?!" and words[j - 1][-1] not in ",.।?!":
                            words[j - 1] += words[j][-1]
                        self.parts[i] = " ".join(words[:j] + words[j + 1:])
                        done = True
                        break
            elif op == "merge words":
                known = [(i, m) for i, part in enumerate(self.parts) for m in KNOWN_MERGES if m in part]
                if known:
                    i, m = rng.choice(known)
                    self.parts[i] = self.parts[i].replace(m, m.replace(" ", ""), 1)
                    done = True
                else:
                    for i in rng.sample(safe, len(safe)):
                        words = self.parts[i].split()
                        pairs = [j for j in range(len(words) - 1) if not self.fact_word(words[j])
                                 and not self.fact_word(words[j + 1]) and not re.search(r"[,.।?!]$", words[j])]
                        if pairs:
                            j = rng.choice(pairs)
                            self.parts[i] = " ".join(words[:j] + [words[j] + words[j + 1]] + words[j + 2:])
                            done = True
                            break
            elif op == "split a word":
                found = [(i, w) for i, part in enumerate(self.parts) for w in KNOWN_SPLITS
                         if re.search(rf"(?<![ऀ-ॿ]){w}(?![ऀ-ॿ])", part)]
                if found:
                    i, w = rng.choice(found)
                    self.parts[i] = self.parts[i].replace(w, KNOWN_SPLITS[w], 1)
                    done = True
            elif op == "repeat a word":
                for i in rng.sample(range(len(self.parts)), len(self.parts)):
                    words = self.parts[i].split()
                    if not words:
                        continue
                    first = words[0].strip(",.।?!")
                    if first in REPEATABLE:
                        self.parts[i] = first + " " + self.parts[i]
                        done = True
                        break
                    if i in safe:
                        can = [j for j, w in enumerate(words) if not self.fact_word(w)
                               and not re.search(r"[,.।?!]$", w)]
                        if can:
                            j = rng.choice(can)
                            self.parts[i] = " ".join(words[:j + 1] + [words[j]] + words[j + 1:])
                            done = True
                            break
            elif op == "filler word":
                i = rng.randrange(len(self.parts) + 1)
                word = rng.choice(ASR_START_FILLERS[family] if i == 0 and rng.random() < 0.4 else ASR_FILLERS[family])
                self.parts.insert(i, word)
                self.slots.insert(i, "filler")
                safe = [s + (s >= i) for s in safe] + [i]
                done = True
            if done:
                self.noise.append(op)

    def noise_text(self, said: str, ops: list[str]) -> str:
        rng = self.rng
        for op in ops:
            new = said
            if op == "nukta/chandrabindu":
                words = said.split(" ")
                for j, w in enumerate(words):
                    if self.fact_word(w):
                        continue
                    if "़" in w and rng.random() < 0.7:
                        w = w.replace("़", "")  # ज़ -> ज, ड़ -> ड
                    if "ँ" in w and rng.random() < 0.7:
                        w = w.replace("ँ", "ं")  # हाँ -> हां
                    if w.startswith("नहीं") and rng.random() < 0.5:
                        w = w.replace("नहीं", "नही", 1)
                    if "ॅ" in w and rng.random() < 0.5:
                        w = w.replace("ॅ", "े")  # हॅलो -> हेलो
                    words[j] = w
                new = " ".join(words)
            elif op == "digits":  # 220 <-> २२०, not inside the alternative's name ("32A")
                to_dev = bool(re.search(r"[0-9]", said))
                table = DEV_DIGITS if to_dev else str.maketrans("०१२३४५६७८९", "0123456789")
                names = {t for a in self.alt_names for t in a.split()}
                words = said.split(" ")
                for j, w in enumerate(words):
                    core = w.strip(",.।?!")
                    if core and core not in names and re.fullmatch(r"[0-9]+" if to_dev else r"[०-९]+", core):
                        words[j] = w.translate(table)
                new = " ".join(words)
            elif op == "no punctuation" and "no punctuation" not in self.flags:
                new = self.gen.strip_punctuation(said)
            if new != said:
                said = new
                self.noise.append(op)
        return said

    def latin_loanwords(self) -> None:
        """Hinglish rows: some English words in Latin letters, as Whisper sometimes writes them."""
        for i, part in enumerate(self.parts):
            words = part.split(" ")
            for j, w in enumerate(words):
                core = w.strip(",.।?!")
                if core in LATIN_LOANS and not self.fact_word(w) and self.rng.random() < 0.7:
                    words[j] = w.replace(core, LATIN_LOANS[core])
            self.parts[i] = " ".join(words)
        self.flags.add("latin loanwords")

    def finish(self) -> str:
        script, rng = SCRIPT[self.base], self.rng
        if len(self.slots) != len(self.parts):
            self.slots = ["?"] * len(self.parts)
        self.parts = [nfc(part) for part in self.parts]
        if rng.random() < 0.3 and self.kind != "short":
            self.flags.add("filler")
            fillers = BANKS4[self.bank_lang("filler")]["filler"]
            first = 1 if "opened" in self.flags and len(self.parts) > 1 else 0
            i = rng.randrange(first, len(self.parts))
            self.parts[i] = rng.choice(fillers) + " " + self.parts[i]
        text_ops: list[str] = []
        if self.noisy:
            weights = dict(NOISE_OPS)
            ops = []
            for _ in range(rng.choice([1, 1, 2, 2, 3])):
                op = rng.choices(list(weights), weights=list(weights.values()))[0]
                ops.append(op)
                weights.pop(op)
            self.noise_parts([o for o in ops if o in ("drop a word", "merge words", "split a word", "repeat a word",
                                                     "filler word")])
            text_ops = [o for o in ops if o in ("nukta/chandrabindu", "digits", "no punctuation")]
        if self.lang in ("hx_dev", "mx_dev") and rng.random() < 0.35:
            self.latin_loanwords()
        sep = rng.choice(SEP[script])
        no_punct = rng.random() < 0.12
        said = self.parts[0]
        for part in self.parts[1:]:
            if no_punct:
                said += ", " + part
                continue
            new_sentence = said.endswith(("?", ".", "!", "।"))
            joiner = " " if new_sentence else sep
            if joiner == " " and not new_sentence and self.gen.ends_with_number(said) and \
                    self.gen.starts_with_number(part):
                joiner = ", "
            said += joiner + part
        if script == "rom" and rng.random() < 0.35:
            said = said[0].upper() + said[1:]
        if no_punct:
            said = self.gen.strip_punctuation(said)
            if script == "rom":
                said = said.lower()
                if self.answer["alternative"]:
                    self.answer["alternative"] = self.answer["alternative"].lower()
            self.flags.add("no punctuation")
        elif not said.endswith(("?", ".", "!", "।")) and rng.random() < 0.4:
            said += "।" if script == "dev" and sep == "। " else "."
        if self.noisy:
            said = self.noise_text(said, text_ops)
            if not self.noise:  # every noisy row gets at least one change
                said = self.noise_text(said, ["no punctuation", "nukta/chandrabindu", "digits"])
            if not self.noise:
                said = ASR_FILLERS[FAMILY[self.base]][0] + " " + said
                self.noise.append("filler word")
        return said


class Generator4(Generator):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.fractions = Fractions(self.numbers)
        self.tokens4 = self.numbers.tokens | self.fractions.tokens
        self.alts4 = alternatives_v4(self.products)

    def ends_with_number(self, text: str) -> bool:
        words = re.findall(r"[\wऀ-ॿ]+", text)
        return bool(words) and (any(c.isdigit() for c in words[-1]) or words[-1].lower() in self.tokens4)

    def starts_with_number(self, text: str) -> bool:
        words = re.findall(r"[\wऀ-ॿ]+", text)
        return bool(words) and (any(c.isdigit() for c in words[0]) or words[0].lower() in self.tokens4)

    # --- prices
    def choose_style(self, row: Row) -> str:
        r = self.rng.random()
        if SCRIPT[row.base] == "dev":
            return "digits" if r < 0.26 else "dev_digits" if r < 0.32 else "words" if r < 0.52 else (
                "fraction" if r < 0.84 else "hundreds")
        return "digits" if r < 0.34 else "words" if r < 0.52 else "fraction" if r < 0.84 else "hundreds"

    def pick_price(self, row: Row, style: str | None = None) -> int:
        style = style or self.choose_style(row)
        fams = self.family_of(row)
        if style != "fraction" or not fams:
            return super().pick_price(row, "colloquial" if style == "fraction" else style)
        product, cands = row.product, self.fractions.prices[fams[0]]
        if product.price is not None:
            base = self.jitter(product.price)
            near = sorted((c for c in cands if abs(c - base) <= 0.15 * base), key=lambda c: abs(c - base))
            near = near[:3]
        else:
            near = [c for c in cands if product.low <= c <= product.high]
        if not near:
            return super().pick_price(row, "colloquial")
        row.price_style = "fraction"
        return self.rng.choice(near)

    def other_style(self, row: Row, key: str) -> str:
        if row.price_style == "fraction" and key not in row.styles:
            row.price_style = "words"
            try:
                return super().other_style(row, key)
            finally:
                row.price_style = "fraction"
        return super().other_style(row, key)

    # --- the alternative: brand + size/type + item, as said
    def alt_phrase(self, row: Row4, allow_short: bool = True) -> tuple[str, str]:
        rng = self.rng
        alt = rng.choice(self.alts4[row.product.name])
        brand, spec, noun, short = alt["brand"], alt["spec"], alt["noun"], alt["short"]
        if row.product.source == "old":
            full = (brand + " " if brand else "") + " ".join(respell_spec(rng, w) for w in noun.split())
            partial = brand or short
        else:
            item = noun if rng.random() < 0.6 else short
            full = " ".join(x for x in (brand, respell_spec(rng, spec) if spec else None, item) if x)
            partial = f"{brand} {item}" if spec else brand
        r = rng.random()
        use_partial = bool(r < 0.14 and partial and partial != full and allow_short)
        name = partial if use_partial else full
        if SCRIPT[row.base] == "dev" and rng.random() < 0.55:
            name = to_devanagari_v4(rng, name, FAMILY[row.base])
        name = nfc(name)
        wala = allow_short and (r > 0.86 or (use_partial and rng.random() < 0.5))
        suffix = {"rom": " wala", "dev": " वाला"}[SCRIPT[row.base]] if wala else ""
        return name + suffix, name

    # --- rows
    def row(self, kind: str, lang: str = "hi_dev", noisy: bool = False) -> tuple[dict, Row4]:
        product = self.product_for(kind, lang)
        row = Row4(self, lang, product, kind, noisy)
        getattr(self, "b_" + kind)(row)
        said = row.finish()
        return {"product": product.name, "said": said, "answer": row.answer}, row

    def b_spoken_fraction(self, row: Row4) -> None:
        self.opening(row)
        row.answer["price_inr"] = row.set_price("fraction")
        if self.rng.random() < 0.25:
            row.add("price")
            self.yes(row)
        else:
            self.yes(row)
            row.add("price")
        row.delivery(0.4)
        self.closing(row)

    def b_discount(self, row: Row4) -> None:
        self.opening(row, 0.2)
        p = row.set_price()
        row.nums["m"] = self.near(p, 1.08, 1.3, {p})
        row.answer["price_inr"] = p
        self.yes(row)
        row.add(self.either(row, "discount"))
        row.delivery(0.35)

    def b_delivery_said(self, row: Row4) -> None:
        self.opening(row, 0.25)
        self.yes(row)
        if self.rng.random() < 0.55:
            row.answer["price_inr"] = row.set_price()
            row.add("price")
        row.delivery(1.0)
        self.closing(row, 0.15)

    def b_delivery_only(self, row: Row4) -> None:
        """Only the delivery question is answered (the voice call asks it on its own): stock and price unknown."""
        self.opening(row, 0.3)
        row.delivery(1.0)
        self.closing(row, 0.1)

    def b_wrong_number(self, row: Row4) -> None:
        self.opening(row, 0.3)
        row.add("wrong_number")

    def b_other_variant_price(self, row: Row4) -> None:
        p = row.set_price()
        row.nums["ap"] = self.near(p, 0.7, 1.4, {p})
        row.answer["price_inr"] = p
        self.yes(row)
        alt = row.alt_text(allow_short=False, label=False)  # in stock: another product is no alternative
        if row.has("other_first") and self.rng.random() < 0.5:
            row.add("other_first", alt=alt)
        else:
            row.add("price")
            row.add("other", alt=alt)
        row.delivery(0.3)

    def b_rambling(self, row: Row4) -> None:
        inner = self.rng.choice(["clear_price", "spoken_fraction", "discount", "mrp_vs_offered", "old_vs_new",
                                 "delivery_fee", "delivery_said", "no_stock", "no_stock_alt", "no_stock_alt_price",
                                 "comes_later", "unclear", "price_correction", "quantity"])
        if inner == "mrp_vs_offered" and not row.product.mrp:
            inner = "clear_price"
        getattr(self, "b_" + inner)(row)
        row.flags.add("inner:" + inner)
        blang = row.bank_lang("chatter")
        used = set(row.parts)
        fresh = [t for t in BANKS4[blang]["chatter"] if row.fill(t, blang) not in used]
        first = 1 if "opened" in row.flags else 0
        last = len(row.parts) - (1 if "closed" in row.flags else 0)
        for template in self.rng.sample(fresh, min(len(fresh), self.rng.randint(2, 4))):
            i = self.rng.randint(first, last)
            row.parts.insert(i, row.fill(template, blang))
            row.slots.insert(i, "chatter")
            last += 1
        if self.rng.random() < 0.4 and "opened" not in row.flags:
            row.parts.insert(0, row.say("open"))
            row.slots.insert(0, "open")
            row.flags.add("opened")
        if self.rng.random() < 0.3 and "closed" not in row.flags:
            row.add("close")
            row.flags.add("closed")

    def b_short(self, row: Row4) -> None:
        r = self.rng.random()
        if r < 0.4:
            self.yes(row)
        elif r < 0.8:
            row.answer["in_stock"] = False
            row.say_no()
        else:
            row.add("unclear")
        if self.rng.random() < 0.5:
            blang = row.bank_lang("address")
            row.parts[-1] += " " + self.rng.choice(BANKS4[blang]["address"])


PLAN_V4 = {
    "clear_price": 870, "spoken_fraction": 480, "in_stock_no_price": 240,
    "mrp_vs_offered": 180, "discount": 240, "old_vs_new": 120, "unit_vs_bulk": 220, "pack_price": 70, "gst": 90,
    "delivery_fee": 140, "quantity": 90, "timing": 50, "other_variant_price": 80,
    "delivery_said": 300, "delivery_only": 120,
    "no_stock": 270, "no_stock_alt": 420, "no_stock_alt_price": 180, "comes_later": 160, "no_stock_price_said": 50,
    "can_order": 80,
    "unclear": 330, "wrong_number": 70, "unclear_price_said": 50,
    "price_correction": 200, "stock_correction_yes": 70, "stock_correction_no": 70, "negation_trap_yes": 90,
    "negation_trap_no": 70, "delivery_trap": 100,
    "rambling": 330, "short": 170,
}
GROUPS_V4 = {
    "single clear price": ["clear_price"],
    "price said with सवा/साढ़े/पौने/डेढ़/ढाई": ["spoken_fraction"],
    "in stock, no price": ["in_stock_no_price"],
    "several numbers (MRP, discount, old/new, unit/box, pack, GST, fee, ...)": [
        "mrp_vs_offered", "discount", "old_vs_new", "unit_vs_bulk", "pack_price", "gst", "delivery_fee", "quantity",
        "timing", "other_variant_price"],
    "delivery said (direct and indirect)": ["delivery_said", "delivery_only"],
    "not in stock": ["no_stock", "no_stock_alt", "no_stock_alt_price", "comes_later", "no_stock_price_said",
                     "can_order"],
    "unclear / call back / wrong number (all null)": ["unclear", "wrong_number", "unclear_price_said"],
    "self-corrections and negation traps": ["price_correction", "stock_correction_yes", "stock_correction_no",
                                            "negation_trap_yes", "negation_trap_no", "delivery_trap"],
    "rambling": ["rambling"],
    "short": ["short"],
}
PRICE_STYLE_NAMES = {"digits": "digits", "dev_digits": "Devanagari digits", "words": "words",
                     "colloquial": "words, सवा/डेढ़ style (on the grid)", "hundreds": "words, hundreds style (बारह सौ)",
                     "fraction": "words, सवा/साढ़े/पौने/डेढ़/ढाई + सौ/हज़ार"}


def make_v4(n: int = N_ROWS_V4, seed: int = SEED_V4) -> tuple[list[dict], list[dict]]:
    gen = Generator4(seed)
    plan = [kind for kind, count in PLAN_V4.items() for _ in range(count)]
    assert len(plan) == n, len(plan)
    gen.rng.shuffle(plan)
    classes = [name for name, (share, _) in MIX_V4.items() for _ in range(round(share * n))]
    assert len(classes) == n, len(classes)
    gen.rng.shuffle(classes)
    dev = [i for i, c in enumerate(classes) if c.endswith("Devanagari")]
    noisy = set(gen.rng.sample(dev, round(NOISE_SHARE_V4 * len(dev))))
    rows, meta, seen = [], [], set()
    for i, (kind, cls) in enumerate(zip(plan, classes, strict=True)):
        langs = MIX_V4[cls][1]
        for _ in range(50):
            lang = gen.rng.choices(list(langs), weights=list(langs.values()))[0]
            ex, row = gen.row(kind, lang, noisy=i in noisy)
            if ex["said"] not in seen:
                break
        seen.add(ex["said"])
        rows.append(ex)
        style = PRICE_STYLE_NAMES.get(row.price_style, "digits") if ex["answer"]["price_inr"] is not None else None
        meta.append({"kind": kind, "lang": row.lang, "mix": cls, "style": style, "flags": sorted(row.flags),
                     "noise": row.noise, "noisy": i in noisy, "source": row.product.source})
    return rows, meta


HELD_OUT = ("test.jsonl", "hard_test.jsonl", "fresh_test.jsonl", "asr_test.jsonl")


def leakage_v4(rows: list[dict], n: int = 8) -> dict:
    """Counts only (no text is printed): held-out rows equal to, or sharing n words in a row with, a training row."""
    def grams(text):
        w = words_of(text)
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

    train_said = {r["said"] for r in rows}
    train_grams = set().union(*(grams(s) for s in train_said))
    out = {}
    for name in HELD_OUT:
        path = HERE / "data" / name
        if not path.exists():
            continue
        held = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[name] = {"identical_said": sum(r["said"] in train_said for r in held),
                     f"rows_sharing_{n}_word_run": sum(bool(grams(r["said"]) & train_grams) for r in held)}
    return out


def stats_v4(rows: list[dict], meta: list[dict]) -> dict:
    n = len(rows)

    def share(counter: Counter) -> dict:
        return {k: f"{v} ({v / n:.1%})" for k, v in counter.most_common()}

    group_of = {kind: g for g, kinds in GROUPS_V4.items() for kind in kinds}
    dev = [m for m in meta if m["mix"].endswith("Devanagari")]
    nulls = sum(all(v is None for v in r["answer"].values()) for r in rows)
    alts = [r["answer"]["alternative"] for r in rows if r["answer"]["alternative"]]
    return {
        "rows": n,
        "categories": share(Counter(group_of[m["kind"]] for m in meta)),
        "kinds": share(Counter(m["kind"] for m in meta)),
        "language_and_script_mix": share(Counter(m["mix"] for m in meta)),
        "language_detail": share(Counter(LANG_NAMES_V4[m["lang"]] for m in meta)),
        "script_of_the_text": share(Counter(script_of(r["said"]) for r in rows)),
        "asr_noise": {"devanagari_rows": len(dev), "with_noise": f"{sum(m['noisy'] for m in dev)} "
                      f"({sum(m['noisy'] for m in dev) / len(dev):.1%} of Devanagari rows)",
                      "changes": dict(Counter(op for m in dev for op in m["noise"]).most_common())},
        "all_four_null": f"{nulls} ({nulls / n:.1%})",
        "in_stock": share(Counter(str(r["answer"]["in_stock"]) for r in rows)),
        "delivery": share(Counter(str(r["answer"]["delivery"]) for r in rows)),
        "alternative": {"null": n - len(alts), "text": len(alts),
                        "words in the name": dict(sorted(Counter(len(a.split()) for a in alts).items())),
                        "in Devanagari": sum(bool(DEVANAGARI.search(a)) for a in alts)},
        "price_inr": {"null": sum(r["answer"]["price_inr"] is None for r in rows),
                      "how_said": dict(Counter(m["style"] for m in meta if m["style"]).most_common())},
        "latin_loanwords_rows": sum("latin loanwords" in m["flags"] for m in meta),
        "no_punctuation": sum("no punctuation" in m["flags"] or "no punctuation" in m["noise"] for m in meta),
        "product_source": share(Counter(m["source"] for m in meta)),
        "distinct_products": len({r["product"] for r in rows}),
        "leakage": leakage_v4(rows),
    }


def print_table(title: str, table: dict) -> None:
    print(f"\n{title}")
    width = max(len(k) for k in table)
    for k, v in table.items():
        print(f"  {k:<{width}}  {v}")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    """python make_data.py: both sets; python make_data.py v2 (or v4): only that one."""
    import sys

    which = sys.argv[1:] or ["v2", "v4"]
    if "v2" in which:
        path = HERE / "data" / "train.jsonl"
        rows, meta = make()
        write_rows(path, rows)
        print(f"wrote {len(rows)} examples to {path}")
        print(json.dumps(stats(rows, meta), ensure_ascii=False, indent=1))
    if "v4" in which:
        path = HERE / "data" / "train_v4.jsonl"
        rows, meta = make_v4()
        write_rows(path, rows)
        print(f"\nwrote {len(rows)} examples to {path} (training set v4)")
        st = stats_v4(rows, meta)
        print_table("v4 categories", st["categories"])
        print_table("v4 language and script mix", st["language_and_script_mix"])
        print_table("v4 speech-to-text noise", st["asr_noise"])
        print(json.dumps({k: v for k, v in st.items() if k not in ("categories", "language_and_script_mix",
                                                                   "asr_noise")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
