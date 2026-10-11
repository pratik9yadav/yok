"""Make sample training data: shopkeeper replies in Hindi, Marathi and English.

This is made-up data to prove the training works end to end. For a real model, replace
data/train.jsonl with real call transcripts in the same format:
    {"product": "...", "said": "...", "answer": {"in_stock": ..., "price_inr": ...,
     "delivery": ..., "alternative": ...}}

Run:  python make_data.py            (writes data/train.jsonl)
"""

import json
import random
from pathlib import Path

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

# Round prices that people often say as words.
WORDS = {
    "hi_rom": {20: "bees", 30: "tees", 40: "chaalis", 50: "pachaas", 100: "sau",
               150: "dedh sau", 200: "do sau", 250: "dhai sau", 500: "paanch sau", 1000: "hazaar"},
    "mr_rom": {20: "vees", 30: "tees", 40: "chaalees", 50: "pannas", 100: "shambhar",
               150: "didshe", 200: "donshe", 250: "adichshe", 500: "paachshe", 1000: "hajaar"},
    "hi_dev": {20: "बीस", 30: "तीस", 40: "चालीस", 50: "पचास", 100: "सौ",
               150: "डेढ़ सौ", 200: "दो सौ", 250: "ढाई सौ", 500: "पाँच सौ", 1000: "हज़ार"},
    "mr_dev": {20: "वीस", 30: "तीस", 40: "चाळीस", 50: "पन्नास", 100: "शंभर",
               150: "दीडशे", 200: "दोनशे", 250: "अडीचशे", 500: "पाचशे", 1000: "हजार"},
    "en": {},
}

PHRASES = {
    "hi_rom": {
        "yes": ["haan hai", "haan ji, mil jayega", "hai bhai", "available hai", "haan, stock mein hai",
                "ji haan, rakha hai"],
        "no": ["nahi hai", "khatam ho gaya", "abhi stock mein nahi hai", "nahi milega bhai",
               "woh hum nahi rakhte", "abhi nahi hai, parso aayega"],
        "unclear": ["ruko, dekh ke batata hun", "malik abhi nahi hain, baad mein phone karo",
                    "pata nahi, godown mein dekhna padega", "thodi der baad call karo"],
        "price": ["{p} rupaye ka hai", "{p} ka padega", "rate {p} hai", "{p} lagega", "₹{p}",
                  "{p} rupaye final"],
        "deliver_yes": ["ghar pe bhej denge", "delivery ho jayegi", "ladka bhej deta hun",
                        "home delivery hai"],
        "deliver_no": ["delivery nahi karte, aake le jao", "dukaan pe aake lena padega",
                       "delivery nahi hai"],
        "alt": ["{alt} hai, chalega?", "uski jagah {alt} de sakte hain", "{alt} rakha hai"],
        "filler": ["haan bolo", "hello, kaun?", "ji boliye", "haan bhai"],
        "hours": ["dukaan {h} baje tak khuli hai", "{h} baje ke pehle aana"],
        "qty": ["{q} piece pade hain", "{q} hi bache hain"],
        "sep": ", ",
    },
    "mr_rom": {
        "yes": ["ho aahe", "aahe na", "ho, milel", "stock madhe aahe", "haan aahe"],
        "no": ["nahi aahe", "sampla", "sadhya nahi", "aamhi te thevat nahi", "udya yeil, aaj nahi"],
        "unclear": ["thamba, baghun sangto", "malak nahit, nantar phone kara",
                    "mahit nahi, godown madhe baghava lagel"],
        "price": ["{p} rupaye", "{p} la padel", "rate {p} aahe", "{p} rupaye honar"],
        "deliver_yes": ["ghari pathavto", "delivery deto", "porga pathavto"],
        "deliver_no": ["delivery nahi, dukanat yeun ghya", "yeun nyava lagel", "delivery nahi karat"],
        "alt": ["{alt} aahe, chalel ka?", "tyachya aivaji {alt} deu shakto"],
        "filler": ["haan bola", "hello, kon?", "bola"],
        "hours": ["dukan {h} vajta paryant ughada aahe", "{h} chya aat ya"],
        "qty": ["{q} piece aahet", "{q} ch urle aahet"],
        "sep": ", ",
    },
    "en": {
        "yes": ["Yes, we have it", "Yes, available", "It's in stock", "Yes sir, we have"],
        "no": ["No, we don't have it", "Out of stock", "Sorry, finished", "We don't keep that",
               "Not today, it comes tomorrow"],
        "unclear": ["Hold on, let me check", "Owner is not here, call later",
                    "Not sure, I'll have to check the godown"],
        "price": ["It's {p} rupees", "Price is {p}", "{p} rupees", "That will be Rs {p}"],
        "deliver_yes": ["We can deliver", "Home delivery is available", "I'll send it with our boy"],
        "deliver_no": ["No delivery, you'll have to come to the shop", "Pickup only",
                       "We don't deliver"],
        "alt": ["We have {alt}, will that work?", "I can give {alt} instead"],
        "filler": ["Hello?", "Yes, tell me", "Hello, who is this?"],
        "hours": ["We are open till {h} pm", "Come before {h}"],
        "qty": ["We have {q} pieces", "Only {q} left"],
        "sep": ". ",
    },
    "hi_dev": {
        "yes": ["हाँ, है", "हाँ जी, मिल जाएगा", "स्टॉक में है", "है भाई"],
        "no": ["नहीं है", "ख़त्म हो गया", "अभी नहीं है, परसों आएगा", "वो हम नहीं रखते"],
        "unclear": ["रुकिए, देखकर बताता हूँ", "मालिक नहीं हैं, बाद में फ़ोन कीजिए"],
        "price": ["{p} रुपये का है", "{p} लगेगा", "रेट {p} है"],
        "deliver_yes": ["घर पर भेज देंगे", "होम डिलीवरी हो जाएगी"],
        "deliver_no": ["डिलीवरी नहीं करते, दुकान पर आकर ले जाइए", "डिलीवरी नहीं है"],
        "alt": ["{alt} है, चलेगा?", "उसकी जगह {alt} दे सकते हैं"],
        "filler": ["हाँ बोलिए", "हैलो, कौन?"],
        "hours": ["दुकान {h} बजे तक खुली है"],
        "qty": ["{q} पीस पड़े हैं"],
        "sep": "। ",
    },
    "mr_dev": {
        "yes": ["हो, आहे", "आहे ना", "स्टॉकमध्ये आहे"],
        "no": ["नाही आहे", "संपलं", "सध्या नाही, उद्या येईल", "आम्ही ते ठेवत नाही"],
        "unclear": ["थांबा, बघून सांगतो", "मालक नाहीत, नंतर फोन करा"],
        "price": ["{p} रुपये", "{p} ला पडेल", "रेट {p} आहे"],
        "deliver_yes": ["घरी पाठवतो", "घरपोच देतो"],
        "deliver_no": ["डिलिव्हरी नाही, दुकानात येऊन घ्या", "येऊन न्यावं लागेल"],
        "alt": ["{alt} आहे, चालेल का?", "त्याच्याऐवजी {alt} देऊ शकतो"],
        "filler": ["हां बोला", "हॅलो, कोण?"],
        "hours": ["दुकान {h} वाजेपर्यंत उघडं आहे"],
        "qty": ["{q} पीस आहेत"],
        "sep": ". ",
    },
}

LANG_WEIGHTS = {"hi_rom": 0.35, "mr_rom": 0.2, "en": 0.2, "hi_dev": 0.15, "mr_dev": 0.1}


def make_price(rng: random.Random, low: int, high: int) -> int:
    step = 5 if high <= 100 else 10 if high <= 1000 else 50
    return rng.randrange(low // step * step, high // step * step + step, step)


def price_text(rng: random.Random, lang: str, price: int) -> str:
    words = WORDS[lang]
    if price in words and rng.random() < 0.5:
        return words[price]
    return str(price)


def make_example(rng: random.Random) -> dict:
    lang = rng.choices(list(LANG_WEIGHTS), weights=list(LANG_WEIGHTS.values()))[0]
    ph = PHRASES[lang]
    product, alts, (low, high) = rng.choice(PRODUCTS)
    status = rng.choices(["yes", "no", "unclear"], weights=[0.5, 0.32, 0.18])[0]
    answer = {"in_stock": None, "price_inr": None, "delivery": None, "alternative": None}
    parts = []

    if rng.random() < 0.3:
        parts.append(rng.choice(ph["filler"]))
    parts.append(rng.choice(ph[status]))

    if status == "yes":
        answer["in_stock"] = True
        if rng.random() < 0.2:
            parts.append(rng.choice(ph["qty"]).format(q=rng.randint(2, 20)))
        if rng.random() < 0.75:
            price = make_price(rng, low, high)
            answer["price_inr"] = price
            template = rng.choice(ph["price"])
            said = str(price) if "₹" in template else price_text(rng, lang, price)
            parts.append(template.format(p=said))
    elif status == "no":
        answer["in_stock"] = False
        if rng.random() < 0.4:
            alt = rng.choice(alts)
            answer["alternative"] = alt
            parts.append(rng.choice(ph["alt"]).format(alt=alt))

    if status != "unclear" and rng.random() < 0.6:
        deliver = rng.random() < 0.5
        answer["delivery"] = deliver
        parts.append(rng.choice(ph["deliver_yes" if deliver else "deliver_no"]))

    # A number that is not the price, so the model learns not to grab any number.
    if rng.random() < 0.15:
        parts.append(rng.choice(ph["hours"]).format(h=rng.randint(7, 10)))

    return {"product": product, "said": ph["sep"].join(parts), "answer": answer}


def make(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        ex = make_example(rng)
        key = (ex["product"], ex["said"])
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def main() -> None:
    path = Path(__file__).parent / "data" / "train.jsonl"
    path.parent.mkdir(exist_ok=True)
    rows = make(400, seed=7)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} examples to {path}")


if __name__ == "__main__":
    main()
