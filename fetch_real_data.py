"""Download real data from open sources. Runs on a GitHub Actions runner (see .github/workflows/fetch_data.yml),
because this project's Claude container cannot reach Kaggle or Hugging Face.

    python fetch_real_data.py products      Kaggle: Flipkart products (CC BY-SA 4.0) and Amazon India 2023
                                            (kept only if Kaggle reports an open licence)
    python fetch_real_data.py indicvoices   Hugging Face: AI4Bharat IndicVoices Hindi + Marathi conversation
                                            transcripts (CC BY 4.0). Gated: needs HF_TOKEN from an account that
                                            clicked "Agree" on https://huggingface.co/datasets/ai4bharat/IndicVoices
    python fetch_real_data.py all           both
    python fetch_real_data.py probe a/b,c/d Hugging Face datasets: licence, gated or not, folders, file types and the
                                            columns of one parquet file (no audio downloaded, nothing written)

Writes data/real/*.jsonl and data/real/fetch_report.json. Licences and attribution: data/real/NOTICE.md.
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "data" / "real"

# Licence names as Kaggle reports them; anything NonCommercial / NoDerivatives / unknown is refused.
OPEN_LICENCES = ("cc0", "cc-by-4.0", "cc-by-sa-4.0", "odc-by", "odbl", "pddl", "attribution 4.0")
CLOSED_MARKERS = ("nc", "nd", "noncommercial", "noderivatives", "other", "unknown")

FLIPKART_TOP = {"Automotive", "Tools & Hardware", "Home Improvement"}
# Amazon India's category names are in Hindi (seen in the 2023 dataset): car and motorbike, industrial
# (tools, fasteners), indoor lighting, helmets, fans and heaters, garden.
AMAZON_HINDI_CATEGORIES = {"कार और मोटरबाइक", "औद्योगिक और वैज्ञानिक", "इनडोर लाइटिंग", "हेलमेट",
                           "हीटिंग, कूलिंग और एयर क्वालिटी", "गार्डन और आउटडोर"}
# English names too, whole words only: "car" must not match "skin care".
AMAZON_CATEGORY = re.compile(
    r"\b(cars?|motorbikes?|motorcycles?|bikes?|automotive|tools?|hardware|electricals?|lighting|lights?|wiring|"
    r"switch(es)?|plumbing|pipes?|adhesives?|sealants?|paints?|painting|industrial|fasteners?|locks?|"
    r"batter(y|ies)|bulbs?|fans?|cables?|home improvement)\b")

# IndicVoices: Devanagari shop-talk cues (Hindi and Marathi).
PRICE = re.compile(r"रुपय|रुपए|रुपये|रूपये|रुपया|रेट|भाव|दाम|कीमत|क़ीमत|किंमत|प्राइस|प्राईस|सौ|हजार|हज़ार|शे\b|डिस्काउंट|सस्ता|महाग|महंगा")
STOCK = re.compile(r"मिल जाएगा|मिल जायेगा|मिलेगा|मिलेंगे|मिलती|मिळेल|मिळतील|मिळणार|स्टॉक|माल|खत्म|ख़त्म|संपल|उपलब्ध|available|है क्या|आहे का")
SHOP = re.compile(r"दुकान|दुकानदार|ग्राहक|किलो|पीस|पैकेट|पाकीट|डिब्बा|डबा|मीटर|लीटर|लिटर|ऑर्डर|order")
DELIVERY = re.compile(r"डिलीवरी|डिलिव्हरी|डिलेवरी|होम डिलीवरी|घरपोच|घर पर भेज|घर पे भेज|घरी पाठव|पाठवतो|भेज दें|भेज देंगे|भेज दूंगा")
PHONE_OR_EMAIL = re.compile(r"[0-9०-९]{6,}|@|\b[\w.]+@[\w.]+\b")


def short_name(name: str, max_words: int = 10, max_chars: int = 80) -> str:
    words = re.sub(r"\s+", " ", name or "").strip().split(" ")
    return " ".join(words[:max_words])[:max_chars].strip(" ,-(")


def to_int(value) -> int | None:
    try:
        number = float(str(value).replace(",", "").replace("₹", "").strip())
    except ValueError:
        return None
    return round(number) if number > 0 else None


def licence_ok(names: list[str]) -> bool:
    """True if at least one reported licence is open and none looks closed."""
    lowered = [n.lower() for n in names]
    if not lowered:
        return False
    if any(m in re.split(r"[^a-z]+", n) for n in lowered for m in CLOSED_MARKERS):
        return False
    return any(any(o in n for o in OPEN_LICENCES) for n in lowered)


def flipkart_products(rows) -> list[dict]:
    out = []
    for r in rows:
        levels = [p.strip(' ["]') for p in (r.get("product_category_tree") or "").split(">>")]
        if not levels or levels[0] not in FLIPKART_TOP:
            continue
        mrp, price = to_int(r.get("retail_price")), to_int(r.get("discounted_price"))
        if not price:
            continue
        out.append({"name": short_name(r.get("product_name", "")), "brand": (r.get("brand") or "").strip(),
                    "mrp": mrp, "price": price, "category": " > ".join(levels[:3])})
    return out


def amazon_products(rows, per_category: int = 1200) -> list[dict]:
    keep = []
    for r in rows:
        category = (r.get("categoryName") or "").strip()
        if category not in AMAZON_HINDI_CATEGORIES and not AMAZON_CATEGORY.search(category.lower()):
            continue
        price = to_int(r.get("price"))
        if not price:
            continue
        keep.append({"name": short_name(r.get("title", "")), "mrp": to_int(r.get("listPrice")), "price": price,
                     "category": category, "_reviews": to_int(r.get("reviews")) or 0})
    keep.sort(key=lambda p: -p["_reviews"])  # the most-bought items are the ones shops really stock
    taken: Counter = Counter()
    out = []
    for p in keep:  # at most `per_category` per category, so no one category swamps the rest
        if taken[p["category"]] < per_category:
            taken[p["category"]] += 1
            out.append({k: v for k, v in p.items() if k != "_reviews"})
    return out


def shop_talk_score(text: str) -> int:
    """How strongly a transcript line looks like shop talk (0 = not at all)."""
    price, stock, shop, delivery = (bool(p.search(text)) for p in (PRICE, STOCK, SHOP, DELIVERY))
    if delivery or (price and (stock or shop)) or (stock and shop):
        return price + stock + shop + 2 * delivery
    return 0


def safe_text(text: str) -> bool:
    return bool(text) and not PHONE_OR_EMAIL.search(text) and 4 <= len(text.split()) <= 120


def write_jsonl(name: str, rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return path


def kaggle(*args: str) -> None:
    subprocess.run(["kaggle", *args], check=True)


def find_licences(meta) -> list[str]:
    """Licence names anywhere in Kaggle's metadata JSON (kaggle 2.x nests them under "info")."""
    if isinstance(meta, dict):
        if isinstance(meta.get("licenses"), list):
            return [lic.get("name", "") if isinstance(lic, dict) else str(lic) for lic in meta["licenses"]]
        meta = list(meta.values())
    if isinstance(meta, list):
        for value in meta:
            found = find_licences(value)
            if found:
                return found
    return []


def kaggle_licences(ref: str, folder: Path) -> list[str]:
    kaggle("datasets", "metadata", ref, "-p", str(folder))
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
    licences = find_licences(meta)
    if not licences:
        print(f"{ref}: no licence found in metadata; top-level keys: {list(meta)[:20]}", flush=True)
    return licences


def fetch_products(report: dict) -> None:
    for ref, fname, convert, out in (
        ("PromptCloudHQ/flipkart-products", "flipkart_com-ecommerce_sample.csv", flipkart_products,
         "products_flipkart.jsonl"),
        ("asaniczka/amazon-india-products-2023-1-5m-products", "amz_in_total_products_data_processed.csv",
         amazon_products, "products_amazon_in.jsonl"),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            licences = kaggle_licences(ref, folder)
            print(f"{ref}: licence(s) {licences}", flush=True)
            if not licence_ok(licences):
                report[out] = {"skipped": f"licence not open: {licences}"}
                continue
            kaggle("datasets", "download", "-d", ref, "-p", str(folder), "--unzip")
            csv.field_size_limit(sys.maxsize)
            with open(folder / fname, encoding="utf-8", errors="replace", newline="") as fh:
                source_rows = list(csv.DictReader(fh))
            all_categories = Counter((r.get("categoryName") or r.get("product_category_tree", "").split(">>")[0])
                                     .strip(' ["') for r in source_rows)
            print(f"{ref}: {len(source_rows)} rows; categories: {all_categories.most_common(400)}", flush=True)
            rows = convert(source_rows)
            write_jsonl(out, rows)
            report[out] = {"source": f"https://www.kaggle.com/datasets/{ref}", "licence": licences,
                           "rows": len(rows), "categories": Counter(r["category"] for r in rows).most_common(25)}


def fetch_indicvoices(report: dict, limit: int = 8000) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set: skipping IndicVoices (see README: 'Real data').", flush=True)
        report["indicvoices_shop_talk.jsonl"] = {"skipped": "no HF_TOKEN secret"}
        return
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    lines, seen, scenarios, tasks, rows_read = [], set(), Counter(), Counter(), 0
    for lang in ("hindi", "marathi"):
        for shard in sorted(fs.glob(f"datasets/ai4bharat/IndicVoices/{lang}/*.parquet")):
            parquet = pq.ParquetFile(fs.open(shard, "rb"))
            names = set(parquet.schema_arrow.names)
            columns = [c for c in ("verbatim", "text", "scenario", "task_name") if c in names]
            for row in parquet.read(columns=columns).to_pylist():  # text columns only: no audio is downloaded
                rows_read += 1
                scenarios[row.get("scenario")] += 1
                if row.get("scenario") != "Conversation":
                    continue
                tasks[row.get("task_name")] += 1
                text = re.sub(r"\s+", " ", row.get("verbatim") or row.get("text") or "").strip()
                score = shop_talk_score(text)
                if score and safe_text(text) and text not in seen:
                    seen.add(text)
                    lines.append({"lang": lang, "said": text, "task": row.get("task_name"), "_score": score})
            print(f"{shard}: {rows_read} rows read, {len(lines)} shop-talk lines so far", flush=True)
    lines.sort(key=lambda r: -r["_score"])
    kept = [{k: v for k, v in r.items() if k != "_score"} for r in lines[:limit]]
    write_jsonl("indicvoices_shop_talk.jsonl", kept)
    report["indicvoices_shop_talk.jsonl"] = {
        "source": "https://huggingface.co/datasets/ai4bharat/IndicVoices", "licence": "CC BY 4.0",
        "rows_read": rows_read, "shop_talk_found": len(lines), "rows": len(kept),
        "by_lang": Counter(r["lang"] for r in kept), "scenarios": scenarios.most_common(),
        "conversation_tasks": tasks.most_common(40)}


def summarise_files(files: list[str], top: int = 12) -> dict:
    """What a dataset repo holds: its top folders (two levels, with file counts) and file types."""
    folders = Counter("/".join(f.split("/")[:2]) if f.count("/") >= 2 else f.split("/")[0] if "/" in f else "."
                      for f in files)
    types = Counter(Path(f).suffix.lower() or "(none)" for f in files)
    return {"files": len(files), "folders": dict(folders.most_common(top)), "types": dict(types.most_common(8))}


def probe_dataset(repo: str, token: str | None = None) -> dict:
    """A Hugging Face dataset's licence, gating and layout, and one parquet file's columns (text only)."""
    from huggingface_hub import HfApi, HfFileSystem

    out: dict = {"repo": repo, "page": f"https://huggingface.co/datasets/{repo}"}
    try:
        info = HfApi(token=token).dataset_info(repo, files_metadata=False)
    except Exception as e:  # noqa: BLE001 - report it and go on to the next dataset
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        return out
    card = info.card_data.to_dict() if info.card_data else {}
    files = [s.rfilename for s in info.siblings or []]
    out.update({"gated": info.gated, "licence": card.get("license"), "languages": card.get("language"),
                "configs": [c.get("config_name") for c in card.get("configs") or []][:20],
                "downloads": info.downloads, **summarise_files(files)})
    parquet = next((f for f in files if f.endswith(".parquet")), None)
    if parquet:
        try:
            import pyarrow.parquet as pq

            meta = pq.ParquetFile(HfFileSystem(token=token).open(f"datasets/{repo}/{parquet}", "rb"))
            out["sample_file"] = {"path": parquet, "rows": meta.metadata.num_rows,
                                  "columns": meta.schema_arrow.names}
        except Exception as e:  # noqa: BLE001 - gated without a token, or not readable
            out["sample_file"] = {"path": parquet, "error": f"{type(e).__name__}: {e}"[:200]}
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "probe":
        repos = [r.strip() for r in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if r.strip()]
        token = os.environ.get("HF_TOKEN") or None
        print("HF token:", "yes" if token else "no (gated sets show as not readable)")
        for repo in repos:
            print(json.dumps(probe_dataset(repo, token), ensure_ascii=False, indent=1, default=str), flush=True)
        return
    report_path = OUT / "fetch_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if which in ("products", "all"):
        fetch_products(report)
    if which in ("indicvoices", "all"):
        fetch_indicvoices(report)
    OUT.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=list) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1, default=list))


if __name__ == "__main__":
    main()
