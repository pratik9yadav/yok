> **In this repository:** `final.jsonl` below is `xpqa_hi_examples.jsonl`. It has 151 rows: one more row was removed because it mentions a PIN code (the fetch script drops every 6+ digit number). `candidates.jsonl`, `stats.json` and `check_report.json` are not included.

# NOTICE: xPQA (Hindi) rows in this folder

Applies to `candidates.jsonl`, `stats.json`, `final.jsonl`, `check_report.json` and the
scripts in `_checker/` in this folder. These files hold changed data ("Enhanced Data" in the
license's terms) taken from xPQA. The changes are listed below; step 6 covers `final.jsonl`.

## Source

- **Dataset:** xPQA, a cross-lingual product question answering dataset. It is part of the
  Amazon Science repository "Contextual Product Question Answering".
- **URL:** https://github.com/amazon-science/contextual-product-qa (folder `xPQA/`)
- **Files used:** `xPQA/train.csv`, `xPQA/dev.csv`, `xPQA/test.csv`. These were downloaded on
  2026-10-12 from `https://raw.githubusercontent.com/amazon-science/contextual-product-qa/main/xPQA/`.
  - train.csv sha256 `06c5751c73e85d2576a29ded96fe222f7f7456cd0fd3d617d5e0c78be012a719`
  - dev.csv sha256 `1e7670a77e78e622f14d871ae85c0f3196664d7230208ff02e38b7f0e306568a`
  - test.csv sha256 `1647f87a48a5537a9449af065d9eed41a42ab6fbd6203c7cf964374af8f5cab4`
- **Data provider:** Amazon (amazon-science).
- **Paper:** Xiaoyu Shen, Akari Asai, Bill Byrne, Adrià de Gispert. "xPQA: Cross-Lingual Product
  Question Answering across 12 Languages." arXiv:2305.09249 (2023).

## License

The data is licensed under the **Community Data License Agreement – Sharing, Version 1.0
(CDLA-Sharing-1.0)**. The source repository's README says "licensed under the CDLA 1.0 Sharing
License", and its LICENSE file starts with `SPDX-License-Identifier: CDLA-Sharing-1.0`.

- `LICENSE-CDLA-Sharing-1.0.txt` in this folder is an unmodified copy of that LICENSE file.
- The license is also online at https://cdla.dev/sharing-1-0/.

The license allows use and redistribution, including commercial use. If you publish these rows,
or any file that contains them (for example a training file they are merged into), you must:

- publish that file under CDLA-Sharing-1.0, unmodified, with no added restrictions;
- include the license text or a link to it;
- keep the attribution below;
- mark the file as changed.

## Attribution (keep this text with the data)

> Contains data from xPQA (https://github.com/amazon-science/contextual-product-qa), provided by
> Amazon and created by Xiaoyu Shen, Akari Asai, Bill Byrne and Adrià de Gispert, licensed under
> the Community Data License Agreement – Sharing 1.0 (CDLA-Sharing-1.0). Rows were selected,
> trimmed and labelled for the WIMB/yok project; see NOTICE.md for the changes.

## What was changed

1. **Selected rows.** Only rows with `lang` = `hi` (Hindi) were kept: 15,494 rows, which cover
   1,500 questions. Of these, 2,592 rows have a written answer. 180 rows were chosen by hand,
   about these topics:
   - availability;
   - a variant not available, with another one named (colour, size, pack, material, model);
   - pack vs piece;
   - accessories included or not;
   - price and delivery.

   18 rows where nothing about these is said were also kept. Rows that contain phone numbers or
   URLs were left out.
2. **`said`.** For 166 rows (`said_source` = `xpqa_annotator_answer_hi`), this is xPQA's
   `answer` column. That is a Hindi answer written by xPQA annotators from product-page
   evidence.
   - For 14 rows (`said_source` = `amazon_community_answer`), it is the answer part of xPQA's
     `candidate` column: the text after `Answer:` or `A:`. These are English or romanized-Hindi
     Amazon.in community answers or product-page text.
   - Whitespace was collapsed; nothing else was changed. The text was not translated, reworded
     or spell-corrected.
3. **`question`.** This is xPQA's `question` column (Hindi), with whitespace collapsed.
4. **New fields, not in xPQA:** `id`, `product`, `answer` (`in_stock`, `price_inr`, `delivery`,
   `alternative`), `label_confidence`, `label_reason` and `said_source`.
   - `product` is a short name written from the xPQA product title, plus the variant the
     question asks about.
   - The labels were written by an AI assistant (Claude) for the yok project. A person has not
     reviewed them yet.
   - Each `alternative` uses only words that appear in `said`.
5. **`source`.** This field keeps xPQA's file name, `qid`, `qa_id`, `ASIN`, candidate source
   and relevance label, so every row can be traced back.
6. **Label check (`final.jsonl`, `check_report.json`).** A second AI pass (also Claude) labelled
   all 180 candidate rows again, without looking at the first answers, and compared the two.
   - 28 rows were dropped:
     - 12 near-duplicates;
     - 5 ambiguous rows;
     - 3 that do not answer about the asked product;
     - 3 "a customer says" rows;
     - 4 product-page banners (not answers);
     - 1 row whose price "12,999" fails yok's price test.
   - In 6 rows, `alternative` was changed from null to the item the answer says comes instead
     (for example "only the almirah comes" gives "अलमीरा").
   - 2 alternatives were shortened.
   - `final.jsonl` keeps 152 rows. Each row has only `id`, `product`, `said` and `answer`; `said`
     is still xPQA's text, unchanged.
   - Every reason is listed in `check_report.json`. `_checker/` holds the scripts and the
     label table.
   - `id` is `xpqa:hi:<split>:<qa_id>`, with `:candidate` added when `said` comes from the
     `candidate` column. Remove `id` before merging into yok's `data/train.jsonl`.

## Personal data

The `said`, `question` and `product` fields were checked by script and by reading:

- **Not found:** names of private people, phone numbers, emails, URLs or @handles.
- **Places that remain:** an Indian state, a district, and one 6-digit postal (PIN) code area.
  These are not personal addresses.
- **People that remain:** a few answers in `candidates.jsonl` say "एक ग्राहक का कहना है" ("a
  customer says"), without a name. These rows are not in `final.jsonl`.
- **Checked again in `final.jsonl`:** the check found no personal data. The only place names
  are 2 states (Jharkhand, Uttar Pradesh), 1 district (Gadchiroli) and 1 PIN code (242042).

## Caveats

- These are not shopkeeper phone transcripts. Most `said` texts are short written answers to
  online shoppers' questions.
- Some phrasing does not fit a shop call: "a customer says", "in this listing", "add to cart".
