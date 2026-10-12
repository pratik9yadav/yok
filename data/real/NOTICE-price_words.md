> **In this repository:** only `price_words.json` is included; the build and check reports mentioned below are not.

# NOTICE: sources and licences for `numbers/`

## Licence of the files in this folder

| File | Licence | Why |
|---|---|---|
| `price_words.json` | **CC BY-SA 4.0** | It holds romanized spellings and counts taken from the Google Dakshina dataset (CC BY-SA 4.0). It also holds MIT and Unicode-licensed parts; their notices are below and must be kept. |
| `build_stats.json` | **CC BY-SA 4.0** | It holds Dakshina-derived Wikipedia word counts. |
| `report.md`, `NOTICE.md` | same as the repository | Written for this project. |
| `scripts/build_price_words.py`, `scripts/sankhya_check.mjs` | same as the repository | Written for this project. They copy no third-party code. They read the sources listed below and do not include them. |
| `check_report.json` | **CC BY-SA 4.0** | Report of the independent check. It quotes entries and a few counts from `price_words.json`. |
| `scripts/check_price_words.py`, `scripts/apply_check_fixes.py` | same as the repository | Written for this project (independent check). They copy no third-party code. They run NeMo and gpu-sankhya as tools and do not include them. |

Note: `report.md` is listed above but is not in this folder yet. The builder's job returned
its text separately (the job could not write it).

If you share `price_words.json` or a large part of it, keep it under CC BY-SA 4.0, keep the
attribution text below, and say that it was changed. Every romanized spelling lists its
source. To get a part without Dakshina content, drop each `roman` entry whose `sources`
include `dakshina-lexicon` or `dakshina-sentences`, and drop the `wiki_count` fields. What is
left comes from MIT, Unicode and our own rules. (This is a practical note, not legal advice.)

No personal data is included. No Wikipedia or social-media sentences are copied: from
Dakshina's Wikipedia text we keep only counts of number words and number phrases.

## Sources

### 1. Google Research Dakshina dataset v1.0 (CC BY-SA 4.0)
- URL: https://github.com/google-research-datasets/dakshina
  (tarball: https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar)
- Licence: Creative Commons Attribution-ShareAlike 4.0 International,
  https://creativecommons.org/licenses/by-sa/4.0/ . The Wikipedia text inside it is from
  Wikipedia (CC BY-SA).
- Attribution text: *"Contains data from the Dakshina dataset v1.0 by Google Research
  (Roark et al., 2020, "Processing South Asian Languages Written in the Latin Script: the
  Dakshina Dataset", LREC 2020), licensed under CC BY-SA 4.0."*
- Files used (hi and mr only):
  - `{hi,mr}/lexicons/{hi,mr}.translit.sampled.{train,dev,test}.tsv`: romanized spellings of
    single words, and how many annotators gave each spelling.
  - `{hi,mr}/romanized/{hi,mr}.romanized.rejoined.aligned.cased_nopunct.tsv`: word-by-word
    romanized sentences. We keep only (number word, spelling, count).
  - `{hi,mr}/native_script_wikipedia/{hi,mr}.wiki-filt.train.text.shuf.txt.gz`: we keep only
    counts of number words and number phrases (`wiki_count`).
- What we changed: we added the counts from the three lexicon splits together, lowercased the
  spellings, put the Devanagari in Unicode NFC, kept only number, unit, fraction and rupee
  words, and ranked the spellings.

### 2. AI4Bharat indic-numtowords 1.1.0 (MIT)
- URL: https://github.com/AI4Bharat/indic-numtowords , PyPI `indic-numtowords==1.1.0`
  (sdist sha256 d1addc21444c332e05bfd8726af427960c096c2a16776d98bee4fbc36ade5d25)
- Used: the Hindi and Marathi words for each price (`num2words(n, lang, variations=True)`,
  called through `hin/mar.cardinal.convert`, which gives the same output).
- What we changed: we fixed `सावा` to `सवा`, and the Marathi `हज़ार` to `हजार`. We turned the
  Marathi eyelash-ra written with ZWJ into U+0931. We dropped `पौने दस सौ` and `पावणे दहाशे`
  (975), a bare `एकशे` for 100, and `एक हजार एकशे` for 1100. See report.md (returned with this job).
- Licence notice:

```
MIT License

Copyright (c) 2022 AI4Bhārat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3. Unicode CLDR, RBNF number spell-out rules (Unicode License v3)
- URL: https://raw.githubusercontent.com/unicode-org/cldr/main/common/rbnf/hi.xml and `.../mr.xml`
  (downloaded 2026-10-12; sha256 hi.xml 0e36dbbbf250d7f3faaddc3ff016fc651b1ea8911f9a335c9c296ff7d290a929,
  mr.xml 78fb9e745590394f35c0a73dc5d78c0f93122df8947b80bf94879a480d580217). The repository
  LICENSE says the data is under the Unicode License v3. The file headers say
  `SPDX-License-Identifier: Unicode-DFS-2016`. Both licences ask for the same notice.
- Used: the `spellout-cardinal` rules, run for every price by a small interpreter in our script.
- What we changed: we turned the eyelash-ra written with ZWJ (74, 84 and 94 in mr) into U+0931.
- Licence notice:

```
Copyright © 1991-2026 Unicode, Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.

SPDX-License-Identifier: Unicode-3.0
```

### 4. gpu-sankhya (MIT; part of its Hinglish spellings come from Dakshina, CC BY-SA 4.0)
- URL: https://github.com/athrvk/gpu-sankhya (branch `master`, read through
  raw.githubusercontent.com on 2026-10-12) and npm `gpu-sankhya@0.8.0`
  (tarball sha1 2f5362df4948d2d62dc6f2405484fe2d5fcc2723).
- Used:
  - from `python/sankhya/langs/hi_deva.py` and `mr_deva.py`: Devanagari spellings of 1-99,
    including the spellings people type on phones;
  - from `python/sankhya/langs/hi_latn.py`: Hinglish spellings of 1-99, सौ, हज़ार and the
    fraction words. English words and digits were left out;
  - the npm parser, run only to check the value of each phrase. It is not included here.
- Which parts come from Dakshina: gpu-sankhya's `docs/DATA_GRAMMAR.md` §1.5d says some
  `hi_latn` spellings were taken from the Dakshina Hindi lexicon (CC BY-SA 4.0). Those
  spellings appear in Dakshina too, so in `price_words.json` they also carry a
  `dakshina_lexicon` count and are covered by the Dakshina attribution above. A spelling that
  only gpu-sankhya lists (no Dakshina count) is MIT. gpu-sankhya's Devanagari packs and its
  npm parser are MIT.
- What we changed: we read the Python lists without running them (`ast.literal_eval`), put
  the text in NFC, and linked each value to its Devanagari spellings.
- Licence notice:

```
MIT License

Copyright (c) 2026 Atharva Kusumbia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## What is our own work
- the colloquial and hundreds-style rules (see report.md);
- choosing the main spelling of each word;
- the spellings marked `composed (this work)` or `rule (this work)`;
- the value checker.

## Independent check (`check_report.json`)
A second job checked `price_words.json` with tools the build did not rely on, and corrected it.
Nothing from these tools is included in this folder; they were only run on the phrases.
- NVIDIA NeMo text processing 1.2.0 (Apache-2.0), https://github.com/NVIDIA/NeMo-text-processing ,
  PyPI `nemo_text_processing==1.2.0` with `pynini==2.1.6.post1`: the Hindi and Marathi inverse
  text normalization grammars (`CardinalFst`) turned each Devanagari phrase back into a number.
- gpu-sankhya 0.8.0 (MIT), npm, downloaded again: `parse()` on every romanized Hindi string.
- Unicode CLDR `mr.xml` (see source 3): the Marathi spell-out rules, re-implemented in
  `scripts/check_price_words.py` and compared with the Marathi formal forms.
- Dakshina Wikipedia text (see source 1): only counts of a few number phrases.
- A hand-typed list of Hindi and Marathi number words 1-99 written for the check.

What the check changed in `price_words.json` (88 changes, all listed in `check_report.json`):
it removed 72 thousands + fractional-hundreds forms (e.g. `दो हज़ार ढाई सौ`) as doubtful,
fixed one romanization (76: `chihttr` → `chhihattar`), removed one doubtful Marathi spelling
(`saet` for सात) from 12 entries, and fixed the note on 3 entries. It also added
`check.independent_parser` and `check.nemo_itn` to every form. `build_stats.json` was not changed:
it still describes the build before the check.
