> **In this repository:** `report.md` below is `romanization_variants_report.md`.

# NOTICE: romanization variants table

Files in this folder:

| file | what it is |
|---|---|
| `romanization_variants.json` | derived table: Devanagari word -> attested romanized spellings, counts, source, weights |
| `word_list.tsv` | the list of shop-call words and phrases (original work for this project) |
| `build_variants.py` | the script that builds the table (original work for this project) |
| `build_log.json` | build statistics and the list of spellings the filters dropped |
| `report.md` | coverage report and known problems (numbers are from the build, before the check) |
| `check_report.json` | independent check: spellings removed and why, recounts against the sources |

## Licence of this folder

The table, the word list, the script's output and the build log are an adaptation of the two
datasets below and are licensed under **Creative Commons Attribution-ShareAlike 4.0 International
(CC BY-SA 4.0)**, https://creativecommons.org/licenses/by-sa/4.0/ . The word list and the script are
offered under the same licence for simplicity.

Suggested attribution when reusing:

> Romanization variants table (yok / WIMB project), derived from the Dakshina dataset v1.0
> (Google Research, CC BY-SA 4.0) and the COMI-LINGUA text-normalization data (Sheth, Beniwal and
> Singh, lingo-iitgn, IIT Gandhinagar, CC BY-SA 4.0). Changed: only single-word spelling counts
> were extracted. Licensed CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/

No warranty: the source data is provided "As Is", and so is this table.

## Source 1: Dakshina dataset v1.0 (Google Research)

- Project: https://github.com/google-research-datasets/dakshina
- Download used: https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar
  (2,008,340,480 bytes, last modified 2020-05-27); only the `hi/` and `mr/` lexicon and romanized
  folders were extracted.
- Files used: `{hi,mr}/lexicons/{hi,mr}.translit.sampled.{train,dev,test}.tsv` and
  `{hi,mr}/romanized/{hi,mr}.romanized.rejoined.aligned.cased_nopunct.tsv`.
- Licence: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/ (stated in the dataset
  README, section "License"). Dakshina's text is itself derived from Wikipedia (CC BY-SA); no
  Wikipedia or Dakshina sentence is reproduced here.
- Attribution / citation requested by the authors (Google Research):

  Brian Roark, Lawrence Wolf-Sonkin, Christo Kirov, Sabrina J. Mielke, Cibu Johny, Işın Demirşahin
  and Keith Hall. 2020. "Processing South Asian Languages Written in the Latin Script: the Dakshina
  Dataset." In Proceedings of the 12th Language Resources and Evaluation Conference (LREC),
  pages 2413-2423. https://www.aclweb.org/anthology/2020.lrec-1.294

## Source 2: COMI-LINGUA, text-normalization (TN) subset (lingo-iitgn, IIT Gandhinagar)

- Project: https://github.com/lingo-iitgn/COMI-LINGUA (project site https://lingo.iitgn.ac.in/codemixing/)
- Files used: https://raw.githubusercontent.com/lingo-iitgn/COMI-LINGUA/main/TN_train.csv and
  https://raw.githubusercontent.com/lingo-iitgn/COMI-LINGUA/main/TN_test.csv (29,547 rows in total).
- Licence: the repository's own licence file,
  https://raw.githubusercontent.com/lingo-iitgn/COMI-LINGUA/main/LICENSE.txt , is the full text of
  **CC BY-SA 4.0** (checked 2026-10-12), and the badge at the top of the README also reads
  "CC BY SA 4.0" (the badge itself links to the Apache-2.0 LICENSE of a different repository,
  lingo-iitgn/commentator). The README text says CC BY 4.0 (twice). We follow the stricter reading,
  CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/ ; this folder is CC BY-SA 4.0, which
  also meets CC BY 4.0 if that is what the authors meant.
- Attribution / citation requested by the authors:

  Rajvee Sheth, Himanshu Beniwal and Mayank Singh. 2025. "COMI-LINGUA: Expert Annotated Large-Scale
  Dataset for Multitask NLP in Hindi-English Code-Mixing." In Findings of the Association for
  Computational Linguistics: EMNLP 2025, pages 7973-7992, Suzhou, China.
  https://aclanthology.org/2025.findings-emnlp.422/

## What was changed

- Nothing from either dataset is copied as text beyond single words: no sentence, comment or
  fragment of a comment from COMI-LINGUA or Dakshina appears in any file here.
- From Dakshina, the romanizations of words in `word_list.tsv` were looked up in the lexicons
  (with their attestation counts) and counted in the token-aligned romanized sentences.
- From COMI-LINGUA, each noisy romanized comment was aligned word by word to the three annotators'
  normalized versions; when at least 2 of 3 annotators normalized a noisy word (or short run of
  words) to a word-list spelling, the noisy spelling was counted. Only these counts are kept.
- Spellings were lower-cased, punctuation was removed, and implausible or wrong-sense spellings
  were filtered out (listed in `build_log.json`); COMI-LINGUA spellings seen only once were dropped.
- Weights per word were computed from the counts (see `report.md`).
- An independent check (2026-10-12) removed 110 (word, spelling) pairs that were another word,
  another form of the word, an unrelated English word or junk, and recomputed the weights with the
  same formula. The removed pairs and the reasons are listed in `check_report.json`; re-running
  `build_variants.py` alone does not apply these removals.
- No personal data: every entry is a common Hindi or Marathi word or an English loanword; names,
  handles, numbers and other personal data are not part of the output.
