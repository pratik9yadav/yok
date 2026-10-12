# Real data: sources, licences, attribution

Files in this folder come from open datasets. They are kept separate from `data/train.jsonl`
until they have been checked, and they are never mixed into `data/test.jsonl` or
`data/hard_test.jsonl`.

| File | Source | Licence | What was changed |
|---|---|---|---|
| `products_flipkart.jsonl` | [Flipkart Products](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products) by PromptCloud | CC BY-SA 4.0 | Only Automotive, Tools & Hardware and Home Improvement; name shortened; MRP, price, brand, category kept. This file is CC BY-SA 4.0. |
| `products_amazon_in.jsonl` | [Amazon India Products 2023](https://www.kaggle.com/datasets/asaniczka/amazon-india-products-2023-1-5m-products) by asaniczka | as reported by Kaggle at download time (kept only if open; see `fetch_report.json`) | Only car/bike, tools, electrical, lighting and similar categories; name shortened; price and list price kept. |
| `indicvoices_shop_talk.jsonl` | [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) by AI4Bharat (Javed et al., 2024) | CC BY 4.0 | Hindi and Marathi "Conversation" transcripts only; lines that look like shop talk; lines with phone numbers or e-mail removed; no speaker information kept. |
| `xpqa_hi_examples.jsonl` | [xPQA](https://github.com/amazon-science/contextual-product-qa) by Amazon Science (Shen et al., 2023), Hindi part | CDLA-Sharing-1.0 (`LICENSE-CDLA-Sharing-1.0.txt`) | 151 real Hindi answers about stock, colours/sizes and price, turned into training rows and labelled twice (builder + independent checker). Details: `NOTICE-xpqa.md`. Any file that includes these rows must keep this licence. |
| `price_words.json` | AI4Bharat indic-numtowords (MIT), Unicode CLDR (Unicode licence), gpu-sankhya (MIT), Google Dakshina (CC BY-SA 4.0) | mixed, see `NOTICE-price_words.md` | How 510 prices are said in Hindi and Marathi (formal, colloquial like डेढ़ सौ / दीडशे, romanized); every form parsed back to its number. |
| `romanization_variants.json` | Google Dakshina and COMI-LINGUA | CC BY-SA 4.0 | How 288 shop-call words are spelled in Latin letters, with counts; 110 wrong spellings removed by a second check. Details: `NOTICE-romanization_variants.md`, `romanization_variants_report.md`. |

Download script: `fetch_real_data.py`, run by `.github/workflows/fetch_data.yml`.
