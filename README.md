# Train your own model (free, on Google Colab)

This repo trains a small free model (Qwen3 4B) to read what a shopkeeper said on a call and
return JSON:

```
Product: 16A MCB
Shopkeeper: haan hai, 220 ka, delivery nahi
→ {"in_stock": true, "price_inr": 220, "delivery": false, "alternative": null}
```

## Run it
1. Open [`train_colab.ipynb` in Colab](https://colab.research.google.com/github/pratik9yadav/yok/blob/main/train_colab.ipynb)
   (on a phone: keep Colab open and the screen on while it runs).
2. **Runtime → Run all**. Allow Google Drive when asked. Wait 30–40 minutes.
3. **File → Save a copy in GitHub** → same repo, branch, and path. The results are then in git.
4. On your PC: download the `wimb-shop-model` folder from Google Drive, then
   `ollama create wimb-shop -f Modelfile` and `ollama run wimb-shop`.

## Files
| File | What it is |
|---|---|
| `train_colab.ipynb` | The notebook. Built by `build_notebook.py`, don't edit by hand. |
| `data/train.jsonl` | 400 **made-up** training examples (Hindi, Marathi, English). Made by `make_data.py`. |
| `data/test.jsonl` | 30 hand-written test calls, never used for training. |
| `score.py` | The prompt, and how answers are scored. |
| `test_training.py` | Checks the data and scoring: `python -m pytest` |

## Use real data
The sample data is made up, to prove training works. For a real model, put real call
transcripts in `data/train.jsonl` (same format, one per line), then run
`python build_notebook.py` and train again.
