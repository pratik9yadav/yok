Since we started entering product names in Hindi, `catalog.py` has been giving us trouble.
The catalog my nephew saved from Excel as "CSV UTF-8" fails to load with `KeyError: 'sku'`, and on the old billing PC (its system language isn't set to UTF-8) saving the catalog crashes with `UnicodeEncodeError`.
A customer searched for ज़ीरा on the phone and got nothing, although ज़ीरा (cumin) is right there in the catalog.
And `receipt_label` sometimes crashes with `UnicodeDecodeError` for Hindi names; when it doesn't crash, the printer sometimes shows a letter with its matra chopped off.
Please fix these; the other code calls `load_catalog`, `save_catalog`, `search` and `receipt_label` exactly as they are now, and English names must keep working the same way.
