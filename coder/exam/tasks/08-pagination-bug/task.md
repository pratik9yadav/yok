Customers say the product list on our app is broken: page 1 never shows the first products, and the last few products never show up on any page. Searching is also odd: searching "atta" doesn't find "Aashirvaad Atta 5kg", though searching "Atta" does.
The code is in `catalog.py` (`paginate(items, page, per_page=20)` and `search_products(products, query, page=1, per_page=20)`). Please fix it.
