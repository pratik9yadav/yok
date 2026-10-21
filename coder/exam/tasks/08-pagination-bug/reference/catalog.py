"""Product listing helpers for the shop app."""

import math


def paginate(items, page, per_page=20):
    """Return one page of items.

    page is 1-based. The result is a dict with the page's items, the page
    number, the total number of pages and whether there is a next page.
    """
    if page < 1:
        raise ValueError("page must be 1 or more")
    if per_page < 1:
        raise ValueError("per_page must be 1 or more")
    total_pages = math.ceil(len(items) / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "items": list(items[start:end]),
        "page": page,
        "total_pages": total_pages,
        "has_next": page < total_pages,
    }


def search_products(products, query, page=1, per_page=20):
    """Products whose name contains query (ignoring case), one page at a time."""
    q = query.strip().casefold()
    matches = [p for p in products if q in p["name"].casefold()]
    return paginate(matches, page, per_page)
