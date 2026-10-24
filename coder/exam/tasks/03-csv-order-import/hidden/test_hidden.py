import time
from decimal import Decimal

import pytest

from order_import import import_orders

HEADER = "order_id,customer,sku,qty,price\n"


def write(tmp_path, text, name="orders.csv", encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return p


def test_good_file(tmp_path):
    p = write(tmp_path, HEADER + "A1,Ravi,ATTA-5KG,2,245.50\nA2,Meena,SALT-1KG,1,28\n")
    orders, errors = import_orders(p)
    assert errors == []
    assert orders == [
        {"order_id": "A1", "customer": "Ravi", "sku": "ATTA-5KG", "qty": 2, "price": Decimal("245.50")},
        {"order_id": "A2", "customer": "Meena", "sku": "SALT-1KG", "qty": 1, "price": Decimal("28")},
    ]
    assert isinstance(orders[0]["qty"], int)
    assert isinstance(orders[0]["price"], Decimal)


def test_accepts_str_path(tmp_path):
    p = write(tmp_path, HEADER + "A1,Ravi,ATTA-5KG,2,245.50\n")
    orders, errors = import_orders(str(p))
    assert len(orders) == 1 and errors == []


def test_column_order_and_extra_columns(tmp_path):
    p = write(tmp_path, "price,notes,qty,sku,customer,order_id\n10.00,urgent,3,DAL-1KG,Asha,B7\n")
    orders, errors = import_orders(p)
    assert errors == []
    assert orders == [{"order_id": "B7", "customer": "Asha", "sku": "DAL-1KG", "qty": 3, "price": Decimal("10.00")}]


def test_excel_bom_and_spaces_in_header(tmp_path):
    text = "order_id, customer, sku, qty, price\r\nA1, Ravi, ATTA-5KG, 2, 245.50\r\n"
    p = write(tmp_path, text, encoding="utf-8-sig")
    orders, errors = import_orders(p)
    assert errors == []
    assert len(orders) == 1
    assert orders[0]["order_id"] == "A1"
    assert orders[0]["sku"] == "ATTA-5KG"
    assert orders[0]["qty"] == 2
    assert orders[0]["price"] == Decimal("245.50")


def test_bad_rows_are_reported_and_skipped(tmp_path):
    text = HEADER + "\n".join([
        "A1,Ravi,ATTA-5KG,2,245.50",   # line 2 ok
        "A2,Meena,SALT-1KG,two,28",    # line 3 qty not a number
        "A3,Asha,DAL-1KG,0,90",        # line 4 qty zero
        "A4,Asha,DAL-1KG,-2,90",       # line 5 negative
        "A5,Asha,DAL-1KG,1.5,90",      # line 6 not whole
        "A6,Ravi,OIL-1L,1,abc",        # line 7 bad price
        "A7,Ravi,OIL-1L,1,-10",        # line 8 negative price
        "A8,Ravi,OIL-1L,1,180",        # line 9 ok
    ]) + "\n"
    orders, errors = import_orders(write(tmp_path, text))
    assert [o["order_id"] for o in orders] == ["A1", "A8"]
    assert [line for line, _ in errors] == [3, 4, 5, 6, 7, 8]
    assert all(isinstance(reason, str) and reason for _, reason in errors)


def test_missing_values_and_short_rows(tmp_path):
    text = HEADER + "A1,,ATTA-5KG,2,245.50\n,Ravi,ATTA-5KG,2,245.50\nA3,Ravi,ATTA-5KG\nA4,Ravi,SALT-1KG,1,20\n"
    orders, errors = import_orders(write(tmp_path, text))
    assert [o["order_id"] for o in orders] == ["A4"]
    assert [line for line, _ in errors] == [2, 3, 4]


def test_duplicate_order_id(tmp_path):
    text = HEADER + "A1,Ravi,ATTA-5KG,2,245.50\nA2,Meena,SALT-1KG,1,28\nA1,Ravi,ATTA-5KG,2,245.50\n"
    orders, errors = import_orders(write(tmp_path, text))
    assert [o["order_id"] for o in orders] == ["A1", "A2"]
    assert [line for line, _ in errors] == [4]


def test_quotes_unicode_and_blank_lines(tmp_path):
    text = HEADER + '"A1","Sharma, Ravi",ATTA-5KG,2,245.50\n\nA2,राम किराना,SALT-1KG,1,28\n'
    orders, errors = import_orders(write(tmp_path, text))
    assert errors == []
    assert [o["customer"] for o in orders] == ["Sharma, Ravi", "राम किराना"]


@pytest.mark.parametrize("text", ["order_id,customer,sku,qty\nA1,Ravi,X,1\n", "id;name;sku\n1;2;3\n", ""])
def test_missing_columns_raise_value_error(tmp_path, text):
    with pytest.raises(ValueError):
        import_orders(write(tmp_path, text))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_orders(tmp_path / "nope.csv")


def test_header_only_file(tmp_path):
    assert import_orders(write(tmp_path, HEADER)) == ([], [])


def test_no_state_between_calls(tmp_path):
    p = write(tmp_path, HEADER + "A1,Ravi,ATTA-5KG,2,245.50\n")
    first = import_orders(p)
    second = import_orders(p)
    assert first == second
    assert second[1] == []  # A1 in the second import is not a duplicate of the first import


def test_large_file_is_fast(tmp_path):
    rows = [f"O{i},Customer {i % 50},SKU-{i % 300},{1 + i % 5},{i % 1000}.25" for i in range(50_000)]
    p = write(tmp_path, HEADER + "\n".join(rows) + "\n")
    start = time.perf_counter()
    orders, errors = import_orders(p)
    assert time.perf_counter() - start < 3.0
    assert len(orders) == 50_000 and errors == []
