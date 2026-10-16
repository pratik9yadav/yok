from bench.prices import parse_price

# Test the original case that was failing
try:
    result = parse_price('₹1,250.50')
    print('Success:', result)
    assert result == 1250.5
    print('Assertion passed!')
except Exception as e:
    print('Error:', type(e).__name__, str(e))
    
# Test existing functionality still works
try:
    assert parse_price("1250") == 1250.0
    assert parse_price("Rs. 1250") == 1250.0
    assert parse_price("1,250.50") == 1250.5
    print("All existing tests pass")
except Exception as e:
    print('Error in existing tests:', type(e).__name__, str(e))
