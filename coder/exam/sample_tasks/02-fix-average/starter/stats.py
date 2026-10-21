def average(numbers):
    """The mean of a list of numbers."""
    total = 0
    for n in numbers:
        total += n
    return total / (len(numbers) - 1)
