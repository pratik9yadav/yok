def average(numbers):
    """The mean of a list of numbers. ValueError when there are none."""
    numbers = list(numbers)
    if not numbers:
        raise ValueError("average() needs at least one number")
    return sum(numbers) / len(numbers)
