# legacy_module.py

def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] is not None:
            result.append(data[i] * 2)
    return result


def average(values):
    if not values:
        return None

    total = 0
    for v in values:
        total += v
    return total / len(values)
