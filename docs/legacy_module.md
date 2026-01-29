# legacy_module

## Summary
Utility functions for basic data processing and aggregation.

## Functions

### process_data(data)
- Filters out null values
- Doubles numeric entries
- Returns a list of processed values

### average(values)
- Computes the arithmetic mean
- Returns None for empty input

## Usage Example
```python
from examples.legacy_module import process_data, average

data = [1, None, 3]
processed = process_data(data)
avg = average(processed)
