import csv

# Helper to save dataset to CSV with all tensor values expanded
def save_dataset_to_csv(dataset, filename):
    if not dataset:
        return

    # Expand all keys and flatten any tensor/list/array values
    def flatten_item(item):
        flat = {}
        for k, v in item.items():
            # Check for tensor-like or list/array values
            if hasattr(v, 'tolist'):
                v = v.tolist()
            if isinstance(v, (list, tuple)):
                for i, val in enumerate(v):
                    flat[f"{k}_{i}"] = val
            else:
                flat[k] = v
        return flat

    # Flatten all items and collect all fieldnames
    flat_dataset = [flatten_item(item) for item in dataset]
    fieldnames = ["index"]
    for item in flat_dataset:
        for k in item.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, item in enumerate(flat_dataset):
            row = {"index": idx}
            row.update(item)
            writer.writerow(row)

