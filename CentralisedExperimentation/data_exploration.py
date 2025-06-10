import glob
import pandas as pd
import os

def analyze_labels(file_pattern):
    folder_path = "/media/ketanatri/72B88D23B88CE747/Users/Ketan/Desktop/PhD/FLOpsInfraDrift/RunArtifacts/ConsolidatedData/TrainTestData"
    files = glob.glob(os.path.join(folder_path, file_pattern))
    total_counts = None
    selected_prefixes = ('0', '1', '2', '4')
    for file in files:
        df = pd.read_csv(file)
        if 'label' not in df.columns:
            print(f"File {file} does not contain a 'label' column.")
            continue
        label_counts = df['label'].value_counts()
        total = len(df)
        print(f"\nFile: {os.path.basename(file)}")
        for label, count in label_counts.sort_index().items():
            percent = (count / total) * 100
            print(f"Label: {label} | Count: {count} | Percentage: {percent:.2f}%")
        print(f"Total samples: {total}")

        # Sum counts for selected files
        if os.path.basename(file).startswith(selected_prefixes):
            if total_counts is None:
                total_counts = label_counts.copy()
            else:
                total_counts = total_counts.add(label_counts, fill_value=0)

    if total_counts is not None:
        total_sum = int(total_counts.sum())
        print("\n--- SUMMED COUNTS FOR FILES STARTING WITH 0, 1, 2, 4 ---")
        for label, count in total_counts.sort_index().items():
            percent = (count / total_sum) * 100
            print(f"Label: {label} | Count: {int(count)} | Percentage: {percent:.2f}%")
        print(f"Total samples: {total_sum}")

def create_balanced_csv(file_pattern, output_csv, label_counts_dict):
    folder_path = "/media/ketanatri/72B88D23B88CE747/Users/Ketan/Desktop/PhD/FLOpsInfraDrift/RunArtifacts/ConsolidatedData/TrainTestData"
    selected_prefixes = ('0', '1', '2', '4')
    files = [f for f in glob.glob(os.path.join(folder_path, file_pattern))
             if os.path.basename(f).startswith(selected_prefixes)]
    dfs = [pd.read_csv(f) for f in files]
    # For each label, collect indices from each file
    selected_rows = []
    for label, target_count in label_counts_dict.items():
        label = int(label)
        label_indices_per_file = [df[df['label'] == label].index.tolist() for df in dfs]
        picked = 0
        file_ptrs = [0] * len(dfs)
        while picked < target_count:
            for i, indices in enumerate(label_indices_per_file):
                if picked >= target_count:
                    break
                if file_ptrs[i] < len(indices):
                    idx = indices[file_ptrs[i]]
                    selected_rows.append(dfs[i].iloc[[idx]])
                    file_ptrs[i] += 1
                    picked += 1
        # If not enough samples, will just pick as many as available
    # Concatenate and save
    if selected_rows:
        result_df = pd.concat(selected_rows, ignore_index=True)
        result_df.to_csv(output_csv, index=False)
        print(f"Balanced CSV saved to {output_csv} with {len(result_df)} rows.")
    else:
        print("No rows selected. Check your label counts and files.")

def create_split_csvs(file_pattern, output_dir, label_counts_dict):
    folder_path = "/mnt/Users/Ketan/Desktop/PhD/FLOpsInfraDrift/RunArtifacts/ConsolidatedData/TrainTestData/"
    selected_prefixes = ('0', '1', '2', '4')
    files = [f for f in glob.glob(os.path.join(folder_path, file_pattern))
             if os.path.basename(f).startswith(selected_prefixes)]
    print(f"Found {len(files)} files matching pattern '{file_pattern}' with selected prefixes {selected_prefixes}.")
    dfs = [pd.read_csv(f) for f in files]

    # Prepare a dict to collect rows for each prefix
    selected_rows_per_prefix = {prefix: [] for prefix in selected_prefixes}

    for label, target_count in label_counts_dict.items():
        label = int(label)
        print(f"Processing label {label} with target count {target_count}.")
        label_indices_per_file = [df[df['label'] == label].index.tolist() for df in dfs]
        picked = 0
        file_ptrs = [0] * len(dfs)
        while picked < target_count:
            for i, indices in enumerate(label_indices_per_file):
                if picked >= target_count:
                    break
                if file_ptrs[i] < len(indices):
                    idx = indices[file_ptrs[i]]
                    prefix = os.path.basename(files[i]).split('-')[0]
                    selected_rows_per_prefix[prefix].append(dfs[i].iloc[[idx]])
                    file_ptrs[i] += 1
                    picked += 1
        print(f"Picked {picked} rows for label {label}.")

    # Concatenate and save for each prefix
    for prefix in selected_prefixes:
        rows = selected_rows_per_prefix[prefix]
        if rows:
            result_df = pd.concat(rows, ignore_index=True)
            output_csv = os.path.join(output_dir, f"{prefix}_subset.csv")
            result_df.to_csv(output_csv, index=False)
            print(f"Subset CSV for prefix {prefix} saved to {output_csv} with {len(result_df)} rows.")
        else:
            print(f"No rows selected for prefix {prefix}. Check your label counts and files.")

if __name__ == "__main__":
    # for suffix in ['train', 'test']:
    #     print(f"\n--- {suffix.upper()} DATASETS ---")
    #     analyze_labels(f"*-{suffix}-data.csv")

    # Only for train data
    label_counts = {
        0: 310,
        1: 316,
        2: 161,
        3: 636,
        4: 263,
        5: 905,
        6: 482,
        7: 218,
        8: 242,
        9: 347
    }
    create_split_csvs("*-train-data.csv",
                        "/home/ketanatri/Desktop/PhD/SplitData",
                        label_counts)