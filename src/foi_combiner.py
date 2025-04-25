import os.path
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm


def combine_foi_parquets(_dir: str, output_dir: str):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # List to hold all DataFrames
    dfs = []

    # Get all parquet files in the directory
    parquet_files = [f for f in os.listdir(_dir) if f.endswith('.parquet')]

    for fpath in tqdm(parquet_files):
        # Construct the full file path
        full_fpath = os.path.join(_dir, fpath)

        # Read each Parquet file into a DataFrame
        df = pd.read_parquet(full_fpath)

        # Add columns to the DataFrame
        qid = fpath.split('_')[0]
        video_id = fpath.split('_')[1]

        df['qid'] = qid
        df['video_id'] = video_id

        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Output path
    output_path = os.path.join(output_dir, "combined_foi.parquet")

    # Save the combined DataFrame to a Parquet file
    combined_df.to_parquet(output_path, index=False)

    print(f"Combined Parquet file saved to: {output_path}")

def combine_combined_foi_parquets(file1_path: str, file2_path: str, output_dir: str):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Read the two Parquet files into DataFrames
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    # Concatenate the DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Save the combined DataFrame to a new Parquet file
    output_path = os.path.join(output_dir, "final_combined_foi.parquet")
    combined_df.to_parquet(output_path, index=False)

    print(f"Combined Parquet file saved to: {output_path}")


if __name__ == "__main__":
    # combine_foi_parquets('/root/outputs', "/root")
    combine_combined_foi_parquets("/root/combined_foi/combined_foi_1.parquet", "/root/combined_foi/combined_foi_2.parquet", "/root/combined_foi/")