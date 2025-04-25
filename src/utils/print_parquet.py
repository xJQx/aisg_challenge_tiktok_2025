import pandas as pd

def print_parquet(file_path: str, num_rows: int = 10):
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)

    # Print the first 'num_rows' of the DataFrame
    print(df.head(num_rows))

    # If the file is large, you could also add a line to print more info (e.g., number of rows/columns)
    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {df.shape[1]}")
    print("\nColumn names:", df.columns.tolist())

if __name__ == "__main__":
    print_parquet('/root/combined_foi/combined_foi.parquet', 20)