import pandas as pd
from datasets import load_dataset


class FillMissing:
    def __init__(self):
        self.benchmark = load_dataset("lmms-lab/AISG_Challenge", split="test")

    def run(self, in_fpath: str, out_fpath: str):
        """
        Fill missing values in the dataset.
        """
        # Load the input dataset
        benchmark_df = self.benchmark.to_pandas()

        benchmark_df = benchmark_df[["qid"]]

        df = pd.read_csv(in_fpath)
        print('Before filling:', len(df))
        df = df.merge(benchmark_df, on="qid", how="right")
        df.fillna('Video not found.', inplace=True)
        print('After filling:', len(df))
        df.to_csv(out_fpath, index=False)

        print(f"Missing values filled and saved to {out_fpath}")

if __name__ == "__main__":
    # Example usage
    filler = FillMissing()
    filler.run("/Users/mouse/Documents/GitHub/aisg_challenge_tiktok_2025-phase1_processing/src/submission_notfull.csv", "/Users/mouse/Documents/GitHub/aisg_challenge_tiktok_2025-phase1_processing/src/submission.csv")
