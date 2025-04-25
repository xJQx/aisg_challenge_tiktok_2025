import os

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
        df = df[["qid", "pred"]]
        df.fillna('Video not found.', inplace=True)
        print('After filling:', len(df))
        df.to_csv(out_fpath, index=False)

        print(f"Missing values filled and saved to {out_fpath}")


class Combine:
    def __init__(self):
        pass

    def run(self):
        f = "/Users/mouse/Documents/GitHub/aisg_challenge_tiktok_2025-phase1_processing/src/subs"
        lst = []
        for _f in os.listdir(f):
            if _f.endswith(".csv"):
                #print(_f)
                df = pd.read_csv(os.path.join(f, _f))
                df = df[["qid", "pred"]]
                lst.append(df)
        df = pd.concat(lst)
        df2 = pd.read_csv("/Users/mouse/Documents/GitHub/aisg_challenge_tiktok_2025-phase1_processing/src/submission_notfull.csv")
        # remove entries in df2 that are in df
        df2 = df2[~df2.qid.isin(df.qid)]
        df2 = pd.concat([df, df2])

        # Remove duplicates, retain first
        df2 = df2.drop_duplicates(subset=["qid"], keep="first")

        df2.to_csv("/Users/mouse/Documents/GitHub/aisg_challenge_tiktok_2025-phase1_processing/src/submission_new.csv", index=False)

if __name__ == "__main__":
    # Example usage
    Combine().run()
    filler = FillMissing()
    filler.run("/Users/mouse/Documents/GitHub/aisg_challenge_tiktok_2025-phase1_processing/src/submission_new.csv", "/Users/mouse/Documents/GitHub/aisg_challenge_tiktok_2025-phase1_processing/src/submission.csv")
