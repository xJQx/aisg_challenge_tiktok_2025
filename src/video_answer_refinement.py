from concurrent.futures import as_completed, ProcessPoolExecutor

import pandas as pd
from datasets import load_dataset
import json
import csv

from tqdm import tqdm

from src.utils.call_qwen2_model import call_qwen2_model
from src.config import OUTPUT_DIR, SUBMISSION_DIR


LOWER = 200
UPPER = 400

# s0: 0, 200; 1000, 1200
# s2: 400, 800; 200, 400
# s3: 800, 1000; 1200, 1344


class VideoAnswerRefinement:
    """ Question and Answering
    Handles Question & Answering using a 3-stage process:
      0-shot: Determine question type (MCQ or Open-Ended)
      1-shot: Validate Open-Ended questions for contextual fit
      2-shot: Answer based on question type

    Model: bartowski/Qwen2-72B-Instruct-GGUF (GH200)

    0 Shot: figure type of question (open ended or mcq)
    - Is the question Open-Ended or MCQ? If it is Open-Ended, answer 'OE'. If it is MCQ, answer it as 'MCQ'.

    1 Shot:  (Ask only if it is OE): Does the question makes sense in relation to the context given
      - Does it make sense? Does what the qns ask for exist in the video? If not, what is the most relevant entity that exists in the video instead.

    2 Shot:
      If MCQ: State your answer and explain in a step-by step manner. Follow the given format strictly when responding:
          ANSWER: {OPTION}
          EXPLAINATION:\n{EXPLANATION}
      If OE: First, answer the sub-questions. Then, use your answer for the sub-questions to answer the main-question.
    """

    def __init__(self, call_model, annot_data_fpath, _range):
        self.range = _range
        self.call_model = call_model
        self.all_pred = {}
        self.annot_data = pd.read_parquet(annot_data_fpath)
        sub_fpath = SUBMISSION_DIR / "submission.csv"
        if sub_fpath.exists():
            self.processed = pd.read_csv(sub_fpath)
        else:
            self.processed = pd.DataFrame({'qid': [], 'video_id': []})

        self.benchmark = load_dataset("lmms-lab/AISG_Challenge", split="test")
        self.to_test = self.get_examples_that_exist()

    def load_processed_examples(self):
        return self.processed['qid'].to_list()

    def get_examples_that_exist(self):
        # Filter out examples that do not exist in the benchmark dataset
        processed = self.load_processed_examples()
        lst = []
        for example in tqdm(self.benchmark):
            qid = example["qid"]
            _qid = list(example["qid"].split('-'))
            if qid not in processed and int(_qid[0]) > self.range[0] and int(_qid[0]) <= self.range[1]:
                vid = example["video_id"]
                out_path = OUTPUT_DIR / f"{qid}_{vid}.json"
                if out_path.exists():
                    lst.append(example)

        return lst

    def _zero_shot(self, question: str) -> str:
        """
          0 Shot: figure type of question (open ended or mcq)
            - Is the question Open-Ended or MCQ? If it is Open-Ended, answer 'OE'. If it is MCQ, answer it as 'MCQ'.
        """
        # print("\tZero Shot...")
        prompt = f"""Goal: Figure out the type of question\n
      Is the question Open-Ended or MCQ? If it is Open-Ended, answer 'OE'. If it is MCQ, answer it as 'MCQ'.\n
      Question={question}

      Follow the following format strictly. Do not include the square brackets '[' and ']'
      Answer=[MCQ / OE]"""

        response = self.call_model(prompt).strip()
        response = response.split('Answer=')[-1].split('\n')[0]
        return response

    def __validate_zero_shot_output(self, zero_shot_output: str) -> bool:
        if zero_shot_output == "MCQ" or zero_shot_output == "OE":
            return True
        return False

    def _one_shot(self, question: str, context: str) -> str:
        """
        Step 1: For OE questions — check if it makes sense with the context.
        """
        # print("\tOne Shot...")
        prompt = f"""Goal: Determine whether this question makes sense in the context of the video.\n
      Context:\n{context}\n
      Question:\n{question}\n
      Does the question makes sense and does the question ask for exist in the video? If not, what is the most relevant entity that exists in the video instead."""
        response = self.call_model(prompt).strip()
        return response

    def _two_shot(self, keyf, question_type: str, question: str, context: str, sub_questions: str, sense_check: str) -> str:
        """
        Step 2: Answer the question based on its type.
        If MCQ: State your answer and explain in a step-by step manner. Follow the given format strictly when responding:
            ANSWER: {OPTION}
            EXPLAINATION:\n{EXPLANATION}
        If OE: First, answer the sub-questions. Then, use your answer for the sub-questions to answer the main-question.
        """
        # print("\tTwo Shot...")
        # MCQ Question
        if question_type == "MCQ":
            prompt = f"""Context:\n{context}\nMain Question:\n{question}\nSub-Questions:\n{sub_questions}\nKeyFrames:{keyf}\nInstructions: Given the context and keyframes, state your multiple-choice answer and explain in a step-by-step manner in the explanation section. Follow the format strictly when responding:\nEXPLANATION:\n[EXPLANATION]\nANSWER:\n[OPTION]\n"""

        # Open-Ended Question
        elif question_type == "OE":
            prompt = f"""Context:\n{context}\nMain Question:\n{question}\nSense Check:{sense_check}\nSub-Questions:\n{sub_questions}\nKeyFrames:{keyf}\nInstructions: Given the context and keyframes, first answer the sub-questions, then use your answers from the sub-questions to help you answer the main question. Explain your answers in a step by step manner in the explanation section. Follow the format strictly when responding:\nEXPLANATION:\n[EXPLANATION]\nANSWER:\n[OPEN-ENDED ANSWER]\n"""

        else:
            return "Invalid question type."

        return self.extract_answer(self.call_model(prompt))

    @staticmethod
    def extract_answer(prompt):
        lines = prompt.split("\n")
        return lines[lines.index('ANSWER:') + 1]

    def _save_result(self, qid: str, pred: str):
        self.all_pred[qid] = pred

        print("Exporting submission results to csv...")
        submission_path = SUBMISSION_DIR / f"submission.csv"
        with open(submission_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "pred"])
            writer.writeheader()  # header
            for qid, pred in self.all_pred.items():
                writer.writerow({"qid": qid, "pred": pred})

        print(f"✓ Done: {submission_path}")

    def _process_wrapper(self, example):
        """
        Helper to call self.process and return (qid, answer).
        Exceptions will propagate so we can catch them per‐future.
        """
        qid = example["qid"]
        pred = self.process(example)
        return qid, pred

    def batch_process(self, batch_size: int, num_workers: int = None):
        """
        Run self.process() over all examples in parallel.

        :param batch_size: (unused here — you could slice self.to_test into
                           chunks of this size if desired)
        :param num_workers: number of processes; defaults to cpu_count().
        """
        qids = []
        preds = []
        errors = []

        # decide number of workers
        num_workers = num_workers or None  # ProcessPoolExecutor will pick cpu_count()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # submit all tasks
            future_to_example = {
                executor.submit(self._process_wrapper, ex): ex
                for ex in self.to_test
            }

            # as each finishes, gather result or exception
            for fut in tqdm(as_completed(future_to_example),
                            total=len(future_to_example),
                            desc="Processing examples"):
                ex = future_to_example[fut]
                qid = ex["qid"]
                try:
                    qid_out, pred = fut.result()
                    qids.append(qid_out)
                    preds.append(pred)
                except Exception as e:
                    errors.append({qid: repr(e)})

        # build DataFrame and write out as before
        df = pd.DataFrame({"qid": qids, "pred": preds})
        df = pd.concat([self.processed, df], ignore_index=True)
        df.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

        # Save errors
        with open(SUBMISSION_DIR / "errors.json", "w") as f:
            json.dump(errors, f, indent=2)

        print(f"✓ Done.  {len(qids)} succeeded, {len(errors)} failed.")

    """def batch_process(self, batch_size):
      # Batch process the examples
      qid_lst = []
      pred_lst = []
      error_lst = []
      for ex in self.to_test:
        try:
          qid_lst.append(ex["qid"])
          pred_lst.append(self.process(ex))
        except Exception as e:
          error_lst.append({ex["qid"]: e})
  
      df = pd.DataFrame({
  
          "qid": qid_lst,
          "pred": pred_lst
      })
  
      # Save errors to file
      with open(SUBMISSION_DIR / "errors.json", "w") as f:
        json.dump(error_lst, f, indent=2)
  
      df.to_csv(SUBMISSION_DIR / "submission.csv", index=False)"""

    def process(self, example):
        qid = example["qid"]
        vid = example["video_id"]
        out_path = OUTPUT_DIR / f"{qid}_{vid}.json"
        keyf = self.annot_data[self.annot_data['qid'] == qid]
        print(f"\nProcessing {qid}, video {vid}…")

        if out_path.exists():
            # READ from video processor json output
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract required fieldds
            context = str(data["annotations"])
            sub_questions = data["sub_questions"]
            overall_main_question = example["question"] + "\n" + example[
                "question_prompt"]  # Combine question and question_prompt

            # Step 0
            q_type = self._zero_shot(overall_main_question)
            # print(q_type)
            if not self.__validate_zero_shot_output(q_type):
                raise Exception("Invalid zero shot response. Response should be 'OE' or 'MCQ'.")

            # Step 1 (if OE)
            if q_type == "OE":
                sense_check = self._one_shot(overall_main_question, context)
                # print("OE Sense Check:", sense_check)
            else:
                sense_check = 'N/A'
            # Step 2
            answer = self._two_shot(keyf, q_type, overall_main_question, context, sub_questions, sense_check)
            # print("Final Answer:\n", answer)
        else:
            print(f"\tOutput file does not exist: {out_path}")
            answer = ""
        return answer
        # self._save_result(qid, answer)


if __name__ == "__main__":
    print("[Video Answering]")
    VideoAnswerRefinement(call_qwen2_model, 'all_keyframes_annotations.parquet', (LOWER, UPPER)).batch_process(3)