from datasets import load_dataset
import json
import csv

from src.utils.call_qwen2_model import call_qwen2_model
from src.config import OUTPUT_DIR, SUBMISSION_DIR

class VideoAnswering:
  """ Question and Answering
  Handles Question & Answering using a 3-stage process:
    0-shot: Determine question type (MCQ or Open-Ended)
    1-shot: Validate Open-Ended questions for contextual fit
    2-shot: Answer based on question type
  
  Model: Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 (GH200)

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
  def __init__(self, call_model):
    self.call_model = call_model
    self.all_pred = {}

  def _zero_shot(self, question: str) -> str:
    """
      0 Shot: figure type of question (open ended or mcq)
        - Is the question Open-Ended or MCQ? If it is Open-Ended, answer 'OE'. If it is MCQ, answer it as 'MCQ'.
    """
    print("\tZero Shot...")
    prompt = f"""Goal: Figure out the type of question\n
      Is the question Open-Ended or MCQ? If it is Open-Ended, answer 'OE'. If it is MCQ, answer it as 'MCQ'.\n
      Question={question}
      Answer="""
      
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
    print("\tOne Shot...")
    prompt = f"""Goal: Determine whether this question makes sense in the context of the video.\n
      Context:\n{context}\n
      Question:\n{question}\n
      Does the question makes sense and does the question ask for exist in the video? If not, what is the most relevant entity that exists in the video instead."""
    response = self.call_model(prompt).strip()
    return response
  
  def _two_shot(self, question_type: str, question: str, context: str, sub_questions: str) -> str:
    """
    Step 2: Answer the question based on its type.
    If MCQ: State your answer and explain in a step-by step manner. Follow the given format strictly when responding:
        ANSWER: {OPTION}
        EXPLAINATION:\n{EXPLANATION}
    If OE: First, answer the sub-questions. Then, use your answer for the sub-questions to answer the main-question.
    """
    print("\tTwo Shot...")
    # MCQ Question
    if question_type == "MCQ":
      prompt = f"""Context:\n{context}\n
Question:\n{question}\n
Instructions: State your answer and explain in a step-by-step manner. Follow the format strictly when responding:\n
ANSWER: {"{OPTION}"}\n
EXPLANATION:\n{"{EXPLANATION}"}"""
      
    # Open-Ended Question
    elif question_type == "OE":
      prompt = f"""Context:\n{context}\n
{sub_questions}\n
Main Question: {question}\n
Instructions: First answer the sub-questions, then use your answers from the sub-questions to help you answer the main question."""
    
    else:
      return "Invalid question type."

    return self.call_model(prompt).strip()
  
  def _save_result(self, qid: str, pred: str):
    self.all_pred[qid] = pred

    print("Exporting submission results to csv...")
    submission_path = SUBMISSION_DIR / f"submission.csv"
    with open(submission_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "pred"])
        writer.writeheader() # header
        for qid, pred in self.all_pred.items():
          writer.writerow({"qid": qid, "pred": pred})
    
    print(f"✓ Done: {submission_path}")

  def process(self, example):
    qid = example["qid"]
    vid = example["video_id"]
    out_path = OUTPUT_DIR / f"{qid}_{vid}.json"
    print(f"\nProcessing {qid}, video {vid}…")

    if out_path.exists():
      # READ from video processor json output
      with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      
      # Extract required fieldds
      context = str(data["annotations"])
      sub_questions = data["sub_questions"]
      overall_main_question = example["question"] + "\n" + example["question_prompt"] # Combine question and question_prompt

      # Step 0
      q_type = self._zero_shot(overall_main_question)
      print(q_type)
      if not self.__validate_zero_shot_output(q_type):
        raise Exception("Invalid zero shot response. Response should be 'OE' or 'MCQ'.")

      # Step 1 (if OE)
      if q_type == "OE":
        sense_check = self._one_shot(overall_main_question, context)
        print("OE Sense Check:", sense_check)

      # Step 2
      answer = self._two_shot(q_type, overall_main_question, context, sub_questions)
      print("Final Answer:\n", answer)
    else:
      print(f"\tOutput file does not exist: {out_path}")
      answer = ""

    self._save_result(qid, answer)

if __name__ == "__main__":
    # Dataset
    print("Loading Dataset...")
    dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
    print(dataset)

    # Call Model Function
    print("Using GH200 Qwen2 Model")

    # Video Answering
    print("[Video Answering]")
    for example in dataset:
      VideoAnswering(call_qwen2_model).process(example)
      break