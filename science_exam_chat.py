from typing import Optional
import fire
import pandas as pd
from llama import Llama
from tqdm import tqdm

import os
import json
import datetime

def form_prompt(row):
    return f"Question: {row['prompt']}\nOption A: {row['A']}\nOption B: {row['B']}\nOption C: {row['C']}\nOption D: {row['D']}\nOption E: {row['E']}"

def form_dialog(prompt):
    return [{
        "role": "user", 
        "content":f'You are a helpful assistant on multiple choices questions. {prompt} Please rank three correct options based on your confidence in descending order and do not provide any explanation.'}]


# def form_dialog(prompt):

#     instructions = """
# You are a group of experts in STEM subjects.
# You need to provide the best possible answer to the user's question.

# First, choose the STEM expert who will answer the question. You can choose beetween: Science, Technology, Engineering, and Mathematics expert.
# Then, let the expert thinks step by steps to answer the question with in-depth explanation.
# Then, summarize the answer in less then 100 words.
# Finally, let the expert select first char (A,B,C,D,E) of the three best options for the question ranked from best to worst.

# Don't write anything but provide result in a single JSON with the following format 
# {
# 	expert="...",
# 	expert_answer_summarized ="...",
# 	best_answers = [ ".",".", "." ],
#  	best_answers_explanations = ["...","...","..."]

# }
# """
#     return [{
#         "role": "user", 
#         "content": f"{instructions} {prompt} "}]


def chunks(X, chunk_size):
    for i in range(0, len(X), chunk_size):
        yield X[i:i+chunk_size]

def pred(generator, max_gen_len, temperature, top_p, dialogs):
    return generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

def average_precision_at_k(recommended, relevant, k=3):
    if len(recommended) > k:
        recommended = recommended[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(recommended):
        if p in relevant and p not in recommended[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not relevant:
        return 0.0
    return score / min(len(relevant), k)

def mean_average_precision_at_k(recommended_lists, relevant_lists, k=3):
    assert len(recommended_lists) == len(relevant_lists)
    return sum(average_precision_at_k(r, rel, k) for r, rel in zip(recommended_lists, relevant_lists)) / len(recommended_lists)

def find_examples(recommended_lists, relevant_lists, k=3, good=True):
    assert len(recommended_lists) == len(relevant_lists)
    examples = []
    for i, (r, rel) in enumerate(zip(recommended_lists, relevant_lists)):
        if good:
            if average_precision_at_k(r, rel, k) == 1:
                examples += [i]
        else:
            if average_precision_at_k(r, rel, k) == 0:
                examples += [i]
    try:
        return examples#[:5]
    except:
        return examples

# def save_examples(good_idx, bad_idx, pred_history, label):
#     good_examples = []
#     bad_examples = []
#     for i in good_idx:
#         cur_example = pred_history[i]
#         cur_example['answer'] = label[i]
#         good_examples += [cur_example]
#     for i in bad_idx:
#         cur_example = pred_history[i]
#         cur_example['answer'] = label[i]
#         bad_examples += [cur_example]
    
#     data = {
#         'good_examples': good_examples,
#         'bad_examples': bad_examples
#     }

#     timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#     os.makedirs("examples", exist_ok=True)
#     filename = f"examples/{timestamp}_examples.json"

#     with open(filename, 'w') as f:
#         json.dump(data, f)
#     return filename

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    data_train = pd.read_csv("data/ScienceQA/train.csv")
    data_train['instruction'] = data_train.apply(form_prompt, axis=1)
    data_train = data_train.drop(columns=['id', 'prompt', 'A', 'B', 'C', 'D', 'E'])

    X_data, y_data = data_train['instruction'].to_numpy(), data_train['answer'].to_numpy()
    pred_ans = []
    # pred_history = []
    for prompt in tqdm(chunks(X_data, max_batch_size)):      
        dialog = [form_dialog(p) for p in prompt]
        results = pred(generator, max_gen_len, temperature, top_p, dialog)
        # pred_history.append({"input": dialog[0][0]['content'], "output": results[0]['generation']['content']})
        # print(f"----------------------------------\nDialog:\n{dialog}\n----------------------------------\nResults:\n{results}\n----------------------------------\n")
        for res in results:
            content = res['generation']['content'].split("Option")[1:4]
            curr_ans = []
            for opt in content:
                curr_ans += [opt[1]]
            pred_ans += [curr_ans]
    label = y_data.tolist()
    print(f"MAP@3: {mean_average_precision_at_k(pred_ans, label, k=3)}\n------------------------\n")
    # good_idx = find_examples(pred_ans, label, k=3, good=True)
    # bad_idx = find_examples(pred_ans, label, k=3, good=False)

    # filename = save_examples(good_idx, bad_idx, pred_history, label)
    # print(f"Data saved to {filename}")




if __name__ == "__main__":
    fire.Fire(main)
