# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from llama import Llama
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def form_prompt(row):
    return f"Question:{row['prompt']}\nOption A: {row['A']}\nOption B: {row['B']}\nOption C: {row['C']}\nOption D: {row['D']}\nOption E: {row['E']}"


def pick_option(output, choice, model):
    embed_1 = model.encode(output, convert_to_tensor=True)
    embed_2 = model.encode(choice, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embed_1, embed_2)
    return cosine_sim

def eval(generator, max_gen_len, temperature, top_p, X_batch, y_batch, model):
    results = generator.text_completion(
        X_batch[:, -1],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    correct = 0
    options = ['A', 'B', 'C', 'D', 'E']
    choices = [X_batch[:, i] for i in range(5)]

    for i, prompt in enumerate(X_batch[:, -1]):
        output = results[i]['generation']
        curr_choice, curr_score = 'A', 0

        for option, choice in zip(options, choices):
            score = pick_option(output, f'Option {option}:{choice[i]}', model).item()
            if score > curr_score:
                curr_choice = option
                curr_score = score

        if curr_choice == y_batch[i]:
            correct += 1

    total = len(X_batch)

    return total, correct


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):

    data_train = pd.read_csv("data/ScienceQA/train.csv")
    data_test = pd.read_csv("data/ScienceQA/test.csv")
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    data_train['instruction'] = data_train.apply(form_prompt, axis=1)
    data_train = data_train.drop(columns=['id', 'prompt'])


    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    def chunks(X, y, chunk_size):
        for i in range(0, len(X), chunk_size):
            yield X[i:i+chunk_size], y[i:i+chunk_size]

    X_data, y_data = data_train[['A', 'B', 'C', 'D', 'E', 'instruction']].to_numpy(), data_train['answer'].to_numpy()

    total, correct = 0, 0
    for X_chunk, y_chunk in tqdm(chunks(X_data, y_data, 4)):
        sub_total, sub_correct = eval(generator, max_gen_len, temperature, top_p, X_chunk, y_chunk, model)
        total += sub_total
        correct += sub_correct

    print(correct / total)

if __name__ == "__main__":
    fire.Fire(main)
