import json
from transformers import set_seed, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import List
from collections import Counter
import torch
import re
import string

set_seed(42)

qa_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]

DATA_NAME_TO_MAX_NEW_TOKENS = {
    "narrativeqa": 128, 
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

    
def evaluation(preds, labels, subtask):
    total_score = 0.
    for (pred, label) in zip(preds, labels):
        score = 0.
        if subtask in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        
        if pred == "":
            continue
        
        for l in label:
            if subtask in qa_datasets:
                score = max(score, qa_f1_score(pred, l))
            else:
                print(f"Subtask {subtask} not supported")
        total_score += score

    return round(100 * total_score / len(preds), 2)


def truncate_input(input_ids: torch.Tensor, max_length: int, manner="middle"):
    input_ids = input_ids.squeeze(0)  # Remove the batch dimension if it exists

    if input_ids.shape[0] <= max_length:
        return input_ids.unsqueeze(0)  # Add the batch dimension back
    if manner == "middle":
        split = max_length // 2
        truncated_input = torch.cat((input_ids[:split], input_ids[-split:]))
        return truncated_input.unsqueeze(0)  # Add the batch dimension back
    else:
        return None


def create_context_and_question(x, prompt_format):
    prompt = prompt_format.format(**x)

    if 'Question' in prompt:
        index = prompt.rfind('Question')
        context, question = prompt[:index], prompt[index:]
        return context, question
    else:
        return prompt, None
    

def custom_collate_fn(batch, tokenizer, query_dependent):
    labels = [
        item['answers']
        for item in batch
    ]

    if query_dependent:
        tokenized_context = [
            tokenizer(
                item['context'],
                padding=False,
                truncation=False,
                return_tensors='pt'
            ) for item in batch
        ]

        context_input_ids = torch.cat(
            [
                tokenized_input['input_ids'] 
                for tokenized_input in tokenized_context
            ], dim=0
        )

        if batch[0]['question']:
            tokenized_question = [
                tokenizer(
                    item['question'],
                    padding=False,
                    truncation=False,
                    return_tensors='pt'
                ) for item in batch
            ]
            question_input_ids = torch.cat(
                [
                    tokenized_input['input_ids'] 
                    for tokenized_input in tokenized_question
                ], dim=0
            )
        else:
            question_input_ids = None

        return {
            'context_input_ids': context_input_ids,
            'question_input_ids': question_input_ids,
            'labels': labels,
        }
    else:
        tokenized_prompt = [
            tokenizer(
                item['prompt'],
                padding=False,
                truncation=False,
                return_tensors='pt'
            ) for item in batch
        ]

        prompt_input_ids = torch.cat(
            [
                tokenized_input['input_ids'] 
                for tokenized_input in tokenized_prompt
            ], dim=0
        )

        return {
            'prompt_input_ids': prompt_input_ids,
            'labels': labels,
        }


def create_dataloader(model_id: str, subtasks: List[str], batch_size: int, query_dependent: bool) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataloaders = {}
    for subtask in subtasks:
        print(f"Creaing dataloader for {subtask}")

        test_dataset = load_dataset('THUDM/LongBench', subtask, split='test')

        dataset2prompt = json.load(open("longbench_prompt.json", "r"))
        prompt_format = dataset2prompt[subtask]

        if query_dependent:
            test_dataset = test_dataset.map(lambda x: {
                **x,
                'subtask': subtask,
                **dict(zip(['context', 'question'], create_context_and_question(x, prompt_format)))
            })
        else:
            test_dataset = test_dataset.map(lambda x: {**x, 'prompt': prompt_format.format(**x)})                                           

        test_dataloader = DataLoader(test_dataset, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, query_dependent), batch_size=batch_size)
        dataloaders[subtask] = test_dataloader

    return dataloaders
