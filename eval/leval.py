import torch
from transformers import set_seed, AutoTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from typing import List
import json
import re
import string
from collections import Counter, defaultdict
import nltk
from rouge_score import rouge_scorer
from multiprocessing import Pool

set_seed(42)

DATA_NAME_TO_MAX_NEW_TOKENS = {
    "coursera": 10,
    "quality": 80,
    "tpo": 10,
    "sci_fi": 20,
    "financial_qa": 300,
    "narrative_qa": 50,
    "scientific_qa": 300,
    "natural_question": 110
}

qa_datasets = ["coursera", "quality", "tpo", "sci_fi" ,"financial_qa", "narrative_qa", "scientific_qa", "natural_question"]

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

## exact match score ##
def exact_match_score(prediction, ground_truth):
    flag = False  # whether with options
    Choice = ['A', 'B', 'C', 'D']
    for char in normalize_answer(ground_truth):
        if char not in Choice:
            flag = True
            break
    res = 0
    if not flag:
        if normalize_answer(prediction) == normalize_answer(ground_truth):
            res = 1
        elif set(normalize_answer(prediction)).issubset(set(normalize_answer(ground_truth))):
            res = 0.25  # has many correct options
    else:
        try:
            pre = float(prediction)
            gt = float(ground_truth)
            res = int(pre == gt)
        except ValueError:
            if ground_truth.lower().replace(" ", "") in prediction.lower().replace(" ", ""):
                res = 1

    print(prediction, ground_truth, f"| score={res}")
    print("=" * 20)
    return res


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_exact_match(predictions, references):
    exact_match = 0
    correct = 0
    half_correct = 0

    for prediction, ground_truths in zip(predictions, references):
        res = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        exact_match += res
        if res == 1:
            correct += 1
        if res == 0.25:
            half_correct += 1
    print(
        f"There are {correct} correct answers \n [for coursera:] {half_correct} can not select all correct options\n Total: {len(predictions)} questions.")
    return 100.0 * exact_match / len(predictions)


## f1 score ##
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_f1(predictions, references):
    f1 = 0
    for prediction, ground_truths in zip(predictions, references):
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return 100.0 * f1 / len(predictions)


## rouge score ##
def compute_rouge(predictions, references, rouge_types=None, use_stemmer=False):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
    with Pool() as p:
        scores = p.starmap(scorer.score, zip(references, predictions))

    result = {}
    for key in scores[0]:
        result[key] = list(score[key] for score in scores)

    return result


def postprocess_text(text):
    # rougeLSum expects newline after each sentence
    return "\n".join(nltk.sent_tokenize(text))


def process_output_mc(response, subtask):
    # Use regular expression to replace anything that is not A, B, C or D with an empty string
    if len(response.strip()) == 0:
        return "None"
    if response in "ABCD":
        return response
    if "coursera" not in subtask:
        for i, chr in enumerate(response):
            if chr in "ABCD":
                return chr
    # retrieve multiple correct answers (for coursera)

    cleaned_str = ""
    for chr in response:
        if chr in "ABCD":
            cleaned_str += chr
            response = response[1:]
        else:
            break

    if len(cleaned_str) > 1:
        return ''.join(sorted(set(cleaned_str)))
    # retrieve multiple correct answers (for coursera)
    response = response.split("Question")[0]
    pattern = r"\s*[A-Z](?=[\s.)])"
    options = re.findall(pattern, response)
    cleaned_str += ' '.join(options).strip()
    cleaned_str = re.sub(r'[^A-D]', '', cleaned_str)
    s_set = set(cleaned_str)
    cleaned_str = "".join(sorted(s_set))
    if len(cleaned_str) < 1:  
        has_answer = False
        for chr in response:
            if chr in "ABCD":
                cleaned_str += chr
                response = response[1:]
                has_answer = True
            elif has_answer:
                break
    if len(cleaned_str) < 1:  
        cleaned_str = "A"
    return cleaned_str


def process_gt_mc(response):
    # Use regular expression to replace anything that is not A, B, C or D with an empty string
    input_str = response.split()[0]
    cleaned_str = re.sub(r'[^A-D]', '', input_str)
    # random guess
    if len(cleaned_str) < 1:
        cleaned_str = "A"
    return cleaned_str

def process_gt_judge(response):
    response = response.lower()
    if "[fact:" in response:
        resp_list = response.split("[fact:")
        loyalty = resp_list[0]
        fact = resp_list[1][:-1]
    else:
        loyalty = response
        fact = ''
    return [loyalty.strip(), fact.strip()]


def process_output_judge(response):
    loyalty, fact = process_gt_judge(response)
    output_list = []
    for word in loyalty.split():
        if "true" in word:
            output_list.append("true")
            break
        elif "false" in word:
            output_list.append("false")
            break
    if len(output_list) == 0:
        output_list.append("<error>")
    for word in fact.split():
        if "true" in word:
            output_list.append("true")
            break
        elif "false" in word:
            output_list.append("false")
            break

    if len(output_list) == 1:
        output_list.append("<error>")

    return output_list # disable random guess for the dataset for high variance

def evaluation(preds, labels, subtask):
    predictions = []
    references = []
    if  subtask == "topic_retrieval_longchat":
        references = [[], [], []]
        predictions = [[], [], []]
    elif subtask == "sci_fi":
        references = [[], []]
        predictions = [[], []]


    for pred, label in zip(preds, labels):
        if subtask in ["coursera", "quality", "tpo"]:
            references.append([process_gt_mc(label)])
            predictions.append(process_output_mc(pred, subtask))
        elif subtask in ["sci_fi"]:
            loyalty, fact = process_gt_judge(label)
            references[0].append([loyalty])
            references[1].append([fact])
            loyalty_pred, fact_pred = process_output_judge(pred)
            predictions[0].append(loyalty_pred)
            predictions[1].append(fact_pred)
        else:
            references.append([label])
            predictions.append(pred)


    if subtask == "sci_fi":
        output_str = ""
        balance_score = 0
        for i in range(len(predictions)):
            pred = predictions[i]
            ref = references[i]
            em_score = compute_exact_match(predictions=pred, references=ref)
            if i ==0:
                output_str += f"loyalty score: {em_score}\n"
            else:
                output_str += f"fact score: {em_score}"
            balance_score += em_score
        output_str += f"balance score: {balance_score/2}"

        return output_str
    elif subtask in ["coursera", "quality", "tpo"]:
        em_score = compute_exact_match(predictions=predictions, references=references)
        return em_score
    elif subtask in ["financial_qa", "narrative_qa", "scientific_qa", "natural_question"]:
        f1_score = compute_f1(predictions=predictions, references=references)
        return f1_score
    else:
        print(f"Unsupported subtask: {subtask}")



## data processing ##
def split_instructions_and_outputs(dataset):
    flattened_samples = []

    for sample in dataset:
        # Iterate over each instruction-output pair and create a new sample
        for instruction, output in zip(sample['instructions'], sample['outputs']):
            new_sample = {
                "instruction": instruction,
                "output": output,
                "input": sample['input'],  
                "source": sample['source'],  
                "evaluation": sample['evaluation'] 
            }
            flattened_samples.append(new_sample)

    new_dataset = Dataset.from_list(flattened_samples)
    
    return new_dataset


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


def extract_context_and_question(prompt):
    keywords = ['Question', 'Instruction']
    for keyword in keywords:
        if keyword in prompt:
            index = prompt.rfind(keyword)
            context, question = prompt[:index], prompt[index:]
            return context, question
    return prompt, None


def create_prompt(x, prompt_format):
    if x['evaluation'] == 'exam':
        prompt = prompt_format.format(x['input'], x['instruction'])
    else:
        prompt = prompt_format.format(x['input'], x['instruction'], len(x['output'].split()))
    return prompt

def create_context_and_question(x, prompt_format):
    document, instruction, output_len = x['input'], x['instruction'], len(x['output'].split())
    if x['evaluation'] == 'exam':
        prompt = prompt_format.format(document, instruction)
    else:
        prompt = prompt_format.format(document, instruction, output_len)

    context, question = extract_context_and_question(prompt)

    return context, question
    

def custom_collate_fn(batch, tokenizer, query_dependent):
    labels = [
        item['output']
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

        test_dataset = load_dataset('L4NLP/LEval', subtask, split='test')
        test_dataset = split_instructions_and_outputs(test_dataset)
        dataset2prompt = json.load(open("leval_prompt.json", "r"))
        prompt_format = dataset2prompt[subtask]

        if query_dependent:
            test_dataset = test_dataset.map(lambda x: {
                **x,
                **dict(zip(['context', 'question'], create_context_and_question(x, prompt_format)))
            })
        else:
            test_dataset = test_dataset.map(lambda x: {
                **x, 
                'prompt': create_prompt(x, prompt_format)
            })

        test_dataloader = DataLoader(test_dataset, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, query_dependent), batch_size=batch_size)
        dataloaders[subtask] = test_dataloader

    return dataloaders