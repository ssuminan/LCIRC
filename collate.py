import torch
import random
import torch.distributed
import torch.utils.checkpoint


def custom_collate_fn(batch, tokenizer, answer_set, multiple_choice_qa_templates):
    prompt_template = random.choice(multiple_choice_qa_templates)
    prompt_templates = []
    answer = []
    for sample in batch:
        if "{OPTION_A}" in prompt_template:
            answer_set_copy = answer_set.copy()
            answer_set_copy.discard(sample['answer'])
            options = random.sample(answer_set_copy, 3)
            options.append(sample['answer'])
            random.shuffle(options)
            prompt_templates.append(
                prompt_template.format(context="{context}", question=sample['question'], OPTION_A=options[0],
                                       OPTION_B=options[1], OPTION_C=options[2], OPTION_D=options[3]))
            answer_idx = options.index(sample['answer'])
            if answer_idx == 0:
                answer.append([29909, 29889])  # token ids of "A."
            elif answer_idx == 1:
                answer.append([29933, 29889])  # token ids of "B."
            elif answer_idx == 2:
                answer.append([29907, 29889])  # token ids of "C."
            else:
                answer.append([29928, 29889])  # token ids of "D."
        elif "{answer_words_len}" in prompt_template:
            prompt_templates.append(prompt_template.format(context="{context}", question=sample['question'],
                                                           answer_words_len=len(sample['answer'].split())))
            answer.append(tokenizer.encode(sample['answer'])[1:])
        else:
            prompt_templates.append(prompt_template.format(context="{context}", question=sample['question']))
            answer.append(tokenizer.encode(sample['answer'])[1:])
    context = [tokenizer.encode(prompt_templates[idx].split("{context}")[0]) + sample['context'] + tokenizer.encode(
        prompt_templates[idx].split("{context}")[1])[2:] + answer[idx] for idx, sample in enumerate(batch)]
    question = [tokenizer.encode(sample[sample.find("Question:"):])[1:] for sample in prompt_templates]
    answer_len = [len(sample) for sample in answer]
    return {
        'context': torch.tensor(context, dtype=torch.long),
        'question': torch.tensor(question, dtype=torch.long),
        'answer': torch.tensor(answer, dtype=torch.long),
        'answer_len': answer_len
    }

def custom_collate_fn_valid(batch, tokenizer, answer_set, multiple_choice_qa_templates):
    random.seed(123456789)
    prompt_template = random.choice(multiple_choice_qa_templates)
    prompt_templates = []
    answer = []
    for sample in batch:
        if "{OPTION_A}" in prompt_template:
            answer_set_copy = answer_set.copy()
            answer_set_copy.discard(sample['answer'])
            options = random.sample(answer_set_copy, 3)
            options.append(sample['answer'])
            random.shuffle(options)
            prompt_templates.append(
                prompt_template.format(context="{context}", question=sample['question'], OPTION_A=options[0],
                                       OPTION_B=options[1], OPTION_C=options[2], OPTION_D=options[3]))
            answer_idx = options.index(sample['answer'])
            if answer_idx == 0:
                answer.append([29909, 29889])  # token ids of "A."
            elif answer_idx == 1:
                answer.append([29933, 29889])  # token ids of "B."
            elif answer_idx == 2:
                answer.append([29907, 29889])  # token ids of "C."
            else:
                answer.append([29928, 29889])  # token ids of "D."
        elif "{answer_words_len}" in prompt_template:
            prompt_templates.append(prompt_template.format(context="{context}", question=sample['question'],
                                                           answer_words_len=len(sample['answer'].split())))
            answer.append(tokenizer.encode(sample['answer'])[1:])
        else:
            prompt_templates.append(prompt_template.format(context="{context}", question=sample['question']))
            answer.append(tokenizer.encode(sample['answer'])[1:])
    context = [tokenizer.encode(prompt_templates[idx].split("{context}")[0]) + sample['context'] + tokenizer.encode(
        prompt_templates[idx].split("{context}")[1])[2:] + answer[idx] for idx, sample in enumerate(batch)]
    question = [tokenizer.encode(sample[sample.find("Question:"):])[1:] for sample in prompt_templates]
    return {
        'context': torch.tensor(context, dtype=torch.long),
        'question': torch.tensor(question, dtype=torch.long),
        'answer': torch.tensor(answer, dtype=torch.long)
    }