import random
import torch

def make_2k_segments(num):
    q = num // 2048
    q2 = (num % 2048) // q
    p = (num % 2048) % q
    li = [(t + 1) * (2048 + q2) + p for t in range(q)]
    li.insert(0, 0)
    return li


def random_segments(max_length, l):
    segments = [0]
    while True:
        random_len = random.choices(range(l, 4097 - l), weights=[i for i in range(l, 4097 - l)], k=1)[0]
        next_seg = random_len + segments[-1]
        if next_seg == max_length:
            segments.append(next_seg)
            break
        elif next_seg > max_length:
            segments.append(max_length)
            break
        else:
            segments.append(next_seg)
    return segments


def truncate_input_mid(input_ids, max_length):
    split = max_length // 2
    truncated_input = torch.cat((input_ids[:, :split], input_ids[:, -split:]), dim=1)
    return truncated_input


def truncate_input_mid_context(input_ids, max_length):
    split = max_length // 2
    return input_ids[:, split:-split]