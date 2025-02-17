import argparse
import torch
import torch.nn.functional as F
import infinitebench, leval, longbench
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import LCIRC, QDLCIRC
from segmenting import make_2k_segments, random_segments, truncate_input_mid, truncate_input_mid_context


def load_model_and_tokenizer(model_id, device, model_name):
    if model_id == "meta-llama/Llama-2-7b-hf" and model_name == "Llama":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        max_length = 4096
    elif model_id == "meta-llama/Llama-2-7b-hf" and (model_name == "LCIRC" or model_name == "QD_LCIRC"):
        K = 64
        NUM_LAYERS = 4
        HID_DIM = 4096
        NUM_HEADS = 32
        HEAD_DIM = 128
        INNER_DIM = 11008
        arr = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        max_length = 4096
        print(args.cp_path)
        if model_name == "LCIRC":
            model = LCIRC(model_id, NUM_LAYERS, HID_DIM, NUM_HEADS, HEAD_DIM, INNER_DIM, K, arr, args.cp_path).to(device)
        elif model_name == "QD_LCIRC":
            model = QDLCIRC(model_id, NUM_LAYERS, HID_DIM, NUM_HEADS, HEAD_DIM, INNER_DIM, K, arr, args.cp_path, 0, True).to(device)

    return model, tokenizer, max_length


def eval_benchmark(args, device):
    # Create dataloaders(dict) for the benchmark
    if args.benchmark == "infinitebench":
        dataloaders = infinitebench.create_dataloader(args.model_id, args.subtasks, args.batch_size,
                                                      args.query_dependent)
    elif args.benchmark == "longbench":
        dataloaders = longbench.create_dataloader(args.model_id, args.subtasks, args.batch_size, args.query_dependent)
    elif args.benchmark == "leval":
        dataloaders = leval.create_dataloader(args.model_id, args.subtasks, args.batch_size, args.query_dependent)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

        # Load the model and tokenizer
    model, tokenizer, max_length = load_model_and_tokenizer(args.model_id, device, args.model_name)

    model.eval()
    for subtask, dataloader in dataloaders.items():
        preds, labels = [], []
        if args.benchmark == "infinitebench":
            max_new_tokens = infinitebench.DATA_NAME_TO_MAX_NEW_TOKENS[subtask]
            max_context_length = max_length - max_new_tokens
        elif args.benchmark == "longbench":
            max_new_tokens = longbench.DATA_NAME_TO_MAX_NEW_TOKENS[subtask]
            max_context_length = max_length - max_new_tokens
        elif args.benchmark == "leval":
            max_new_tokens = leval.DATA_NAME_TO_MAX_NEW_TOKENS[subtask]
            max_context_length = max_length - max_new_tokens
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
                try:
                    print(f"Processing batch {batch_idx} out of {len(dataloader)} for subtask {subtask}")

                    if args.model_name == "Llama":
                        input_ids = truncate_input_mid(batch["prompt_input_ids"], max_context_length).to(device)
                        label = batch["labels"]
                        past_kv = None

                        for i in range(max_new_tokens):
                            outputs = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)

                            next_token_probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
                            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)

                            input_ids = next_token
                            past_kv = outputs.past_key_values

                            if i == 0:
                                generated = next_token.clone()
                            else:
                                generated = torch.cat((generated, next_token), dim=-1)

                            if next_token.item() == tokenizer.eos_token_id:
                                break

                    elif args.model_name == "LCIRC" or args.model_name == "QD_LCIRC":
                        if args.model_name == "LCIRC":
                            input_ids = truncate_input_mid(batch["prompt_input_ids"], max_context_length)
                            context_ids = truncate_input_mid_context(batch["prompt_input_ids"], max_context_length).to(device)
                            label = batch["labels"]

                            total_soft_prompt = None
                            if batch["prompt_input_ids"].size(1) - max_context_length > 0:
                                if context_ids.size(1) < 2048:  ## case of Ours 2k segmentation
                                    segs = [0, context_ids.size(1)]
                                elif args.rand_seg:
                                    segs = random_segments(context_ids.size(1), 64)
                                else:
                                    segs = make_2k_segments(context_ids.size(1))

                                soft_prompt = model.latent.repeat(1, 1, 1)
                                for i in range(len(segs) - 1):
                                    context_embedding = model.model.model.model.embed_tokens(
                                        context_ids[:, segs[i]:segs[i + 1]].to(device))
                                    soft_prompt = model.compressor(soft_prompt, context_embedding)
                                    if i == 0:
                                        total_soft_prompt = soft_prompt.clone().detach()
                                    if i > 0:
                                        total_soft_prompt = torch.cat((total_soft_prompt, soft_prompt), dim=1)

                        elif args.model_name == "QD_LCIRC":
                            full_data = torch.concat((batch["context_input_ids"], batch["question_input_ids"][:, 1:]), dim=1)
                            input_ids = truncate_input_mid(full_data, max_context_length)
                            context_ids = truncate_input_mid_context(full_data, max_context_length).to(device)
                            label = batch["labels"]

                            total_soft_prompt = None
                            if full_data.size(1) - max_context_length > 0:
                                if context_ids.size(1) < 2048:  ## case of Ours 2k segmentation
                                    segs = [0, context_ids.size(1)]
                                elif args.rand_seg:
                                    segs = random_segments(context_ids.size(1), 64)
                                else:
                                    segs = make_2k_segments(context_ids.size(1))

                                soft_prompt = model.latent.repeat(1, 1, 1)
                                query_embedding = model.model.model.model.embed_tokens(batch["question_input_ids"][:, 1:].to(device))
                                for i in range(len(segs) - 1):
                                    soft_prompt = model.mixer(soft_prompt, query_embedding)
                                    context_embedding = model.model.model.model.embed_tokens(
                                        context_ids[:, segs[i]:segs[i + 1]].to(device))
                                    soft_prompt = model.compressor(soft_prompt, context_embedding)
                                    if i == 0:
                                        total_soft_prompt = soft_prompt.clone().detach()
                                    if i > 0:
                                        total_soft_prompt = torch.cat((total_soft_prompt, soft_prompt), dim=1)

                        input = input_ids.to(device)
                        past_kv = None

                        for i in range(max_new_tokens):
                            outputs = model.model(input_ids=input,
                                                  soft_prompt=total_soft_prompt,
                                                  past_key_values=past_kv, use_cache=True)

                            next_token_probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
                            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)

                            input = next_token
                            past_kv = outputs.past_key_values

                            if i == 0:
                                generated = next_token.clone()
                            else:
                                generated = torch.cat((generated, next_token), dim=-1)

                            if next_token.item() == tokenizer.eos_token_id:
                                break

                    predicted_labels = [tokenizer.decode(generated[i], skip_special_tokens=True) for i in
                                        range(generated.shape[0])]
                    ''' For LongBench, L-Eval '''
                    if args.benchmark == "longbench" or args.benchmark == "leval":
                        for i in range(len(predicted_labels)):
                            predicted_labels[i] = predicted_labels[i].lstrip('\n').split('\n')[0]

                    print(f"Generated: {predicted_labels} | subtask: {subtask} | label: {label}")

                    preds.extend(predicted_labels)
                    labels.extend(label)
                except Exception as e:
                    print(f"Error: {e}")
                    continue

        # Calculate the metrics
        if args.benchmark == "infinitebench":
            score = infinitebench.evaluation(preds, labels, subtask, args.model_id)
            print(f"Score: {score} | subtask: {subtask}")
        elif args.benchmark == "longbench":
            score = longbench.evaluation(preds, labels, subtask)
            print(f"Score: {score} | subtask: {subtask}")
        elif args.benchmark == "leval":
            score = leval.evaluation(preds, labels, subtask)
            print(f"Score: {score} | subtask: {subtask}")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--subtasks", type=str, required=True, nargs="+")
    parser.add_argument("--query_dependent", action="store_true", help="Use query dependent generation")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument('--rand_seg', action='store_true', help="Use randomized segmentation")
    parser.add_argument("--cp_path", type=str, default=None)

    args = parser.parse_args()

    GPU_NUM = 0
    device = torch.device(f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu")

    eval_benchmark(args, device)