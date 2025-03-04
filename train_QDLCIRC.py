import torch
from torch import nn
from transformers import set_seed, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import torch.utils.checkpoint
from cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts
import json
from models import QDLCIRC
from segmenting import make_2k_segments, random_segments, truncate_input_mid, truncate_input_mid_context
from collate import custom_collate_fn, custom_collate_fn_valid


## initialize process group
torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(seconds=5400))
rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)
device = torch.cuda.current_device()
world_size = torch.distributed.get_world_size()
print(f"rank: {rank}, device: {device}, world_size: {world_size}")

## Hyperparameter Settings
EPOCHS = 10
LR = 0.00002
K = 64
NUM_LAYERS = 4
HID_DIM = 4096
NUM_HEADS = 32
HEAD_DIM = 128
INNER_DIM = 11008
NUM_GPUS = 8
BATCH_SIZE = 1*NUM_GPUS
BATCH_SIZE_VALID = 1
PRE_CP_PATH = "LCIRC_checkpoint_epoch0_final.pt"
arr = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31]
WARMUP_STEP = 300
NUM_SELECT = 8

if rank == 0:
    print(f"EPOCHS: {EPOCHS}")
    print(f"LEARNING_RATE: {LR}")
    print(f"K: {K}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"arr: {arr}")
    print(f"PRE_CP_PATH: {PRE_CP_PATH}")
    print(f"WARMUP_STEP: {WARMUP_STEP}")
    print(f"NUM_SELECT: {NUM_SELECT}")

## training prompt templates
multiple_choice_qa_templates = [
    "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is ",
    "Please answer the question based on the long texts below. \n{context}\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\nAnswer: ",
    "Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there could be a sinlge correct option or multiple correct options. Please only provide the letter corresponding to the answer (like A or AB) when answering.\nDocument is as follows. {context} \nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\nPlease directly give answer without any additonal output or explanation.\nAnswer: ",
    "Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there is only a sinlge correct option. Please only provide the letter corresponding to the answer (like A or B) when answering. For other questions, please directly give the concise and accurate answer.\nDocument is as follows. {context} \nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\nPlease directly give answer without any additonal output or explanation.\nAnswer: ",
    "You are given an article or a long dialogue, a multiple-choice question with four optional answer(marked by A, B, C, D). Please choose the best answer to the question based on the given content by responsing its corresponding letter (either A, B, C, or D). Do not provide any explanation.\n\nArticle: {context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nAnswer(Only give the corresponding letter (either A, B, C, or D)): ",
    "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer: ",
    "Please answer the question based on the long texts below. \n{context}\nQuestion: {question}\nAnswer: ",
    "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {question}\n\nAnswer: ",
    "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\nAnswer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {question}\n\nAnswer: ",
    "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer: ",
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer: ",
    "Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there is only a sinlge correct option. Please only provide the letter corresponding to the answer (like A or B) when answering. For other questions, please directly give the concise and accurate answer.\nDocument is as follows. {context} \nQuestion: {question}\nPlease directly give answer without any additonal output or explanation.\nAnswer: ",
    "Now you are given a very long document. Please follow the instruction after this document. These instructions may include summarizing a document, answering questions based on the document, or writing a required paragraph.\nDocument is as follows.\n{context}\nQuestion: {question}\n\nAnswer this question with {answer_words_len} words.\nAnswer: ",
    "I will provide you multiple news article below. Answer the question based on a specfic article.\n{context}\nQuestion: {question}\nAnswer: ",
    "Write a high-quality answer for the given question using only the provided plots (some of which might be irrelevant).\n{context}\nQuestion: {question}\nAnswer: ",
    "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n{context}\nQuestion: {question}\nAnswer: ",
    "Answer the question based on the given paragraphs. Note that some paragraphs might be irrelevant.\n{context}\nQuestion: {question}\nAnswer: ",
    "Answer the question based on the given Context.\n{context}\nQuestion: {question}\nAnswer: ",
    "You are a helpful AI bot that answers questions for a user. Keep your response short and direct.\nRead the document and answer the question.\n{context}\nQuestion: {question}\nAnswer: ",
]

## Load dataset
train_dataset = load_dataset("ssumin/fineweb-lqa", split="train")
valid_dataset = load_dataset("ssumin/fineweb-lqa", split="validation")

with open("./data/FineWeb_LQA_train_answer_sample.json", 'r', encoding='utf-8') as file:
    train_answer_set = set(json.load(file))
with open("./data/FineWeb_LQA_validation_answer_sample.json", 'r', encoding='utf-8') as file:
    valid_answer_set = set(json.load(file))

if rank == 0:
    print(f"Length of train data: {len(train_dataset)}")
    print(f"Length of valid data: {len(valid_dataset)}")
    print(f"Length of train answer set: {len(train_answer_set)}")
    print(f"Length of valid answer set: {len(valid_answer_set)}")
    print("dataset completed!")

## Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

## Load dataloader
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
train_dataloader = DataLoader(train_dataset,
                              batch_size=int(BATCH_SIZE/world_size),
                              shuffle=False,
                              num_workers=0,
                              sampler=train_sampler,
                              collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, train_answer_set, multiple_choice_qa_templates))
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_VALID, shuffle=False, collate_fn=lambda batch: custom_collate_fn_valid(batch, tokenizer, valid_answer_set, multiple_choice_qa_templates))
if rank == 0:
    print(f"Length of train dataloader: {len(train_dataloader)}")
    print(f"Length of valid dataloader: {len(valid_dataloader)}")
    print("dataloader completed!")

## Load model
set_seed(42+rank)
model_name = "meta-llama/Llama-2-7b-hf"
llama = QDLCIRC(model_name, NUM_LAYERS, HID_DIM, NUM_HEADS, HEAD_DIM, INNER_DIM, K, arr, PRE_CP_PATH, rank).cuda(rank)
llama = DDP(module=llama, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=True)

optimizer = torch.optim.Adam(llama.parameters(), lr=0)
scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=230352*4, T_mult=1, eta_max=LR, T_up=WARMUP_STEP, gamma=1., last_epoch=-1)
criterion = nn.CrossEntropyLoss(reduction='mean')

if rank == 0:
    print(f"Model size: {sum([p.numel() * 2 for p in llama.module.model.parameters()]) / 1024 / 1024 / 1024}")
    print(f"Compressor size: {sum([p.numel() * 2 for p in llama.module.compressor.parameters()]) / 1024 / 1024 / 1024}")
    print(f"Latent size: {llama.module.latent.numel() * 2 / 1024 / 1024 / 1024}")
    print(f"Mixer size: {sum([p.numel() * 2 for p in llama.module.mixer.parameters()]) / 1024 / 1024 / 1024}")
    print(f"Total size: {sum([p.numel() * 2 for p in llama.module.parameters()]) / 1024 / 1024 / 1024}")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024/1024/1024}")

if rank == 0:
    print("model completed!")

torch.cuda.empty_cache()
ISZERO = rank == 0
train_loss_all = []
valid_loss_all = []
REAL_STEP = 0
UPDATE_STEP = 8
optimizer.zero_grad()
for epoch in tqdm(range(EPOCHS), disable=not ISZERO):
    train_sampler.set_epoch(epoch)
    for idx, data in enumerate(tqdm(train_dataloader, disable=not ISZERO)):
        # Training
        if data['answer_len'][0] > 500:
            continue

        input_ids = truncate_input_mid(data['context'], 4096)
        context_ids = truncate_input_mid_context(data['context'], 4096)

        # random segmenting
        seg = random_segments(context_ids.size(1), K)

        llama.train()

        data_gt = data['answer'].to(device)
        query_embedding = llama.module.model.model.model.embed_tokens(data['question'].to(device))
        if len(seg) - 1 <= NUM_SELECT:
            selected_segs = list(range(len(seg) - 1))
        else:
            selected_segs = np.random.randint(0, (len(seg) - 1), size=NUM_SELECT)
        total_soft_prompt = None
        soft_prompt = llama.module.latent.repeat((BATCH_SIZE // NUM_GPUS), 1, 1)
        for i in range(len(seg) - 1):
            if i not in selected_segs:
                with torch.no_grad():
                    soft_prompt = llama.module.mixer(soft_prompt, query_embedding)
                    seg_embedding = llama.module.model.model.model.embed_tokens(context_ids[:, seg[i]:seg[i + 1]].to(device))
                    soft_prompt = llama.module.compressor(soft_prompt, seg_embedding)
                    if i == 0:
                        total_soft_prompt = soft_prompt.clone()
                    else:
                        total_soft_prompt = torch.cat((total_soft_prompt, soft_prompt), dim=1)
            else:
                soft_prompt = llama.module.mixer(soft_prompt, query_embedding)
                seg_embedding = llama.module.model.model.model.embed_tokens(context_ids[:, seg[i]:seg[i + 1]].to(device))
                soft_prompt = llama.module.compressor(soft_prompt, seg_embedding)
                if i == 0:
                    total_soft_prompt = soft_prompt.clone()
                else:
                    total_soft_prompt = torch.cat((total_soft_prompt, soft_prompt), dim=1)
        total_soft_prompt = total_soft_prompt[:, -64 * 50:]

        prob = llama(input_ids.to(device), data_gt.size(1)+1, total_soft_prompt)

        loss = criterion(prob.transpose(1, 2), data_gt)

        loss = loss / UPDATE_STEP
        loss.backward()
        REAL_STEP = REAL_STEP + 1

        if REAL_STEP == 8:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            REAL_STEP = 0
            torch.cuda.empty_cache()

        if rank == 0:
            train_loss_all.append((loss.item()*UPDATE_STEP))

        # Validation
        if (idx == 0 or (idx + 1) % 2000 == 0) and rank == 0:
            torch.cuda.empty_cache()
            losses_eval = []
            for idx_valid, valid_data in enumerate(valid_dataloader):
                data_length = valid_data['context'].size(1)

                llama.eval()

                input_ids = truncate_input_mid(valid_data['context'], 4096)
                context_ids = truncate_input_mid_context(valid_data['context'], 4096)

                if data_length - 4096 >= 2048:
                    segs = make_2k_segments(data_length - 4096)
                else:
                    segs = [0, data_length - 4096]

                total_soft_prompt = None
                soft_prompt = llama.module.latent.repeat(BATCH_SIZE_VALID, 1, 1)
                query_embedding = llama.module.model.model.model.embed_tokens(valid_data['question'].to(device))
                with torch.no_grad():
                    for i in range(len(segs) - 1):
                        soft_prompt = llama.module.mixer(soft_prompt, query_embedding)
                        context_embedding = llama.module.model.model.model.embed_tokens(context_ids[:, segs[i]:segs[i + 1]].to(device))
                        soft_prompt = llama.module.compressor(soft_prompt, context_embedding)
                        if i == 0:
                            total_soft_prompt = soft_prompt.clone().detach()
                        else:
                            total_soft_prompt = torch.cat((total_soft_prompt, soft_prompt), dim=1)

                    output = llama(input_ids.to(device), valid_data['answer'].size(1)+1, total_soft_prompt)

                    loss = criterion(output.transpose(1, 2),
                                     valid_data['answer'].to(device))

                    losses_eval.append(loss.item())

            valid_loss = torch.tensor(losses_eval).mean()

            if (idx + 1) % 2000 == 0:
                if len(valid_loss_all) == 0:
                    CP_PATH = f'QDLCIRC_checkpoint_epoch{epoch}_{(idx + 1)}.pt'
                    torch.save({'Xattn_layers_state_dict': llama.module.model.Xattn_layers.state_dict(),
                                'compressor_state_dict': llama.module.compressor.state_dict(),
                                'mixer_state_dict': llama.module.mixer.state_dict(),
                                'latent_state_dict': {k: v for k, v in llama.module.state_dict().items() if k == "latent"},
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch}, CP_PATH)
                else:
                    if valid_loss.item() < valid_loss_all[-1]:
                        CP_PATH = f'QDLCIRC_checkpoint_epoch{epoch}_{(idx + 1)}.pt'
                        torch.save({'Xattn_layers_state_dict': llama.module.model.Xattn_layers.state_dict(),
                                    'compressor_state_dict': llama.module.compressor.state_dict(),
                                    'mixer_state_dict': llama.module.mixer.state_dict(),
                                    'latent_state_dict': {k: v for k, v in llama.module.state_dict().items() if k == "latent"},
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'epoch': epoch}, CP_PATH)

                valid_loss_all.append(valid_loss.item())
                np.save(f"QDLCIRC_train_loss", np.array(train_loss_all))
                np.save(f"QDLCIRC_valid_loss", np.array(valid_loss_all))

            torch.cuda.empty_cache()

    if rank == 0:
        CP_PATH = f'QDLCIRC_checkpoint_epoch{epoch}_final.pt'
        torch.save({'Xattn_layers_state_dict': llama.module.model.Xattn_layers.state_dict(),
                    'compressor_state_dict': llama.module.compressor.state_dict(),
                    'mixer_state_dict': llama.module.mixer.state_dict(),
                    'latent_state_dict': {k: v for k, v in llama.module.state_dict().items() if k == "latent"},
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch}, CP_PATH)
        np.save(f"QDLCIRC_train_loss", np.array(train_loss_all))
        np.save(f"QDLCIRC_valid_loss", np.array(valid_loss_all))
