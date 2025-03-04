import torch
from torch import nn
from transformers import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import torch.utils.checkpoint
from models import LCIRC
from segmenting import make_2k_segments, random_segments

def custom_collate_fn(batch):
    input_ids = [sample['input_ids'] for sample in batch]
    return torch.tensor(input_ids, dtype=torch.long)

## initialize process group
torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(seconds=5400))
rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)
device = torch.cuda.current_device()
world_size = torch.distributed.get_world_size()
print(f"rank: {rank}, device: {device}, world_size: {world_size}")

## Hyperparameter Settings
EPOCHS = 10
LR = 0.00005
NUM_PROMPTS_VALID = 500
K = 64
NUM_LAYERS = 4
HID_DIM = 4096
NUM_HEADS = 32
HEAD_DIM = 128
INNER_DIM = 11008
NUM_GPUS = 8
BATCH_SIZE = 1*NUM_GPUS
arr = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31]

if rank == 0:
    print(f"EPOCHS: {EPOCHS}")
    print(f"LEARNING_RATE: {LR}")
    print(f"NUM_PROMPTS_VALID: {NUM_PROMPTS_VALID}")
    print(f"K: {K}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"arr: {arr}")


## Load dataset
train_dataset = load_dataset("ssumin/fineweb-edu-long", split="train")
if rank == 0:
    print(len(train_dataset))

valid_dataset = load_dataset("ssumin/fineweb-edu-long", split="validation")
if rank == 0:
    print(len(valid_dataset))

if rank == 0:
    print("dataset completed!")

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
train_dataloader = DataLoader(train_dataset,
                              batch_size=int(BATCH_SIZE/world_size),
                              shuffle=False,
                              num_workers=0,
                              sampler=train_sampler,
                              collate_fn=custom_collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
if rank == 0:
    print("dataloader completed!")


## Load model
set_seed(42+rank)
model_name = "meta-llama/Llama-2-7b-hf"
llama = LCIRC(model_name, NUM_LAYERS, HID_DIM, NUM_HEADS, HEAD_DIM, INNER_DIM, K, arr).cuda(rank)
llama = DDP(module=llama, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=True)

optimizer = torch.optim.Adam(llama.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=140181*4, eta_min=0)
criterion = nn.CrossEntropyLoss(reduction='mean').to(rank)

if rank == 0:
    print(f"Model size: {sum([p.numel() * 2 for p in llama.module.model.parameters()]) / 1024 / 1024 / 1024}")
    print(f"Compressor size: {sum([p.numel() * 2 for p in llama.module.compressor.parameters()]) / 1024 / 1024 / 1024}")
    print(f"Latent size: {llama.module.latent.numel() * 2 / 1024 / 1024 / 1024}")
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
        data_length = data.size(1)

        # random segmenting
        seg = random_segments(data_length, K)

        llama.train()

        total_soft_prompt = None
        soft_prompt = llama.module.latent.repeat((BATCH_SIZE//NUM_GPUS), 1, 1)
        train_loss = 0
        for i in range(len(seg) - 2):
            next_max_len = seg[i + 1] + 4096
            if next_max_len > data_length:
                next_max_len = data_length

            data_gt = data[:, seg[i + 1]:next_max_len].to(device)

            # embedding of a segment
            seg_embedding = llama.module.model.model.model.embed_tokens(data[:, seg[i]:seg[i + 1]].to(device))

            # compress a segment
            soft_prompt = llama.module.compressor(soft_prompt, seg_embedding)

            # aggregate compressed features
            if i == 0:
                total_soft_prompt = soft_prompt.clone()
            else:
                total_soft_prompt = torch.cat((total_soft_prompt, soft_prompt), dim=1)

            prob = llama(data_gt, data_gt.size(1), total_soft_prompt)

            loss = criterion(prob.transpose(1, 2),
                             data_gt[:, 1:])

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
                train_loss += loss.item()

            soft_prompt = soft_prompt.detach()
            total_soft_prompt = total_soft_prompt[:, -64*50:].detach()

        if rank == 0:
            train_loss_all.append((train_loss*UPDATE_STEP)/(len(seg) - 2))

        # Validation
        if (idx + 1) % 1000 == 0 and rank == 0:
            torch.cuda.empty_cache()
            losses_eval = []
            for idx_valid, valid_data in enumerate(valid_dataloader):
                data_length = valid_data.size(1)

                llama.eval()

                if data_length - 4096 >= 2048:
                    segs = make_2k_segments(data_length - 4096)

                    total_soft_prompt = None
                    soft_prompt = llama.module.latent.repeat((BATCH_SIZE // NUM_GPUS), 1, 1)
                    with torch.no_grad():
                        for i in range(len(segs) - 1):
                            context_embedding = llama.module.model.model.model.embed_tokens(valid_data[:, segs[i]:segs[i + 1]].to(device))
                            soft_prompt = llama.module.compressor(soft_prompt, context_embedding)
                            if i == 0:
                                total_soft_prompt = soft_prompt.clone().detach()
                            else:
                                total_soft_prompt = torch.cat((total_soft_prompt, soft_prompt), dim=1)

                        output = llama.module.model(input_ids=valid_data[:, data_length - 4096:].to(device),
                                                    output_hidden_states=True, soft_prompt=total_soft_prompt).logits[:, -2048:-1]

                        loss = criterion(output.transpose(1, 2),
                                         valid_data[:, data_length - 2048 + 1:].to(device))

                        losses_eval.append(loss.item())
                else:
                    soft_prompt = llama.module.latent.repeat((BATCH_SIZE // NUM_GPUS), 1, 1)
                    with torch.no_grad():
                        context_embedding = llama.module.model.model.model.embed_tokens(valid_data[:, :data_length - 4096].to(device))
                        soft_prompt = llama.module.compressor(soft_prompt, context_embedding)

                        output = llama.module.model(input_ids=valid_data[:, data_length - 4096:].to(device),
                                                    output_hidden_states=True, soft_prompt=soft_prompt).logits[:, -2048:-1]

                        loss = criterion(output.transpose(1, 2),
                                         valid_data[:, data_length - 2048 + 1:].to(device))

                        losses_eval.append(loss.item())

            valid_loss = torch.tensor(losses_eval).mean()

            # save the model weights
            if len(valid_loss_all) == 0:
                CP_PATH = f'LCIRC_checkpoint_epoch{epoch}_{(idx + 1)}.pt'
                torch.save({'Xattn_layers_state_dict': llama.module.model.Xattn_layers.state_dict(),
                            'compressor_state_dict': llama.module.compressor.state_dict(),
                            'latent_state_dict': {k: v for k, v in llama.module.state_dict().items() if k == "latent"},
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}, CP_PATH)
            else:
                if valid_loss.item() < valid_loss_all[-1]:
                    CP_PATH = f'LCIRC_checkpoint_epoch{epoch}_{(idx + 1)}.pt'
                    torch.save({'Xattn_layers_state_dict': llama.module.model.Xattn_layers.state_dict(),
                                'compressor_state_dict': llama.module.compressor.state_dict(),
                                'latent_state_dict': {k: v for k, v in llama.module.state_dict().items() if k == "latent"},
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch}, CP_PATH)

            valid_loss_all.append(valid_loss.item())
            np.save(f"LCIRC_train_loss", np.array(train_loss_all))
            np.save(f"LCIRC_valid_loss", np.array(valid_loss_all))
            torch.cuda.empty_cache()

    # save the model weights
    if rank == 0:
        CP_PATH = f'LCIRC_checkpoint_epoch{epoch}_final.pt'
        torch.save({'Xattn_layers_state_dict': llama.module.model.Xattn_layers.state_dict(),
                    'compressor_state_dict': llama.module.compressor.state_dict(),
                    'latent_state_dict': {k: v for k, v in llama.module.state_dict().items() if k == "latent"},
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch}, CP_PATH)
        np.save(f"LCIRC_train_loss", np.array(train_loss_all))
        np.save(f"LCIRC_valid_loss", np.array(valid_loss_all))