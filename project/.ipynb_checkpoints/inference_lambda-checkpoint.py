import torch
from typing import Callable, Optional, Tuple
from model_args import ModelArgs
from transformer import CustomTransformer
from tqdm import tqdm
import argparse
import torch
from data_loader import vocab, train_loader, valid_loader, numericalize_validation
import torch
import torch.autograd.profiler as profiler
from torcheval.metrics import Perplexity

# Set up argument parser
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
parser = argparse.ArgumentParser(description="Choose the device for computation.")
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    choices=["cuda", "cpu", "mps"],
    help="Device to use for computation (cuda, cpu, or mps).",
)
parser.add_argument(
    "--kvcache",
    type=str_to_bool,
    default=True,
    help="Inference with or without KV Cache (True/False)",
)
parser.add_argument(
    "--training",
    type=str_to_bool,
    default=False,
    help="Training yes or no",
)

# Parse arguments
args = parser.parse_args()

# Set the device
device = args.device
perplexity_metric = Perplexity(device = device)

print(device)

# ------------------ data loading -------------------
from data_loader import numericalize
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import datasets
from datasets import load_dataset
ds = load_dataset("EleutherAI/lambada_openai", "en")

# Tokenize the dataset
def tokenize_dataset(dataset):
    tokenized_data = []
    for example in dataset:
        tokenized_text = numericalize(example['text'])
        tokenized_data.append(tokenized_text)
    return tokenized_data

test_data = tokenize_dataset(ds['test'])

def collate_fn(batch):
    # Pad sequences to the same length
    padded_batch = pad_sequence([torch.tensor(seq) for seq in batch], batch_first=True, padding_value=vocab['<pad>'])
    return padded_batch

# Create DataLoader for training and test sets
batch_size = 32
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# --------------------- END --------------------------------------

model_args = ModelArgs()
model_args.vocab_size = len(vocab)
model_args.device = device

# ------------------ Training and KV Cache flags ---------------------
model_args.is_training = args.training
model_args.is_kv_cache = args.kvcache
print(model_args.is_training, model_args.is_kv_cache)
# ---------------------------------------
model = CustomTransformer(model_args).to(device)
model.load_state_dict(torch.load('./model_60'), )


def text_completion(model, prompts: list[int], temperature: float = 0.6, top_p: float = 0.9,
                    max_gen_len: Optional[int] = None):
    model.eval()  # Set model to evaluation mode
   
    prompts = prompts.to(device)
    if max_gen_len is None:
        max_gen_len = model.max_seq_len - 1

    batch_size = len(prompts)
    assert batch_size <= model.params.max_batch_size, f"batch size must be less than or equal to {model.params.max_batch_size}"
    max_prompt_len = max(len(prompt) for prompt in prompts)
    # # Make sure the prompt length is not larger than the maximum sequence length
    assert max_prompt_len <= model.params.max_seq_len, f"prompt length must be less than or equal to {model.params.max_seq_len}"
    total_len = min(model.params.max_seq_len, max_gen_len + max_prompt_len)
    
    total_len = max_prompt_len
    

    # # Create the list that will contain the generated tokens, along with the initial prompt tokens
    pad_id = vocab['<pad>']
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)  # initialised with pad_id
    for k, t in enumerate(prompts):
        tokens[k, :torch.tensor(t).shape[-1]] = torch.tensor(t, dtype=torch.long, device=device).squeeze(0)
    
    score_index = 10

    prompt_tokens_mask = torch.arange(tokens.size(1), device=tokens.device) <= score_index  # True for indices <= 10
    prompt_tokens_mask = prompt_tokens_mask.unsqueeze(0).expand(batch_size, -1)  # Expand to match batch size

    eos_reached = torch.tensor([False] * batch_size, device=device)
#     prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
    cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")

    for cur_pos in cur_iterator:
        with torch.no_grad():
            if model.is_kv_cache : # i will do token by token
                logits = model(tokens[:, cur_pos - 1:cur_pos], cur_pos - 1)
            else:
                logits = model(tokens[:, :cur_pos], 0)
                

        next_token = torch.argmax(logits[:, -1], dim=-1)
        
        if cur_pos > score_index:

            probs = torch.softmax(logits[:, -1], dim=-1).unsqueeze(1)
            ground_truth_token = tokens[:, cur_pos].unsqueeze(1)  # Shape: (batch_size,1)

            # Update the Perplexity metric
            perplexity_metric.update(probs, ground_truth_token)
            
            # next_token = torch.argmax(logits[:, -1], dim=-1)
            # acc = (next_token == ground_truth_token).sum().item() / total_len
            # print("accuracy: ",acc)
        

        next_token = next_token.reshape(-1)
        # Only replace token if it is a padding token
        # next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], tokens[:, cur_pos])
        tokens[:, cur_pos] = next_token
        # EOS is reached only if we found an EOS token for a padding position
        eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == vocab['<eos>'])
        if all(eos_reached):
            break


    out_tokens = []
    out_text = []
    for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
        # Cut to the EOS token, if present
        if vocab['<eos>'] in current_prompt_tokens:
            eos_idx = current_prompt_tokens.index(vocab['<eos>'])
            current_prompt_tokens = current_prompt_tokens[:eos_idx]
        out_tokens.append(current_prompt_tokens)
        out_text.append([vocab.get_itos()[curr_prmpt_tkns] for curr_prmpt_tkns in (current_prompt_tokens)])
    return (out_tokens, out_text)



original_prompt = []
original_target = []
d = {}
results = []

inference_data_count = 1
c = 0


with profiler.profile(
                          enabled=True,
                          use_device=device,
                          record_shapes=True,
                          with_flops=True,
                          profile_memory=True,
                          with_stack=True,
                          with_modules=False,
                          use_kineto=False,
                          use_cpu=True,
                          ) as prof:
    for batch in test_loader:
#         input_batch, target_batch = batch
        for p in batch:
            original_prompt.append([vocab.get_itos()[tok] for tok in (p)])
    
        for p in batch:
            original_target.append([vocab.get_itos()[tok] for tok in (p)])
    
        out_tokens, out_texts = text_completion(model, batch, temperature=0, max_gen_len=0)
    
        for i in range(len(out_texts)):
            results.append((original_prompt[i], out_texts[i] ,original_target[i]))
    
        c+=1
        if c >= inference_data_count:
            break

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# -------------- Extract total CUDA and CPU memory usage --------------
total_cuda_memory = 0
total_cpu_memory = 0

for event in prof.key_averages():
    if event.key == "cuda_memory_usage":
        total_cuda_memory += event.cuda_memory_usage
    elif event.key == "cpu_memory_usage":
        total_cpu_memory += event.cpu_memory_usage

# Print total memory usage
print(f"Total CUDA Memory Usage: {total_cuda_memory / 1024 ** 2:.2f} MB")
print(f"Total CPU Memory Usage: {total_cpu_memory / 1024 ** 2:.2f} MB")

# -------------- Perplexity --------------

perplexity = perplexity_metric.compute()
print(perplexity)

# -------------- ----------------------------

# table = []
# for avg in prof.key_averages():
#     table.append([
#         avg.key,
#         avg.cpu_time_total,
#         avg.cuda_time_total,
#         avg.cpu_memory_usage,
#         avg.cuda_memory_usage,
#         avg.flops,
#     ])
# headers = ["Name", "CPU Time (ms)", "CUDA Time (ms)", "CPU Memory (MB)", "CUDA Memory (MB)", "FLOPs"]
# table_sorted = sorted(table, key=lambda x: (-x[2], -x[4], -x[5]))  # Sort by CUDA time (desc), CUDA memory (desc), FLOPs (desc)

# table_sorted = table_sorted[:10]
# print(tabulate(table_sorted, headers=headers, tablefmt="pretty"))
# total_cpu_time = sum(row[1] for row in table)
# total_cuda_time = sum(row[2] for row in table)
# total_cpu_memory = sum(row[3] for row in table)
# total_cuda_memory = sum(row[4] for row in table)
# print("\nTotals:")
# print(f"Total CPU Time: {total_cpu_time:.2f} ms")
# print(f"Total CUDA Time: {total_cuda_time:.2f} ms")
# print(f"Total CPU Memory: {total_cpu_memory:.2f} MB")
# print(f"Total CUDA Memory: {total_cuda_memory:.2f} MB")

# for i in range(len(results)):

#     print(f"Input Text: {' '.join(results[i][0])}")
#     print(f"Generated Text: {' '.join(results[i][1])}")
    # print(f"Actual Text: {" ".join(results[i][2])}")
    # print(results[i])
    # break
