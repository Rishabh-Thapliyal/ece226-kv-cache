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

# Set up argument parser
parser = argparse.ArgumentParser(description="Choose the device for computation.")
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    choices=["cuda", "cpu", "mps"],
    help="Device to use for computation (cuda, cpu, or mps).",
)

# Parse arguments
args = parser.parse_args()

# Set the device
device = args.device

print(device)

model_args = ModelArgs()
model_args.vocab_size = len(vocab)
model_args.device = device

model_args.is_training = False
model = CustomTransformer(model_args).to(device)
model.load_state_dict(torch.load('./model_60'), )


def text_completion(model, prompts: list[int], temperature: float = 0.6, top_p: float = 0.9,
                    max_gen_len: Optional[int] = None):
    model.eval()  # Set model to evaluation mode
    model.is_training = False
    prompts = prompts.to(device)
    if max_gen_len is None:
        max_gen_len = model.max_seq_len - 1

    batch_size = len(prompts)
    assert batch_size <= model.params.max_batch_size, f"batch size must be less than or equal to {model.params.max_batch_size}"
    max_prompt_len = max(len(prompt) for prompt in prompts)
    # # Make sure the prompt length is not larger than the maximum sequence length
    assert max_prompt_len <= model.params.max_seq_len, f"prompt length must be less than or equal to {model.params.max_seq_len}"
    total_len = min(model.params.max_seq_len, max_gen_len + max_prompt_len)
    #
    # # Create the list that will contain the generated tokens, along with the initial prompt tokens
    pad_id = vocab['<pad>']
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)  # initialised with pad_id
    for k, t in enumerate(prompts):
        tokens[k, :torch.tensor(t).shape[-1]] = torch.tensor(t, dtype=torch.long, device=device).squeeze(0)

    eos_reached = torch.tensor([False] * batch_size, device=device)
    prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
    cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")

    for cur_pos in cur_iterator:
        with torch.no_grad():
            model.is_training = False
            logits = model(tokens[:, cur_pos - 1:cur_pos], cur_pos - 1)

        next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # Only replace token if it is a padding token
        next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
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

inference_data_count = 10
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
    for batch in valid_loader:
        input_batch, target_batch = batch
        for p in input_batch:
            original_prompt.append([vocab.get_itos()[tok] for tok in (p)])
    
        for p in target_batch:
            original_target.append([vocab.get_itos()[tok] for tok in (p)])
    
        out_tokens, out_texts = text_completion(model, input_batch, temperature=0, max_gen_len=64)
    
        for i in range(len(out_texts)):
            results.append((original_prompt[i], out_texts[i] ,original_target[i]))
    
        c+=1
        if c >= inference_data_count:
            break

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

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