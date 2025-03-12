import torch
from typing import Callable, Optional, Tuple
from model_args import ModelArgs
from transformer import CustomTransformer
from tqdm import tqdm
import argparse
import torch
from data_loader import vocab, train_loader, valid_loader, numericalize_validation

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

# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'mps'
print(device)

model_args = ModelArgs()
model_args.vocab_size = len(vocab)
model_args.device = device

model_args.is_training = False
model = CustomTransformer(model_args).to(device)
model.load_state_dict(torch.load('./model'), )

def text_completion(model, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9,
                    max_gen_len: Optional[int] = None):
    model.eval()  # Set model to evaluation mode
    model.is_training = False

    if max_gen_len is None:
        max_gen_len = model.max_seq_len - 1

    # Convert each prompt into tokens
    prompt_tokens = []
    for p in prompts:
        tokens = torch.tensor(numericalize_validation(p), dtype=torch.long).unsqueeze(0).to(
            device)  # Add batch dimension
        prompt_tokens.append(tokens)
    print(prompt_tokens)
    # prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
    # Make sure the batch size is not too large
    batch_size = len(prompt_tokens)
    assert batch_size <= model.params.max_batch_size, f"batch size must be less than or equal to {model.params.max_batch_size}"
    max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    # Make sure the prompt length is not larger than the maximum sequence length
    assert max_prompt_len <= model.params.max_seq_len, f"prompt length must be less than or equal to {model.params.max_seq_len}"
    total_len = min(model.params.max_seq_len, max_gen_len + max_prompt_len)

    # Create the list that will contain the generated tokens, along with the initial prompt tokens
    pad_id = vocab['<pad>']
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)  # initialised with pad_id
    for k, t in enumerate(prompt_tokens):
        # Populate the initial tokens with the prompt tokens
        # print(f"Shape of t: {torch.tensor(t).shape[-1]}")
        # print(len(t))
        # print(f"Shape of t: {t.squeeze(0).shape}")

        tokens[k, :torch.tensor(t).shape[-1]] = torch.tensor(t, dtype=torch.long, device=device).squeeze(0)

    eos_reached = torch.tensor([False] * batch_size, device=device)
    prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
    cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
    strt_pos = 0
    for cur_pos in cur_iterator:
        with torch.no_grad():
            model.is_training = False
            logits = model(tokens[:, cur_pos - 1:cur_pos], cur_pos - 1)

        # if temperature > 0:
        #     # The temperature is applied before the softmax
        #     probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        #     next_token = self._sample_top_p(probs, top_p)
        # else:
            # Greedily select the token with the max probability
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


# prompts = [
#         "Simply put, the theory of relativity states that ",
#         "If Google was an Italian company founded in Milan, it would",
#         # Few shot promt
#         """Translate English to French:

#         sea otter => loutre de mer
#         peppermint => menthe poivrée
#         plush girafe => girafe peluche
#         cheese =>""",
#         # Zero shot prompt
#         """Tell me if the following person is actually Doraemon disguised as human:
#         Name: Umar Jamil
#         Decision:
#         """
#     ]
prompts = [
    "Simply put, the theory of relativity states that ",
]
out_tokens, out_texts = text_completion(model, prompts, temperature=0, max_gen_len=64)
assert len(out_texts) == len(prompts)
for i in range(len(out_texts)):
    print(f'{out_texts[i]}')
    print('-' * 50)