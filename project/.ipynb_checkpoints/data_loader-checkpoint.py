import torch, torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
# Step 1: Load dataset
train_iter, valid_iter, test_iter = torchtext.datasets.PennTreebank(root='.data', split=('train', 'valid', 'test'))

# Step 2: Tokenization
tokenizer = get_tokenizer("basic_english")

# Function to yield tokens for vocabulary building
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Step 3: Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<pad>', '<unk>', '<bos>', '<eos>'])
vocab.set_default_index(vocab['<unk>'])  # Handle out-of-vocabulary tokens

# Convert text to token IDs
def numericalize(text):
    return [vocab['<bos>']] + [vocab[token] for token in tokenizer(text)] + [vocab['<eos>']]

def numericalize_validation(text):
    return [vocab['<bos>']] + [vocab[token] for token in tokenizer(text)]


# Step 4: Prepare dataset for training
class PennTreebankDataset(Dataset):
    def __init__(self, data_iter):
        self.data = [numericalize(text) for text in data_iter]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])  # Input, Target

# Create dataset instances
train_dataset = PennTreebankDataset(torchtext.datasets.PennTreebank(root='.data', split='train'))
valid_dataset = PennTreebankDataset(torchtext.datasets.PennTreebank(root='.data', split='valid'))

# Step 5: Create DataLoader with padding
def collate_batch(batch):
    inputs, targets = zip(*batch)
    input_lens = [len(seq) for seq in inputs]
    max_len = max(input_lens)

    input_batch = [torch.cat([seq, torch.full((max_len - len(seq),), vocab['<pad>'])]) for seq in inputs]
    target_batch = [torch.cat([seq, torch.full((max_len - len(seq),), vocab['<pad>'])]) for seq in targets]

    return torch.stack(input_batch), torch.stack(target_batch)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
