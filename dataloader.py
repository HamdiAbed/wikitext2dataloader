import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import WikiText2
from transformers import GPT2Tokenizer

class WT2Dataset(Dataset):
    def __init__(self, split, max_len=512):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_len = max_len
        self.data = self.load_data(split)

    def load_data(self, split):
        dataset = WikiText2(split=split)
        tokenized_data = []
        for item in dataset:
            tokens = self.tokenizer.encode(item)
            tokenized_data.extend(tokens)
        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx:idx + self.max_len]

# Create a collate function to handle batching and padding
def collate_fn(batch):
    # Convert inner lists to tensors
    batch = [torch.tensor(seq) for seq in batch]
    # Sort the batch by sequence length (descending order)
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    # Get the lengths of each sequence
    lengths = [len(seq) for seq in batch]
    # Pad sequences to have the same length
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return padded_batch, lengths

# Create the dataset
seq_len = 32
dataset = WT2Dataset(split='train', max_len = seq_len)

# Create a DataLoader object
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

