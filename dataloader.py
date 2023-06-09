import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import WikiText2
from transformers import GPT2Tokenizer

class WT2Dataset(Dataset):
    def __init__(self,datasetname, split, max_len=512):
        self.dataset = datasetname
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = self.tokenizer.vocab_size
        self.vocab = self.tokenizer.get_vocab()
        self.max_len = max_len
        self.data = self.load_data(split)

    def load_data(self, split):
        dataset = self.dataset(split=split)
        tokenized_data = []
        for item in dataset:
            tokens = self.tokenizer.encode(item)
            tokenized_data.extend(tokens)
        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx:idx + self.max_len]
    



def collate_fn(batch, max_len):
    padding_value = 0  # Define the padding value here
    # Sort the batch by sequence length (descending order)
    batch = sorted(batch, key=lambda x: len(x), reverse=True)

    # Get the lengths of each sequence
    lengths = [len(seq) for seq in batch]

    # Pad sequences to the maximum length
    padded_batch = []
    for seq in batch:
        padded_seq = seq + [padding_value] * (max_len - len(seq))
        padded_batch.append(padded_seq)

    return torch.tensor(padded_batch)

#load the data
dataset = WikiText2()
train_dataset = WT2Dataset(dataset, split='train', max_len = args.seq_len)
valid_dataset = WT2Dataset(dataset, split='valid', max_len = args.seq_len)

#create dataloader
dataset_train = DataLoader(train_dataset,  batch_size=args.batch_size,drop_last = True, collate_fn=lambda batch: collate_fn(batch, args.seq_len))
dataset_val   = DataLoader(valid_dataset, batch_size = args.eval_batch_size,drop_last = True, collate_fn=lambda batch: collate_fn(batch, args.seq_len))
ntokens = train_dataset.vocab_size
vocab = train_dataset.vocab

