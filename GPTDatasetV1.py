import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # use tokenizer gpt2 to encoding txt
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader



with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

data_loader = create_dataloader_v1(raw_text, batch_size=4, max_length=4, stride=1, shuffle=False)
# dataset contains a iterator
data_iter = iter(data_loader)
inputs, targets = next(data_iter)

vocab_size = 50257
output_dim = 256

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embeddings = embedding_layer(torch.arange(4))
print(pos_embeddings.shape)
