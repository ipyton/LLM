import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class Tokenizer(object):
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        words = [item.strip() for item in words if item.strip()]
        words = [item if item in self.str_to_int
                 else "<|unk|>" for item in words]
        ids = [self.str_to_int[s] for s in words]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text,  allowed_special={"<|endoftext|>"})
tokens = tokenizer.decode(integers)

with open("./the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    print(raw_text)
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))


print(torch.__version__)
print(torch.cuda.is_available())