import os
import requests
import tiktoken
import numpy as np


# les miserables
#
# https://www.gutenberg.org/cache/epub/17489/pg17489.txt
# https://www.gutenberg.org/cache/epub/17493/pg17493.txt
# https://www.gutenberg.org/cache/epub/17494/pg17494.txt
# https://www.gutenberg.org/cache/epub/17518/pg17518.txt
# https://www.gutenberg.org/cache/epub/17519/pg17519.txt
#
# notre dame de Paris
#
# https://www.gutenberg.org/cache/epub/19657/pg19657.txt
#
# le Rhin
#
# https://www.gutenberg.org/files/41986/41986-0.txt
# https://www.gutenberg.org/files/42151/42151-0.txt
# https://www.gutenberg.org/files/40172/40172-0.txt
# https://www.gutenberg.org/files/40239/40239-0.txt
# https://www.gutenberg.org/files/29549/29549-0.txt
#
#
# download hugo's book notre dame de Paris
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_urls = [
'https://www.gutenberg.org/cache/epub/17489/pg17489.txt',
'https://www.gutenberg.org/cache/epub/17493/pg17493.txt',
'https://www.gutenberg.org/cache/epub/17494/pg17494.txt',
'https://www.gutenberg.org/cache/epub/17518/pg17518.txt',
'https://www.gutenberg.org/cache/epub/17519/pg17519.txt',
'https://www.gutenberg.org/cache/epub/19657/pg19657.txt',
'https://www.gutenberg.org/files/41986/41986-0.txt'
'https://www.gutenberg.org/files/42151/42151-0.txt'
'https://www.gutenberg.org/files/40172/40172-0.txt'
'https://www.gutenberg.org/files/40239/40239-0.txt'
'https://www.gutenberg.org/files/29549/29549-0.txt'
 ]
    for data_url in data_urls :
      with open(input_file_path, 'a') as f:
        f.write(requests.get(data_url).text)
        f.write('\n')

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
