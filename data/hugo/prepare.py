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

# start from
include_patterns = [
  ("*** START", "            *** END")
]
# remove license and html in books
exclude_patterns = [
  ("<!DOCTYPE html>", "</html>")
]

data = ""
counter = 0
logEveryCount = 1000

with open(input_file_path, 'r') as f:
    exclude_patern = None
    include_patern = None
    while True:
      line = f.readline()
      if not line:
        break
      counter += 1

      if counter % logEveryCount == 0:
        print(f'reading line {counter}')

      # end patern
      if include_patern:
        if line.startswith(include_patern[1]):
          print(f'including ended:{counter}')
          include_patern = None
          continue
      elif exclude_patern:
        if line.startswith(exclude_patern[1]):
          print(f'excluding ended:{counter}')
          exclude_patern = None
        # don't process excluded lines
        continue

      # start patern
      for patern in include_patterns:
        if line.startswith(patern[0]):
          include_patern = patern
          print(f'including from:{counter}')
          continue
      for patern in exclude_patterns:
        if line.startswith(patern[0]):
          exclude_patern = patern
          print(f'excluding from:{counter}')
          continue
      data += line

n = len(data)

# shuffle data or we'll always validate the same book
# check that it matches train_hugo.py
block_size = 1024
chunk = 10*1024
train_data = ""
val_data = ""
for i in range(0,n,chunk):
  train_data += data[i:i+int(chunk*0.9)]
  j = i+int(chunk*0.9)
  # avoid cuting lines
  while j<(n-1) and data[j] != '\n':
    train_data += data[j]
    j+=1
  if j == n-1:
    break
  k = i+chunk-1
  if k>(n-1):
    k = n-1
    if k > j:
      break
  while k>j and data[k] != '\n':
    k-=1
  val_data += data[j:k]
  train_data += data[k:i+chunk]


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
