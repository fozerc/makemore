import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()

block_size = 10
X, Y = [], []

for w in words[5:]:
    print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]

C = torch.randn(27, 2)
