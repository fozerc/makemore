import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(words)))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i;s for i,s in stoi.items()}

block_size = 10
X, Y = [], []

for w in words[5:]:
    print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + ix

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn(27, 2)
