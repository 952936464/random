import torch
import opacus
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())
import random

print(random.random())

dict1 = {'A': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}
print(dict1)

for i in range(len(dict1['A'])):
    for j in range(len(dict1['A'][0])):
        dict1['A'][i][j] = random.random()
print(dict1)
