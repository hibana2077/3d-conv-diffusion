import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()
x = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
y = torch.tensor([0, 1])
loss = ce(x, y)