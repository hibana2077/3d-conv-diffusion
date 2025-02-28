import torch
import torch.nn as nn

# class_labels = torch.randint(0, 10, (10,))
class_labels = torch.arange(0, 10).long()
print(class_labels)