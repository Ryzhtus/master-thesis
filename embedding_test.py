import torch
import torch.nn as nn 

linear = nn.Linear(5, 10)
embedding = nn.Embedding(5, 10)

input_example_linear = torch.tensor([1, 2, 1, 0, 3], dtype=torch.float).unsqueeze(dim=0)
input_example_embedding = torch.tensor([1, 2, 1, 0, 3], dtype=torch.long).unsqueeze(dim=0)

print("Linear Output: ", linear(input_example_linear))
print("Embedding Output: ", embedding(input_example_embedding))