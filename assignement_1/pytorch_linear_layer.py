import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        
        self.linear = nn.Linear(input_dim, output_dim)
        torch.manual_seed(42)
        nn.init.uniform_(self.linear.weight, a=-1.0, b=1.0)  
        nn.init.uniform_(self.linear.bias, a=-1.0, b=1.0)    
    
    def forward(self, x):
        x = self.normalize_columns(x)
        x = self.linear(x)
        return x
    
    def normalize_columns(self, x):
        column_sums = torch.sum(x**2, dim=0, keepdim=True)
        x_normalized = x / torch.sqrt(column_sums + 1e-8)  
        return x_normalized

input_dim = 10
output_dim = 3

model = LinearModule(input_dim=input_dim, output_dim=output_dim)

inputs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float64)

print(type(inputs))

model = model.to(dtype=torch.float64)


print("Outputs:", outputs)


cpp_output = [-9.35618, -3.75788, 5.15797]  # Replace with actual C++ output
assert torch.allclose(torch.tensor(cpp_output), outputs, atol=1e-6), "Outputs are not close!"

