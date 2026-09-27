import torch

from fl_model import FLIDSModel


model = FLIDSModel()

parameters = {}

for name, parameter in model.state_dict().items():
    parameters[name] = parameter.detach().clone()

print("Extracted model parameters:\n")

for name, parameter in parameters.items():
    print(name)
    print("Shape:", parameter.shape)
    print()