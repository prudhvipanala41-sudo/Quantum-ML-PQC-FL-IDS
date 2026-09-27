import torch
import torch.nn as nn


class FLIDSModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = FLIDSModel()

    x = torch.randn(5, 8)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print("\nModel:")
    print(model)

    print("\nParameters:")
    for name, parameter in model.named_parameters():
        print(name, parameter.shape)