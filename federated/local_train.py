import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from federated.fl_model import FLIDSModel


def train_client(
    client_id,
    global_state=None,
    epochs=2,
    batch_size=128,
    learning_rate=0.001
):

    x_path = f"data/processed/reduced/iid/client_{client_id}_X.npy"
    y_path = f"data/partitions/iid/client_{client_id}_y.npy"

    X = np.load(x_path)
    y = np.load(y_path)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FLIDSModel()

    if global_state is not None:
        model.load_state_dict(global_state)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    total_loss = 0.0

    for epoch in range(epochs):

        epoch_loss = 0.0

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            output = model(batch_X)

            loss = criterion(output, batch_y)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        total_loss = epoch_loss / len(loader)

    client_update = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }

    return client_update, len(dataset), total_loss


if __name__ == "__main__":

    update, samples, loss = train_client(client_id=1)

    print("Client:", 1)
    print("Samples:", samples)
    print("Final loss:", loss)