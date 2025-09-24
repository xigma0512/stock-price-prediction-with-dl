import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data_loader(X, y, batch_size = 64):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader