import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, values):
        self.X = torch.from_numpy(X)
        self.values = torch.from_numpy(values)

    def __len__(self):
        return len(self.meanders)

    def __getitem__(self, idx):
        return self.meanders[idx], self.spirals[idx], self.circles[idx], self.values[idx]
