import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore


class CustomKafkaDataset(Dataset):
    def __init__(self, data, label=None):
        if len(data) == 0:
            raise ValueError("Empty dataset provided.")
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the data point at the given index
        sample = self.data[idx]

        # Convert the data point to tensors
        # TODO: Check other datatypes
        features = torch.tensor([value for key, value in sample.items() if key != self.label], dtype=torch.float32)
        
        label = torch.tensor(sample[self.label], dtype=torch.float32)
        return features, label
