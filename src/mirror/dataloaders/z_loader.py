import torch
from torch.utils.data import DataLoader, Dataset


class LatentDataset(Dataset):
    def __init__(self, num_samples: int, latent_dim: int):
        self.z = torch.randn(num_samples, latent_dim)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx]


def build_latent_dataloader(
    num_samples, latent_dim: int, batch_size: int = 1024, num_workers: int = 4
):
    return DataLoader(
        LatentDataset(num_samples, latent_dim),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
