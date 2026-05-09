import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms


class AugmentedDataset(Dataset):
    """Dataset wrapper that applies a transform to one element on __getitem__."""

    def __init__(self, dataset, transform, transform_index=0):
        self.dataset = dataset
        self.transform = transform
        self.transform_index = transform_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img = data[self.transform_index]
        return tuple((self.transform(img), *data[1:]))


def compute_embeddings(x, process_fn, batch_size, input_size=None):
    """Run a model over a numpy array of images and return embeddings as numpy."""
    data = TensorDataset(torch.from_numpy(x))

    if input_size is not None:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
        data = AugmentedDataset(data, data_transforms)

    loader = DataLoader(data, batch_size=batch_size)
    result = []

    with torch.no_grad():
        for batch in loader:
            result.append(process_fn(batch))

    return torch.cat(result).cpu().numpy()
