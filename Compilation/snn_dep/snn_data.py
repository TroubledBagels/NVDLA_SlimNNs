import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip, Resize, CenterCrop


def load_data(ds: str, bs: int, v: int) -> (DataLoader, DataLoader):
    if v > 1: print(f"[OK] Loading dataset: {ds}...")

    test_data = None
    train_data = None

    if ds == "CIFAR10":
        print("[OK] Calculating transforms...")
        transform = Compose([
            Resize((70, 70)),
            CenterCrop((64, 64)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        transform1 = Compose([
            Resize((70, 70)),
            CenterCrop((64, 64)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            RandomHorizontalFlip(p=1)
        ])
        transform2 = Compose([
            Resize((70, 70)),
            CenterCrop((64, 64)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            RandomResizedCrop(64, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))
        ])
        transform3 = Compose([
            Resize((70, 70)),
            CenterCrop((64, 64)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            RandomResizedCrop(64, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
            RandomHorizontalFlip(p=1)
        ])

        print("[OK] Applying transforms...")
        train_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform
        )
        train_data1 = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform1
        )
        train_data2 = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform2
        )
        train_data3 = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform3
        )

        print("[OK] Concatenating datasets...")
        train_data = torch.utils.data.ConcatDataset([train_data, train_data1, train_data2, train_data3])

        print("[OK] Applying test transforms...")
        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform
        )

    if v > 1:
        print(f"[OK] Dataset {ds} loaded.")
        print(f"    Training data size: {len(train_data)}")
        print(f"    Test data size: {len(test_data)}")
        print(f"[OK] Creating dataloaders with batch size {bs}...")

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=True)

    if v > 1: print(f"[OK] Dataloaders created.")

    return train_dataloader, test_dataloader