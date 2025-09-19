import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class CompactMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: 1 → 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        # Block 2: 8 → 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # Block 3: 16 → 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.05)  # lighter dropout
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10, bias=False)

    def forward(self, x):
        x = self.pool(self.conv1(x))   # 8x14x14
        x = self.dropout(x)
        x = self.pool(self.conv2(x))   # 16x7x7
        x = self.dropout(x)
        x = self.conv3(x)              # 32x7x7
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for data, target in tqdm(train_loader, leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(target).sum().item()
        total += data.size(0)
    return running_loss / total, 100. * correct / total


def evaluate(model, device, loader, criterion):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss += criterion(outputs, target).item() * data.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(target).sum().item()
            total += data.size(0)
    return loss / total, 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Slightly lighter augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation(5),  # smaller rotation
        transforms.RandomAffine(0, translate=(0.04, 0.04)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST('.', train=True, download=True, transform=transform_train)
    train_set, val_set = random_split(full_train, [50000, 10000])
    val_set.dataset = datasets.MNIST('.', train=True, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)

    model = CompactMNISTNet().to(device)
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    assert total_params < 20000, f"Parameter count {total_params} exceeds 20k limit"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch:02d}: Train acc {tr_acc:.2f}% | Val acc {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_mnist.pth")
        if best_val_acc >= 99.4:
            break

    print(f"Best validation acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()