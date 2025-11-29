import torch
import wandb

from models.customnet import CustomNet2
from dataset.dataset import get_dataloaders
from eval import validate
from train import train_epoch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "architecture": "CustomNet2",
        "dataset": "TinyImageNet"
    }

    dataset_path = 'data/tiny-imagenet-200'

    train_loader, val_loader = get_dataloaders(dataset_path, batch_size=config["batch_size"], num_workers=2)

    wandb.init(project='AML-lab', name='lab03-CustomNet2', config=config)

    model = CustomNet2().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])

    best_acc = 0

    for epoch in range(1, config["num_epochs"] + 1):
        print('\nEpoch %d\n--------------------------' % epoch)

        train_loss, train_acc = train_epoch(device, epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_loss, val_acc = validate(model, val_loader, criterion)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # Best validation accuracy
        best_acc = max(best_acc, val_acc)

    print(f'Best validation accuracy: {best_acc:.2f}%')
