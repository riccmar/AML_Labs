import torch

from dataset.dataset import get_dataloaders
from models.customnet import CustomNet2
from eval import validate

def train_epoch(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    size = len(train_loader.dataset)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 500 == 0:
          loss, current = loss.item(), batch_idx * len(inputs)
          print('loss: %.7f [%d/%d]' % (loss, current, size))

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'\nTrain Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    momentum = 0.9

    train_loader, val_loader = get_dataloaders('data/tiny-imagenet-200', batch_size=batch_size, num_workers=2)

    model = CustomNet2().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    best_acc = 0

    for epoch in range(1, num_epochs + 1):
        print('\nEpoch %d\n--------------------------' % epoch)

        train_epoch(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)


    print(f'Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()