from torch import nn

# Define the custom neural network (taken from slide)
class CustomNet2(nn.Module):
    def __init__(self):
        super(CustomNet2, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2)

        self.flatten = nn.Flatten(2)

        self.fc1 = nn.Linear(1024, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x).relu()

        x = self.flatten(x).mean(-1)

        x = self.fc1(x)

        return x

# Define the custom neural network (created in lab2)
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Add dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Assuming the input image size is 224x224 and applying maxpool after each conv
        # After conv1 (224x224) -> maxpool (112x112)
        # After conv2 (112x112) -> maxpool (56x56)
        # After conv3 (56x56) -> maxpool (28x28)
        self.fc1 = nn.Linear(256 * 28 * 28, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # B x 64 x 112 x 112
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # B x 128 x 56 x 56
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # B x 256 x 28 x 28
        # -1: This is a placeholder that tells PyTorch to automatically infer the dimension
        # based on the other dimensions and the total number of elements in the tensor.
        # In this case, it will be the batch size B.
        x = x.view(-1, 256 * 28 * 28) # Flatten the tensor
        # Apply dropout
        x = self.dropout(x)
        x = self.fc1(x)

        return x