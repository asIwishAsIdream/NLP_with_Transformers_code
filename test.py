# Create a convolutional neural network to clssify MNIST images in PyTorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x))) # 28x28x1 --> 14x14x32
        x = self.pool(F.relu(self.conv2(x))) # 14x14x32 --> 7x7x64
        # Flatten image input
        x = x.view(-1, 64 * 7 * 7)
        # Add dropout layer
        x = self.dropout(x)
        # Add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # Add dropout layer
        x = self.dropout(x)
        # Add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x
    
# Create the network and look at its text representation
model = CNN()
print(model)
# test
