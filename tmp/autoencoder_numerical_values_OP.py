create an autoencoder pytorch encoding continous variables

import torch
import torch.nn as nn
from torch.autograd import Variable

# Create an autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoder_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        
        # Encoder layers
        self.fc1 = nn.Linear(in_features=self.input_dim, out_features=self.encoder_dim)
        self.fc2 = nn.Linear(in_features=self.encoder_dim, out_features=self.encoder_dim)
        
        # Decoder layers
        self.fc3 = nn.Linear(in_features=self.encoder_dim, out_features=self.input_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Encoder
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Decoder
        x = self.fc3(x)
        
        return x

# Create an instance of the autoencoder
autoencoder = Autoencoder(input_dim=4, encoder_dim=2)
print(autoencoder)

# Create an optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Create a loss function
criterion = nn.MSELoss()

# Train the autoencoder
for i in range(1000):
    # Create a random tensor
    x = Variable(torch.randn(4))
    
    # Reset the gradients
    optimizer.zero_grad()
    
    # Forward pass
    y = autoencoder(x)
    
    # Compute the loss
    loss = criterion(y, x)
    
    # Backward pass
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    # Print the loss
    if (i+1) % 200 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(i+1, 1000, loss.item()))
