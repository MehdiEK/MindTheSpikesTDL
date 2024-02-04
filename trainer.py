"""
"""

import torch
import torch.nn as nn
import torch.optim as optim

def train_neural_model(model, X, y, num_epochs, verbose=False):
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)

        # Compute the loss
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')    

    return model