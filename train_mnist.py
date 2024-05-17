import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os
from src.models import PixelCNN

# Hyperparameters
batch_size = 128
n_epochs = 50
learning_rate = 0.001
save_path = './logs'

if not os.path.exists(save_path):
    os.makedirs(save_path)

transform = transforms.Compose([
    transforms.ToTensor()
])

# Train, Validation, Test dataset
dataset = datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
test_dataset = datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PixelCNN(n_channel=1, h=128, layers=15, feature_maps=32).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

# Load the last model if exists
best_loss = float('inf')
best_ckpt_path = os.path.join(save_path, 'best.ckpt')
if os.path.exists(best_ckpt_path):
    model.load_state_dict(torch.load(best_ckpt_path))
    print(f'Loaded best model with loss {best_loss} from {best_ckpt_path}')

# Training
for epoch in range(n_epochs):
    
    # Train
    model.train()
    train_losses = []
    train_iterator = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{n_epochs}', unit='batch')
    for batch_idx, (data, target) in enumerate(train_iterator):
        
        # [batch_size, 1, 28, 28]
        data = data.to(device)
        target = (data * 255).long().to(device)             # 0-1 -> 0-255
        optimizer.zero_grad()
    
        output = model(data)                                # [batch_size, 256, 1, 28, 28]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        train_iterator.set_postfix(loss=loss.item())
    
    avg_train_loss = sum(train_losses) / len(train_losses)

    # Validation
    val_losses = []
    model.eval()
    val_iterator = tqdm(val_loader, desc=f'Validate Epoch {epoch + 1}/{n_epochs}', unit='batch')
    with torch.no_grad():
        for data, target in val_iterator:
            
            # [batch_size, 1, 28, 28]
            data = data.to(device)
            target = (data * 255).long().to(device)
            output = model(data)                                                    # [batch_size, 256, 1, 28, 28]
            
            # sample
            probs = torch.softmax(output, dim = 1).data                             # [batch_size, 256, 1, 28, 28]
            output_tensor = torch.argmax(probs, dim=1)/ 255.                        # [batch_size, 1, 28, 28]
            save_image(data, os.path.join(save_path, f'validatation_src.png'))
            save_image(output_tensor, os.path.join(save_path, f'validatation.png'))
            
            loss = criterion(output, target)
            val_losses.append(loss.item())
            val_iterator.set_postfix(loss=loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)

    # Learning rate scheduler
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(save_path, 'best.ckpt'))
    
    tqdm.write(f'====> Epoch: {epoch + 1} Average validation loss: {avg_val_loss:.4f} | Best loss: {best_loss:.4f} | Average train loss: {avg_train_loss:.4f} | LR: {current_lr}')

    # sample
    print('Generating from validation data...')
    for data, _ in val_loader:
        # [batch_size, 1, 28, 28]
        data = data.to(device)
        ori_data = data.clone()
        data[:, :, 20:, :] = 0  # set the bottom half to 0 to generate from top half
        break  # only generate from the first batch
    
    save_image(data, os.path.join(save_path, f'generated_from_val_epoch_{epoch + 1}_src.png'))

    # Auto-regressive generation the bottom half
    model.eval()
    with torch.no_grad():
        for i in range(20, 28):                                        # generate bottom half
            for j in range(28):
                out = model(data)                                      # [batch_size, 256, 1, 28, 28]
                probs = torch.softmax(out[:, :, 0, i, j], dim=1).data  # [batch_size, 256]
                pixel = torch.multinomial(probs, 1).float() / 255.
                data[:, 0, i, j] = pixel[:, 0]                         # update pixel value
    
    mse = torch.mean((data.float() - ori_data.float()).pow(2))
    print(f'MSE: {mse.item()}')
    save_image(data, os.path.join(save_path, f'generated_from_val_epoch_{epoch + 1}.png'))
