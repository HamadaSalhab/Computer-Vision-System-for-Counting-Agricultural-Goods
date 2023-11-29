import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import pickle

class ClassificationModel(nn.Module):
    def __init__(self, input_channels):  
        super(ClassificationModel, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(64 * 16 * 16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through the network
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc(x)
        return self.sigmoid(x)


def train_one_epoch(model, loader, optimizer, loss_fn, epoch_num=-1):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: train",  
        leave=True,
    )
    model.train()
    train_loss = 0.0
    total = 0
    for i, batch in loop:
        images, labels = batch
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = loss_fn(outputs, labels.float())
        loss.backward()
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        train_loss += loss.item()
        total += 1  # Increment total to avoid division by zero
        loop.set_postfix({"loss": train_loss / total})
    return model



def val_one_epoch(model, loader, loss_fn, epoch_num=-1, best_so_far=0.0, ckpt_path=os.path.join(os.path.dirname(__file__),'../../data/interim/model.pkl' )):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: val", 
        leave=True,
    )
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for i, batch in loop:
            images, labels = batch
            outputs = model(images)
            loss = loss_fn(outputs.squeeze(), labels.float())  # Ensure outputs and labels have the same shape
            val_loss += loss.item()
            total += labels.size(0)

            # Apply a threshold to convert probabilities to binary class labels
            predicted = outputs >= 0.5
            correct += (predicted.squeeze().long() == labels).sum().item()

        accuracy = correct / total
        loop.set_postfix({"loss": val_loss / total, "acc": accuracy})

        if accuracy > best_so_far:
            torch.save(model.state_dict(), ckpt_path)
            best_so_far = accuracy

    return best_so_far

class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__(root, transform)



def train_model(model, epochs, optimizer, loss_fn, train_dataloader, val_dataloader):
    best = -float('inf')
    for epoch in range(epochs):
        model = train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch)
        best = val_one_epoch(model, val_dataloader, loss_fn, epoch, best_so_far=best)
    print("Best accuracy is :", best)
    return model


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

DATASET_RAW_RELATIVE_PATH = '../../data/interim/train'
DATASET_RAW_DIR = os.path.join(os.path.dirname(__file__), DATASET_RAW_RELATIVE_PATH)

if __name__ == '__main__':
    
    # Load the dataset
    full_dataset = CustomDataset(DATASET_RAW_DIR, transform=transform)


    # Split the dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Assuming 'CustomDataset' can provide labels for each sample
    labels = [label for _, label in train_dataset]

    # Calculate class weights
    class_counts = [labels.count(0), labels.count(1)]
    num_samples = sum(class_counts)
    weights = [num_samples / class_counts[label] for label in labels]

    # Create the sampler
    sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model initialization
    model = ClassificationModel(3)  # 3 input channels for RGB images

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()  # Ensure the output layer of your model is compatible with BCELoss

    trained_model = train_model(model = model, epochs=15, optimizer=optimizer, loss_fn=loss_fn, train_dataloader=train_dataloader, val_dataloader=val_dataloader)


