import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    def init(self, input_channels, num_classes):
        super(ClassificationModel, self).init()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
    
    def forward(self, x):
        # Forward pass through the network
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc(x)
        return x


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    epoch_num=-1
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch}: train",
        leave=True,
    )
    model.train()
    train_loss = 0.0
    total = 0
    for i, batch in loop:
        images, labels = batch
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        train_loss +=loss.item()
        loop.set_postfix({"loss": train_loss/total})




def val_one_epoch(
    model,
    loader,
    loss_fn,
    epoch_num=-1,
    best_so_far=0.0,
    ckpt_path='best.pt'
):
    
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch}: val",
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
            loss = loss_fn(outputs, labels)
            val_loss+=loss.item()
            loop.set_postfix({"loss": val_loss/total, "acc": correct / total})
        
        if correct / total > best:
            torch.save(model.state_dict(), ckpt_path)
            return correct / total

    return best_so_far





def train_model(epochs, optimizer, loss_fn, num_classes):
    best = -float('inf')
    model = ClassificationModel(3, num_classes)
    for epoch in range(epochs):
        train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch)
        best = val_one_epoch(model, val_dataloader, loss_fn, epoch, best_so_far=best)

    return model
