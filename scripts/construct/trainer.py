import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
from .dataloader import collate_fn

class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, device, 
                 batch_size=32, learning_rate=3e-4, num_workers=4):
        """
        Args:
            model: The encoder-decoder model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: torch device (cuda/cpu)
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_workers: Number of workers for data loading
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        # Initialize dataloaders with custom collate function
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.wrd2idx["<pad>"])
        
        # Move model and criterion to device
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        
        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        
    def train_epoch(self):
        """Run one epoch of training"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, captions) in enumerate(progress_bar):
            # Move batch to device
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(images, captions)
            
            # Calculate loss (excluding <start> token)
            targets = captions[:, 1:]
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, captions in self.val_loader:
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                outputs, _ = self.model(images, captions)
                targets = captions[:, 1:]
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.reshape(-1))
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs, checkpoint_dir="checkpoints"):
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save model checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'best_loss': self.best_loss
                }, checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}") 