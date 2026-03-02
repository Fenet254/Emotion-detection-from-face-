"""
Emotion Detection Model Training Script
Using PyTorch with GPU acceleration
Detects 9 emotions: angry, confused, disgust, fear, happy, neutral, sad, shy, surprise
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import time
from collections import Counter

# =====================================================
# CONFIGURATION - Modify these settings as needed
# =====================================================

# Dataset path
DATA_DIR = "EDFF_dataset"

# Training settings
NUM_EPOCHS = 100  # High number of epochs for best accuracy
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

# Device configuration - Auto-detect GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Emotion classes (9 classes from dataset)
EMOTION_CLASSES = [
    'angry', 'confused', 'disgust', 'fear', 
    'happy', 'neutral', 'sad', 'shy', 'surprise'
]

NUM_CLASSES = len(EMOTION_CLASSES)
print(f"Number of emotion classes: {NUM_CLASSES}")
print(f"Emotion classes: {EMOTION_CLASSES}")

# =====================================================
# DATA LOADING
# =====================================================

class EmotionDataset(Dataset):
    """Custom Dataset for Emotion Detection"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all images and labels
        for emotion_idx, emotion in enumerate(EMOTION_CLASSES):
            emotion_dir = self.root_dir / emotion
            if emotion_dir.exists():
                for img_path in emotion_dir.glob("*.jpg"):
                    self.images.append(str(img_path))
                    self.labels.append(emotion_idx)
        
        print(f"Loaded {len(self.images)} images")
        print(f"Class distribution: {Counter(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Data transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def split_dataset(dataset, val_split=0.2):
    """Split dataset into training and validation sets"""
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply different transforms
    train_dataset.dataset = dataset
    val_dataset.dataset = dataset
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    return train_dataset, val_dataset


# =====================================================
# MODEL DEFINITION
# =====================================================

def create_model(num_classes):
    """Create emotion detection model using ResNet18 transfer learning"""
    
    # Use ResNet18 pretrained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers for faster training (optional - can unfreeze for fine-tuning)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Modify the final fully connected layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


# =====================================================
# TRAINING FUNCTIONS
# =====================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx+1}/{len(dataloader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


# =====================================================
# MAIN TRAINING LOOP
# =====================================================

def main():
    print("=" * 60)
    print("EMOTION DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Create dataset
    print("\n[1/5] Loading dataset...")
    full_dataset = EmotionDataset(DATA_DIR, transform=train_transform)
    
    # Split into train and validation
    train_dataset, val_dataset = split_dataset(full_dataset, val_split=0.2)
    
    # Apply validation transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\n[2/5] Creating model...")
    model = create_model(NUM_CLASSES)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    best_val_acc = 0.0
    best_model_path = "best_emotion_model.pth"
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'emotion_classes': EMOTION_CLASSES
            }, best_model_path)
            print(f"  *** New best model saved! Accuracy: {val_acc:.2f}% ***")
    
    total_time = time.time() - start_time
    
    # Save final model
    print("\n[4/5] Saving final model...")
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'emotion_classes': EMOTION_CLASSES
    }, "final_emotion_model.pth")
    
    # Print training summary
    print("\n[5/5] Training Summary")
    print("=" * 60)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final training accuracy: {train_accs[-1]:.2f}%")
    print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"Model saved to: {best_model_path}")
    print("=" * 60)
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }
    torch.save(history, "training_history.pth")
    print("\nTraining history saved to training_history.pth")


if __name__ == "__main__":
    main()
