"""
Emotion Detection Model Training - Resume Training Script
Continues training from existing checkpoint for better accuracy
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
import json

# =====================================================
# CONFIGURATION - Adjust these settings as needed
# =====================================================

# Dataset path
DATA_DIR = "EDFF_dataset"

# Training settings
NUM_EPOCHS = 50  # Additional epochs to train
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Lower LR for fine-tuning existing model
IMAGE_SIZE = 224

# Resume from checkpoint
CHECKPOINT_PATH = "best_emotion_model.pth"
RESUME_TRAINING = True  # Set to True to resume from checkpoint

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Emotion classes
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
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Data transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def split_dataset(dataset, val_split=0.15):
    """Split dataset into training and validation sets"""
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    return train_dataset, val_dataset


# =====================================================
# MODEL DEFINITION - EfficientNet-B0
# =====================================================

def create_model(num_classes):
    """Create emotion detection model using EfficientNet-B0"""
    
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    except:
        model = models.efficientnet_b0(weights=None)
    
    num_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint if exists"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, 0.0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded successfully!")
        print(f"  Resuming from epoch: {start_epoch}")
        print(f"  Previous best accuracy: {best_val_acc:.2f}%")
        
        return start_epoch, best_val_acc
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Try loading just the weights
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint)
            print("Model weights loaded (without optimizer state)")
            return 0, 0.0
        except Exception as e2:
            print(f"Could not load any checkpoint data: {e2}")
            return 0, 0.0


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
        
        if (batch_idx + 1) % 20 == 0:
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
    print("=" * 70)
    print("EMOTION DETECTION MODEL - RESUME TRAINING")
    print("=" * 70)
    
    # Create dataset
    print("\n[1/6] Loading dataset...")
    full_dataset = EmotionDataset(DATA_DIR, transform=train_transform)
    
    # Split into train and validation
    train_dataset, val_dataset = split_dataset(full_dataset, val_split=0.15)
    
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
    print("\n[2/6] Creating model...")
    model = create_model(NUM_CLASSES)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Load checkpoint if exists
    start_epoch = 0
    best_val_acc = 0.0
    
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print("\n[3/6] Loading checkpoint...")
        start_epoch, best_val_acc = load_checkpoint(model, optimizer, CHECKPOINT_PATH)
    else:
        print("\n[3/6] Starting fresh training...")
    
    # Training loop
    print(f"\n[4/6] Starting training for {NUM_EPOCHS} epochs...")
    print("=" * 70)
    
    best_model_path = "best_emotion_model.pth"
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        current_epoch = start_epoch + epoch
        epoch_start = time.time()
        print(f"\nEpoch [{current_epoch+1}/{start_epoch + NUM_EPOCHS}]")
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
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'emotion_classes': EMOTION_CLASSES,
                'learning_rate': current_lr
            }, best_model_path)
            print(f"  *** New best model saved! Accuracy: {val_acc:.2f}% ***")
        
        # Check if target accuracy achieved
        if val_acc >= 85.0:
            print(f"\n  🎯 Target accuracy of 85% achieved! ({val_acc:.2f}%)")
    
    total_time = time.time() - start_time
    
    # Save final model
    print("\n[5/6] Saving final model...")
    torch.save({
        'epoch': start_epoch + NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'emotion_classes': EMOTION_CLASSES
    }, "final_emotion_model.pth")
    
    # Print training summary
    print("\n[6/6] Training Summary")
    print("=" * 70)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Total epochs trained: {NUM_EPOCHS}")
    print(f"Starting epoch: {start_epoch}")
    print(f"Ending epoch: {start_epoch + NUM_EPOCHS}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final training accuracy: {train_accs[-1]:.2f}%")
    print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"Model saved to: {best_model_path}")
    
    if best_val_acc >= 95.0:
        print("\n🎉 EXCELLENT! Target of 95% accuracy achieved!")
    elif best_val_acc >= 85.0:
        print("\n🎯 Target of 85% accuracy achieved!")
    else:
        print(f"\n⚠️ Target not fully met. Current best: {best_val_acc:.2f}%")
    
    print("=" * 70)
    
    # Save training history
    print("\nSaving training history...")
    history = {
        'train_loss': [float(x) for x in train_losses],
        'train_acc': [float(x) for x in train_accs],
        'val_loss': [float(x) for x in val_losses],
        'val_acc': [float(x) for x in val_accs],
        'emotion_classes': EMOTION_CLASSES,
        'total_epochs': start_epoch + NUM_EPOCHS,
        'best_accuracy': best_val_acc
    }
    
    with open("training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("Training history saved to training_history.json")
    print("\n✅ Resume training complete!")


if __name__ == "__main__":
    main()

