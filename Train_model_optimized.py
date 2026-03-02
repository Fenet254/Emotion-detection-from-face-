"""
Emotion Detection Model Training Script - Optimized Version
Target: 85-95% accuracy using EfficientNet-B0 with advanced techniques
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import time
from collections import Counter
import json

# =====================================================
# CONFIGURATION - Optimized for High Accuracy
# =====================================================

# Dataset path
DATA_DIR = "EDFF_dataset"

# Training settings - Optimized
NUM_EPOCHS = 50  # EfficientNet converges faster
BATCH_SIZE = 32
LEARNING_RATE = 0.0003  # Lower LR for fine-tuning
IMAGE_SIZE = 224
USE_MIXED_PRECISION = True  # Faster training with Tensor Cores

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    """Custom Dataset for Emotion Detection with augmentation"""
    
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_training = is_training
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
        class_counts = Counter(self.labels)
        print(f"Class distribution: {dict(sorted(class_counts.items()))}")
    
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


# Advanced data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # Resize larger
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),  # Random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),  # More rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # Cutout-like augmentation
])

# Validation transform - deterministic
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def create_class_weights(labels):
    """Create class weights for imbalanced dataset"""
    class_counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(NUM_CLASSES):
        count = class_counts.get(i, 1)
        weights.append(total / (NUM_CLASSES * count))
    
    # Normalize weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * NUM_CLASSES
    
    return weights


def split_dataset(dataset, val_split=0.15):
    """Split dataset into training and validation sets"""
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset


# =====================================================
# MODEL DEFINITION - EfficientNet-B0
# =====================================================

def create_model(num_classes, pretrained=True):
    """Create emotion detection model using EfficientNet-B0"""
    
    # Use EfficientNet-B0 (better accuracy than ResNet)
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Get number of features from classifier
    num_features = model.classifier[1].in_features
    
    # Replace classifier with custom head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


# =====================================================
# TRAINING FUNCTIONS
# =====================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
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
    all_predictions = []
    all_labels = []
    
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
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_predictions, all_labels


def calculate_per_class_accuracy(predictions, labels):
    """Calculate accuracy for each class"""
    class_correct = Counter()
    class_total = Counter()
    
    for pred, label in zip(predictions, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    per_class_acc = {}
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            per_class_acc[EMOTION_CLASSES[i]] = 100. * class_correct[i] / class_total[i]
        else:
            per_class_acc[EMOTION_CLASSES[i]] = 0.0
    
    return per_class_acc


# =====================================================
# MAIN TRAINING LOOP
# =====================================================

def main():
    print("=" * 70)
    print("EMOTION DETECTION MODEL TRAINING - OPTIMIZED VERSION")
    print("Target: 85-95% Accuracy")
    print("=" * 70)
    
    # Create dataset
    print("\n[1/6] Loading dataset...")
    full_dataset = EmotionDataset(DATA_DIR, transform=train_transform, is_training=True)
    
    # Get labels for class weights
    all_labels = full_dataset.labels
    
    # Create class weights for imbalanced data
    class_weights = create_class_weights(all_labels).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Split into train and validation
    train_dataset, val_dataset = split_dataset(full_dataset, val_split=0.15)
    
    # Apply validation transform
    val_dataset.dataset.transform = val_transform
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create weighted sampler for balanced training
    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    class_sample_counts = Counter(train_labels)
    weights = [1.0 / class_sample_counts[label] for label in train_labels]
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
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
    print("\n[2/6] Creating EfficientNet-B0 model...")
    model = create_model(NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with different LR for different layers
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': LEARNING_RATE * 0.1},  # Lower LR for pretrained layers
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=0.01)
    
    # Learning rate scheduler - Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and USE_MIXED_PRECISION else None
    
    # Training loop
    print(f"\n[3/6] Starting training for {NUM_EPOCHS} epochs...")
    print("=" * 70)
    
    best_val_acc = 0.0
    best_model_path = "best_emotion_model.pth"
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    start_time = time.time()
    patience = 10  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, predictions, labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        
        # Calculate per-class accuracy
        per_class_acc = calculate_per_class_accuracy(predictions, labels)
        print("\n  Per-class accuracy:")
        for emotion, acc in per_class_acc.items():
            print(f"    {emotion}: {acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'emotion_classes': EMOTION_CLASSES,
                'class_weights': class_weights.cpu().numpy()
            }, best_model_path)
            print(f"\n  *** New best model saved! Accuracy: {val_acc:.2f}% ***")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience and epoch >= 20:
            print(f"\n  Early stopping triggered after {epoch+1} epochs")
            break
        
        # Check if target accuracy achieved
        if val_acc >= 85.0:
            print(f"\n  🎯 Target accuracy of 85% achieved! ({val_acc:.2f}%)")
    
    total_time = time.time() - start_time
    
    # Save final model
    print("\n[4/6] Saving final model...")
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'emotion_classes': EMOTION_CLASSES
    }, "final_emotion_model.pth")
    
    # Print training summary
    print("\n[5/6] Training Summary")
    print("=" * 70)
    print(f"Total training time: {total_time/60:.2f} minutes")
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
    print("\n[6/6] Saving training history...")
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'emotion_classes': EMOTION_CLASSES
    }
    
    with open("training_history.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            'train_loss': [float(x) for x in train_losses],
            'train_acc': [float(x) for x in train_accs],
            'val_loss': [float(x) for x in val_losses],
            'val_acc': [float(x) for x in val_accs],
            'emotion_classes': EMOTION_CLASSES
        }
        json.dump(history_serializable, f, indent=2)
    
    print("Training history saved to training_history.json")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
