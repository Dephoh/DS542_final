import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm.auto import tqdm
import argparse

class FlatDirectoryDataset(Dataset):
    """For loading images"""
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, 0

class SimCLRTransforms:
    """Data augmentation module for SimCLR"""
    def __init__(self, size=224):
        kernel_size = int(0.1 * size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=kernel_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

class EncoderResNet(nn.Module):
    """ResNet encoder with projection head"""
    def __init__(self, output_dim=128):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze()
        z = self.projector(h)
        return F.normalize(z, dim=1)

class SimCLR(nn.Module):
    """SimCLR model with encoder and NT-Xent loss"""
    def __init__(self, encoder, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def forward(self, x_i, x_j):
        z_i = self.encoder(x_i)
        z_j = self.encoder(x_j)
        
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),representations.unsqueeze(0),dim=2)
        batch_size = x_i.size(0)
        temperature = self.temperature
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(x_i.device)
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(x_i.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(x_i.device)

        logits = logits / temperature

        loss = F.cross_entropy(logits, labels)
        
        return loss

def train_simclr(model, train_loader, optimizer, device, epoch):
    """Training schedule for SimCLR"""
    model.train()
    total_loss = 0
    
    # progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', unit='batch', leave=False)

    for batch_idx, (images, _) in enumerate(pbar):
        x_i = images[0].to(device)
        x_j = images[1].to(device)
        
        optimizer.zero_grad()
        loss = model(x_i, x_j)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(train_loader)

# Aadd arguments for the file run in the shell script
def parse_args():
    parser = argparse.ArgumentParser(description='Train SimCLR on a directory of images')
    parser.add_argument('--data-dir', type=str, required=True, help='jpg directory')
    parser.add_argument('--batch-size', type=int, default=256, help='default: 256')
    parser.add_argument('--epochs', type=int, default=100, help='default: 100')
    parser.add_argument('--lr', type=float, default=3e-4, help='default: 0.0003')
    parser.add_argument('--output-dim', type=int, default=128, help='default: 128')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for NT-Xent loss - default: 0.5')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers  - default: 4')
    parser.add_argument('--checkpoint-freq', type=int, default=10, help='default: 10')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = SimCLRTransforms(size=224)
    dataset = FlatDirectoryDataset(directory=args.data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, 
                              pin_memory=True if torch.cuda.is_available() else False)

    # Initialize model
    encoder = EncoderResNet(output_dim=args.output_dim).to(device)

    model = SimCLR(encoder, temperature=args.temperature).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Epoch progress bar
    epoch_pbar = tqdm(range(args.epochs), desc='Training Progress', unit='epoch')

    # Training loop
    for epoch in epoch_pbar:
        train_loss = train_simclr(model, train_loader, optimizer, device, epoch)
        epoch_pbar.set_postfix({'avg_loss': f'{train_loss:.4f}'})
        #save
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(f'./simclr_encoder_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            checkpoint_path = os.path.join(f'./simclr_model_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"\nCheckpoint: {checkpoint_path}")

    # save
    model_path = os.path.join('./simclr_model_final.pth')
    encoder_path = os.path.join('./simclr_encoder_final.pth')
    torch.save(model.state_dict(), model_path)
    torch.save(model.encoder.state_dict(), encoder_path)
    print("\nTraining completed, models saved")



if __name__ == "__main__":

    main()
