import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Set random seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# Configuration
class CFG:
    img_size = 384
    epochs = 4
    folds = 10
    train_bs = 16
    valid_bs = 32
    lr = 1e-4
    weight_decay = 1e-5
    model_name = 'swin_large_patch4_window12_384'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0  # Set to 0 for Windows compatibility
    output_dir = './saved_models'

# Dataset Class
class PawpularityDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row['filepath']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        label = torch.tensor([row['Pawpularity'] / 100], dtype=torch.float32)
        return image, label

# Data Augmentation
def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

# Model Definition
class PawpularityModel(nn.Module):
    def __init__(self):
        super(PawpularityModel, self).__init__()
        self.model = timm.create_model(
            CFG.model_name,
            pretrained=True,
            num_classes=1,
            global_pool='avg'  # ensures [B, 1] output
        )

    def forward(self, x):
        return self.model(x)

# Training Function
def train_fn(model, dataloader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader):
        images = images.to(CFG.device)
        labels = labels.to(CFG.device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Validation Function
def valid_fn(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    preds = []
    true_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(CFG.device)
            labels = labels.to(CFG.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds.extend(outputs.sigmoid().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    rmse = mean_squared_error(true_labels, preds, squared=False)
    return epoch_loss, rmse

# Test Dataset (no labels)
class TestDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row['filepath']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        return image

# Main Training Loop
def main():
    # Load Data
    df = pd.read_csv('train.csv')
    df['filepath'] = df['Id'].apply(lambda x: os.path.join('train', f'{x}.jpg'))

    # Stratified K-Fold
    df['norm_score'] = df['Pawpularity'] / 100
    df['bins'] = pd.cut(df['norm_score'], bins=10, labels=False)
    skf = StratifiedKFold(n_splits=CFG.folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['bins'])):
        print(f'Fold {fold + 1}')
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = PawpularityDataset(train_df, transforms=get_transforms('train'))
        val_dataset = PawpularityDataset(val_df, transforms=get_transforms('valid'))

        train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, shuffle=True, num_workers=CFG.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

        model = PawpularityModel().to(CFG.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * CFG.epochs)

        best_rmse = np.inf
        for epoch in range(CFG.epochs):
            print(f'Epoch {epoch + 1}/{CFG.epochs}')
            train_loss = train_fn(model, train_loader, criterion, optimizer, scheduler)
            val_loss, val_rmse = valid_fn(model, val_loader, criterion)
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}')
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                os.makedirs(CFG.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(CFG.output_dir, f'model_fold{fold + 1}.pth'))
                print('Model saved.')

# Test Prediction & CSV Submission
def predict_test():
    test_df = pd.read_csv('test.csv')
    test_df['filepath'] = test_df['Id'].apply(lambda x: os.path.join('test', f'{x}.jpg'))
    test_dataset = TestDataset(test_df, transforms=get_transforms('valid'))
    test_loader = DataLoader(test_dataset, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

    predictions = np.zeros(len(test_df))

    for fold in range(CFG.folds):
        print(f'Predicting with fold {fold + 1} model...')
        model = PawpularityModel().to(CFG.device)
        model.load_state_dict(torch.load(os.path.join(CFG.output_dir, f'model_fold{fold + 1}.pth')))
        model.eval()

        preds = []
        with torch.no_grad():
            for images in tqdm(test_loader):
                images = images.to(CFG.device)
                outputs = model(images)
                preds.extend(outputs.sigmoid().cpu().numpy().squeeze())

        predictions += np.array(preds)

    predictions /= CFG.folds
    predictions *= 100  # Rescale to original range

    submission = pd.read_csv('sample_submission.csv')
    submission['Pawpularity'] = predictions
    submission.to_csv('large.csv', index=False)
    print('submission.csv saved.')

if __name__ == '__main__':
    main()
    predict_test()