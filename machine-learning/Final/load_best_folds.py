import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import joblib 

# Configuration
class CFG:
    img_size = 384
    test_bs = 64
    model_name = 'swin_large_patch4_window12_384'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = '/kaggle/input/swin-best10'
    n_folds = 10
    n_tta = 4 
    use_xgb = False 
    xgb_model_path = '/kaggle/input/xgb-model/xgb_model.pkl'


# Set seed for reproducibility
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


# Dataset
class TestDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['filepath']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        return image


# Augmentations
def get_test_transforms():
    return A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_tta_transforms():
    return A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])


# Model definition
class PawpularityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            CFG.model_name,
            pretrained=False,
            num_classes=1,
            global_pool='avg'
        )

    def forward(self, x):
        return self.model(x)


# Load test data
print("Loading test data...")
test_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')
test_df['filepath'] = test_df['Id'].apply(lambda x: f"/kaggle/input/petfinder-pawpularity-score/test/{x}.jpg")

# Prepare dataloaders
test_dataset = TestDataset(test_df, transforms=get_test_transforms())
test_loader = DataLoader(test_dataset, batch_size=CFG.test_bs, shuffle=False, num_workers=0)

tta_dataset = TestDataset(test_df, transforms=get_tta_transforms())
tta_loader = DataLoader(tta_dataset, batch_size=CFG.test_bs, shuffle=False, num_workers=0)


# Inference with TTA
def predict_with_tta(model, loaders):
    model.eval()
    final_preds = []
    with torch.no_grad():
        for i in range(CFG.n_tta):
            current_loader = loaders[1] if i > 0 else loaders[0]
            preds = []
            for images in tqdm(current_loader, desc=f"TTA round {i+1}", leave=False):
                images = images.to(CFG.device)
                outputs = model(images)
                preds.extend(outputs.sigmoid().cpu().numpy())
            final_preds.append(np.array(preds).squeeze())
    return np.mean(final_preds, axis=0)


# Main inference loop
print("Starting inference with TTA...")
all_preds = []

for fold in range(1, CFG.n_folds + 1):
    model_path = os.path.join(CFG.model_dir, f'model_fold{fold}.pth')
    print(f"\nFold {fold}: loading model from {model_path}")

    model = PawpularityModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    preds = predict_with_tta(model, loaders=(test_loader, tta_loader))
    all_preds.append(preds)

# Average predictions across folds
image_preds = np.mean(all_preds, axis=0) * 100

# Optional: XGBoost Stacking
if CFG.use_xgb:
    print("Using XGBoost for final stacking")
    xgb_model = joblib.load(CFG.xgb_model_path)
    meta_features = pd.read_csv('/kaggle/input/metadata-features/test_meta.csv')
    meta_features['img_pred'] = image_preds
    final_preds = xgb_model.predict(meta_features)
else:
    final_preds = image_preds

# Save submission
submission = test_df[['Id']].copy()
submission['Pawpularity'] = final_preds
submission.to_csv('submission.csv', index=False)
print("Saved to submission.csv")