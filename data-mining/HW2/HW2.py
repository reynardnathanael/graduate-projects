# General Packages
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import copy
import random
import warnings

# Model Packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Hyperparameters
IMG_WIDTH = 256         
IMG_HEIGHT = 256    
IMG_CHANNELS = 3
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 20
CAE_EPOCHS = 100   
CAE_LR = 1e-4
CAE_WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT = 0.2 
PATIENCE = 100           
TRAIN_PATH = 'data/train/'
TEST_PATH = 'data/test/'

# Seeding for consistent random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images from the folder
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.file_list = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        self.transform = transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img_path = os.path.join(self.folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, filename
        except Exception as e: print(f"Error loading {img_path}: {e}"); return torch.zeros((IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)), filename
    def get_filenames(self): return self.file_list

# Main pipeline
def main():
    # Transform train data
    train_transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.RandomHorizontalFlip(),
        T.ToTensor(), # Scales images to [0, 1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Transform eval data
    eval_transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_train_dataset_aug = ImageFolderDataset(TRAIN_PATH, transform=train_transform)
    full_train_dataset_noaug = ImageFolderDataset(TRAIN_PATH, transform=eval_transform)
    test_dataset = ImageFolderDataset(TEST_PATH, transform=eval_transform)

    val_size = int(len(full_train_dataset_aug) * VALIDATION_SPLIT)
    train_size = len(full_train_dataset_aug) - val_size
    indices = list(range(len(full_train_dataset_aug)))
    train_indices, val_indices = random_split(indices, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(SEED))

    # Splitting data into validation set
    train_subset_aug = Subset(full_train_dataset_aug, train_indices)
    train_subset_noaug = Subset(full_train_dataset_noaug, train_indices)
    val_subset_noaug = Subset(full_train_dataset_noaug, val_indices)

    # Dataloaders
    train_loader_cae = DataLoader(full_train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_cae = DataLoader(val_subset_noaug, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

    train_loader_extract = DataLoader(train_subset_noaug, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_extract = DataLoader(val_subset_noaug, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
    test_loader_extract = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)

    # ConvAE Architecture
    class ConvAutoencoder(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.latent_dim = latent_dim

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear(512 * 8 * 8, self.latent_dim),
            )

            # Decoder
            self.decoder_fc = nn.Linear(self.latent_dim, 512 * 8 * 8)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            h = self.decoder_fc(z)
            h = h.view(h.size(0), 512, 8, 8)
            return self.decoder(h)

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z)

    # Declare the model
    cae_model = ConvAutoencoder().to(device)
    optimizer = optim.Adam(cae_model.parameters(), lr=CAE_LR, weight_decay=CAE_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Training Stage
    for epoch in range(CAE_EPOCHS):
        cae_model.train()
        train_loss = 0.0
        for (images, _) in tqdm(train_loader_cae, desc=f"CAE Train Epoch {epoch+1}/{CAE_EPOCHS}"):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = cae_model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        cae_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (images, _) in val_loader_cae:
                images = images.to(device)
                outputs = cae_model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader_cae)
        avg_val_loss = val_loss / len(val_loader_cae)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(cae_model.state_dict())
            patience_counter = 0
            print(f"  New best validation loss: {best_val_loss:.6f}. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    cae_model.load_state_dict(best_model_state)
    cae_model.eval()

    # Save the best weights, can be used later without re-training
    cae_save_path = "cae_model_best.pth"
    torch.save(cae_model.state_dict(), cae_save_path)
    print(f"Best CAE model state saved to {cae_save_path}")

    # Features extraction
    def extract_cnn_features(model, data_loader, device):
        model.eval()
        all_features = []
        all_filenames = []
        with torch.no_grad():
            for (images, filenames) in tqdm(data_loader, desc="Extracting Features"):
                images = images.to(device)
                features = model.encode(images)
                all_features.append(features.cpu().numpy())
                all_filenames.extend(filenames)
        return np.concatenate(all_features, axis=0), all_filenames

    train_features_np, _ = extract_cnn_features(cae_model, train_loader_extract, device)
    print(f"Training features shape: {train_features_np.shape}")

    val_features_np, _ = extract_cnn_features(cae_model, val_loader_extract, device)
    print(f"Validation features shape: {val_features_np.shape}")

    test_features_np, test_filenames = extract_cnn_features(cae_model, test_loader_extract, device)
    print(f"Test features shape: {test_features_np.shape}")

    # PCA
    pca = PCA(n_components=256, random_state=SEED)
    train_features_np = pca.fit_transform(train_features_np)
    val_features_np = pca.transform(val_features_np)
    test_features_np = pca.transform(test_features_np)

    # KMeans
    cluster = 8
    kmeans = KMeans(n_clusters=cluster, random_state=SEED, n_init='auto')
    kmeans.fit(train_features_np)
    
    val_labels = kmeans.predict(val_features_np)
    train_labels = kmeans.predict(train_features_np)

    val_distances = kmeans.transform(val_features_np)
    train_distances = kmeans.transform(train_features_np)

    cluster_distances = {i: [] for i in range(cluster)}
    for i in range(len(train_features_np)):
        label = train_labels[i]
        distance = train_distances[i, label]
        cluster_distances[label].append(distance)

    for i in range(len(val_features_np)):
        label = val_labels[i]
        distance = val_distances[i, label]
        cluster_distances[label].append(distance)

    test_labels = kmeans.predict(test_features_np)
    test_distances = kmeans.transform(test_features_np)

    threshold_radii = {}
    for cluster_id, distances in cluster_distances.items():
        if not distances:
            threshold_radii[cluster_id] = 0.0
        else:
            threshold_radii[cluster_id] = np.percentile(distances, 30)

    predictions = []
    for i in range(len(test_features_np)):
        label = test_labels[i]
        distance = test_distances[i, label]
        threshold = threshold_radii[label]
        if distance > threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    
    submission_ids = [int(os.path.splitext(f)[0]) for f in test_filenames]
    sub_df = pd.DataFrame({'id': submission_ids, 'prediction': predictions})
    sub_df = sub_df.sort_values(by='id').reset_index(drop=True)
    
    submission_file_name = f'submission.csv'
    sub_df.to_csv(submission_file_name, index=False)
    print(f"Saved '{submission_file_name}'")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()