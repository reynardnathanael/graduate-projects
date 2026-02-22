import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Constant variables for training
EPOCHS = 15
LEARNING_RATE = 1e-4

# Preparing Dataset
class NewsDataset(Dataset):
    def __init__(self, x_data, y_data=None):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        user_vec, news_vec = self.x[idx]
        user_t = torch.tensor(user_vec, dtype=torch.float32)
        news_t = torch.tensor(news_vec, dtype=torch.float32)
        
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return (user_t, news_t), label
        else:
            return (user_t, news_t)

# Training the model
class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Binary Cross Entropy Loss
        self.criterion = nn.BCELoss()
        
        # Adam Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # History tracking
        self.history = {'train_loss': [], 'val_loss': []}

    # Train
    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx+1}/{EPOCHS} [Train]", leave=False)
        
        for (user_vec, news_vec), labels in pbar:
            user_vec = user_vec.to(self.device)
            news_vec = news_vec.to(self.device)
            labels = labels.float().to(self.device).view(-1, 1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(user_vec, news_vec)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    # Validate
    def validate(self, epoch_idx):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch_idx+1}/{EPOCHS} [Val]", leave=False)
            for (user_vec, news_vec), labels in pbar:
                user_vec = user_vec.to(self.device)
                news_vec = news_vec.to(self.device)
                labels = labels.float().to(self.device).view(-1, 1)
                
                outputs = self.model(user_vec, news_vec)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def fit(self):
        for epoch in range(EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        return self.model

    def plot_history(self, save_path='loss_history.png'):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)

def generate_submission(model, test_loader, impression_ids, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for (user_vec, news_vec) in tqdm(test_loader):
            user_vec = user_vec.to(device)
            news_vec = news_vec.to(device)
            
            # Get probability score
            outputs = model(user_vec, news_vec)
            predictions.extend(outputs.cpu().numpy().flatten())
            
    grouped_scores = defaultdict(list)
    for imp_id, score in zip(impression_ids, predictions):
        grouped_scores[imp_id].append(score)
        
    submission_rows = []
    for imp_id, scores in grouped_scores.items():
        assert len(scores) == 15, f"Impression {imp_id} has {len(scores)} items (expected 15)"
        row = [imp_id] + scores
        submission_rows.append(row)
        
    cols = ['id'] + [f'p{i}' for i in range(1, 16)]
    df_sub = pd.DataFrame(submission_rows, columns=cols)
    
    filename = "submission.csv"
    df_sub.to_csv(filename, index=False)