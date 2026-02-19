import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


def load_and_filter_data(path="./data/final_df_with_features_expanded.csv", go_min_freq=4):
    """Load CSV and filter rare GO terms (same logic as XGBoost notebook)."""
    df = pd.read_csv(path)

    go_cols = [col for col in df.columns if col.startswith("GO:")]
    go_counts = df[go_cols].sum()
    keep_go = go_counts[go_counts >= go_min_freq].index.tolist()
    drop_go = go_counts[go_counts < go_min_freq].index.tolist()

    print(f"Total GO terms: {len(go_cols)}")
    print(f"Keeping (freq >= {go_min_freq}): {len(keep_go)}")
    print(f"Dropping (freq < {go_min_freq}): {len(drop_go)}")

    df = df.drop(columns=drop_go)

    remaining_go = [col for col in df.columns if col.startswith("GO:")]
    mask = df[remaining_go].sum(axis=1) > 0
    print(f"Rows before: {len(df)}, after removing zero-annotation rows: {mask.sum()}")
    df = df[mask].reset_index(drop=True)

    return df


def prepare_splits(df):
    """Stratified multi-label split: 70% train, 15% val, 15% test."""
    features = [col for col in df.columns if col not in ["sequence"] + [col for col in df.columns if col.startswith("GO:")]]
    go_terms = [col for col in df.columns if col.startswith("GO:")]

    X = df[features].values
    y = df[go_terms].values

    # First split: 70% train, 30% temp
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(msss1.split(X, y))
    X_train, X_temp = X[train_idx], X[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]

    # Second split: 50/50 of temp → 15% val, 15% test
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(msss2.split(X_temp, y_temp))
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, go_terms


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer3(x))
        return x


def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=0.001, save_path="./data/best_dnn_model.pth"):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start

        marker = " *" if val_loss < best_val_loss else ""
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.1f}s{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")
    print(f"Best model saved to {save_path}")
    return best_val_loss


def evaluate_model(model, test_loader, go_terms, device):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            y_pred.append((outputs > 0.5).int().cpu().numpy())
            y_true.append(batch_y.cpu().numpy())

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=go_terms, zero_division=0))


if __name__ == "__main__":
    # Load and filter
    df = load_and_filter_data()

    # Prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test, go_terms = prepare_splits(df)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    # Model
    model = DNN(X_train.shape[1], y_train.shape[1]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    save_path = "./data/best_dnn_model.pth"
    train_model(model, train_loader, val_loader, device, num_epochs=20, save_path=save_path)

    # Load best and evaluate
    model.load_state_dict(torch.load(save_path, weights_only=True))
    evaluate_model(model, test_loader, go_terms, device)
