# utils/train_tsr.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

def train_tsr(model, train_loader, test_loader, device, num_epochs=20, lr=0.001, ckpt_dir="results/checkpoints/tsr"):
    os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    patience, patience_counter = 5, 0  # for early stopping

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

        # --- Evaluation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        print(f" Test Accuracy: {test_acc:.2f}%")

        # --- Checkpoint and Early Stopping ---
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            best_path = os.path.join(ckpt_dir, "tsr_best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": best_acc,
            }, best_path)
            print(f"✅ Saved best model: {best_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

    print(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")
    return model, best_acc
