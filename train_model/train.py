import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

from clipforfakedetection.clipfordetectiondata.datasets import TrainDataset, TestDataset
from clipforfakedetection.models.clipnet import OpenClipLinear

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluate_model(model, dataloader, device, writer, prefix, epoch):
    model.eval()
    predictions = []
    labels = []
    total_loss = 0
    folder_stats = defaultdict(lambda: {'predictions': [], 'labels': []})

    with torch.no_grad():
        for batch_idx, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f'Evaluating Epoch {epoch + 1}',
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

            for folder_name, pred, label in zip(
                folder_names, predicted.cpu().numpy(), targets.cpu().numpy()
            ):
                folder_stats[folder_name]['predictions'].append(pred)
                folder_stats[folder_name]['labels'].append(label)

    ac = accuracy_score(labels, predictions)
    loss = total_loss / len(dataloader)

    writer.add_scalar(f'{prefix}/loss', loss, epoch)
    writer.add_scalar(f'{prefix}/accuracy', ac, epoch)

    for folder_name, stats in folder_stats.items():
        folder_predictions = stats['predictions']
        folder_labels = stats['labels']
        folder_ac = accuracy_score(folder_labels, folder_predictions)

        folder_0_predictions = [p for p, l in zip(folder_predictions, folder_labels) if l == 0]
        folder_0_labels = [0] * len(folder_0_predictions)
        folder_0_ac = accuracy_score(folder_0_labels, folder_0_predictions) if folder_0_labels else 0

        folder_1_predictions = [p for p, l in zip(folder_predictions, folder_labels) if l == 1]
        folder_1_labels = [1] * len(folder_1_predictions)
        folder_1_ac = accuracy_score(folder_1_labels, folder_1_predictions) if folder_1_labels else 0

        print(
            f"Folder: {folder_name}, Accuracy: {folder_ac:.4f}, "
            f"0 Accuracy: {folder_0_ac:.4f}, 1 Accuracy: {folder_1_ac:.4f}"
        )

    return ac, -ac


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss


def train_model(model, train_dataloader, test_dataloader, epochs, device, save_path):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,
        weight_decay=0,
    )

    best_ac = 0
    best_model_state = None
    early_stopping = EarlyStopping(patience=5, verbose=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(save_path, f"train_log{current_time}")
    eval_log_dir = os.path.join(save_path, f"eval_log{current_time}")
    train_writer = SummaryWriter(train_log_dir)
    eval_writer = SummaryWriter(eval_log_dir)

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_dataloader)

        for batch_idx, (data, target) in tqdm(
            enumerate(train_dataloader),
            total=total_batches,
            desc=f'Epoch {epoch + 1}/{epochs}',
        ):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / total_batches
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')
        train_writer.add_scalar('training_loss', epoch_loss, epoch)

        ac, val_loss = evaluate_model(
            model, test_dataloader, device, eval_writer, 'validation', epoch
        )
        print(f'Epoch {epoch + 1}, Test AC: {ac:.4f}')

        model_save_path = os.path.join(save_path, f"model_epoch_{epoch + 1}_{current_time}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model for epoch {epoch + 1} to {model_save_path}")

        if ac > best_ac:
            best_ac = ac
            best_model_state = model.state_dict()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Finished Training')
    train_writer.close()
    eval_writer.close()

    if best_model_state:
        model.load_state_dict(best_model_state)
        best_model_save_path = os.path.join(save_path, f"best_model_{current_time}.pth")
        torch.save(model.state_dict(), best_model_save_path)
        print(f'Best Test AC: {best_ac:.4f}, saved to {best_model_save_path}')
    else:
        print('No better model found.')


if __name__ == '__main__':
    train_dataset = TrainDataset(
        is_train=True,
        args={'data_path': '/home/work/ktg0829/final_project/Dual-Path-AI-Generated-Image-Detection/trainset/'},
    )
    test_dataset = TestDataset(
        is_train=False,
        args={
            'data_path': '/home/work/ktg0829/final_project/Dual-Path-AI-Generated-Image-Detection/valset/',
            'eval_data_path': '/home/work/ktg0829/final_project/Dual-Path-AI-Generated-Image-Detection/valset/',
        },
    )

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    pretrained_weights_path = '../weights/open_clip_pytorch_model.bin'

    try:
        model = OpenClipLinear(
            normalize=True,
            next_to_last=False,
            pretrained_model_path=pretrained_weights_path,
            freeze_clip=True,
        )
        print("Loaded CLIP model: ViT-L-14")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        raise

    save_path = '../weights/model_save'
    train_model(model, train_dataloader, test_dataloader, epochs=5, device=device, save_path=save_path)
