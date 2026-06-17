import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paths import CLIP_WEIGHTS, MODEL_SAVE_DIR, TESTSET
from clipfordetectiondata.datasets import TestDataset1
from models.clipnet import OpenClipLinear

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    labels = []
    probabilities = []
    total_loss = 0
    folder_accuracies = {}
    folder_probabilities = {}
    folder_labels = {}
    folder_predictions = {}

    with torch.no_grad():
        for batch_idx, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc='Evaluating'
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = torch.nn.BCEWithLogitsLoss()(outputs, targets.float())
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

            for i, folder_name in enumerate(folder_names):
                if folder_name not in folder_accuracies:
                    folder_accuracies[folder_name] = {
                        'correct_0': 0, 'total_0': 0, 'correct_1': 0, 'total_1': 0,
                    }
                if targets[i].item() == 0:
                    folder_accuracies[folder_name]['total_0'] += 1
                    if predicted[i].item() == 0:
                        folder_accuracies[folder_name]['correct_0'] += 1
                else:
                    folder_accuracies[folder_name]['total_1'] += 1
                    if predicted[i].item() == 1:
                        folder_accuracies[folder_name]['correct_1'] += 1

            batch_probabilities = torch.sigmoid(outputs).cpu().numpy()
            probabilities.extend(batch_probabilities)

            for i, folder_name in enumerate(folder_names):
                if folder_name not in folder_probabilities:
                    folder_probabilities[folder_name] = []
                    folder_labels[folder_name] = []
                    folder_predictions[folder_name] = []
                folder_probabilities[folder_name].append(batch_probabilities[i])
                folder_labels[folder_name].append(targets[i].item())
                folder_predictions[folder_name].append(predicted[i].item())

    accuracy = accuracy_score(labels, predictions)
    average_precision = average_precision_score(labels, probabilities)
    loss = total_loss / len(dataloader)

    folder_aps = {}
    folder_total_accuracies = {}
    for folder_name in folder_probabilities:
        folder_probs = folder_probabilities[folder_name]
        folder_labs = folder_labels[folder_name]
        folder_preds = folder_predictions[folder_name]

        try:
            ap = average_precision_score(folder_labs, folder_probs)
        except ValueError:
            ap = 0

        total_correct = sum(1 for lab, pred in zip(folder_labs, folder_preds) if lab == pred)
        total_samples = len(folder_labs)
        total_accuracy = total_correct / total_samples if total_samples > 0 else 0

        folder_aps[folder_name] = ap
        folder_total_accuracies[folder_name] = total_accuracy

    return accuracy, folder_accuracies, loss, average_precision, folder_aps, folder_total_accuracies


def test_model(model, dataloader, device):
    accuracy, folder_accuracies, loss, average_precision, folder_aps, folder_total_accuracies = (
        evaluate_model(model, dataloader, device)
    )
    print(f'Overall Test Loss: {loss:.4f}')
    print(f'Overall Test Accuracy: {accuracy:.4f}')
    print(f'Overall Test Average Precision: {average_precision:.4f}')

    for folder_name, accuracies in folder_accuracies.items():
        accuracy_0 = accuracies['correct_0'] / accuracies['total_0'] if accuracies['total_0'] > 0 else 0
        accuracy_1 = accuracies['correct_1'] / accuracies['total_1'] if accuracies['total_1'] > 0 else 0
        ap = folder_aps.get(folder_name, 0)
        total_accuracy = folder_total_accuracies.get(folder_name, 0)
        print(
            f'Folder: {folder_name}, Accuracy for 0: {accuracy_0:.4f}, '
            f'Accuracy for 1: {accuracy_1:.4f}, Total Accuracy: {total_accuracy:.4f}, AP: {ap:.4f}'
        )


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_save_path = MODEL_SAVE_DIR / "model_epoch_1_20250417-211519.pth"
    model = OpenClipLinear(
        normalize=True,
        next_to_last=False,
        pretrained_model_path=str(CLIP_WEIGHTS),
        freeze_clip=True,
    )
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    print(f"Using device: {device}")

    test_dataset = TestDataset1(
        is_train=False,
        args={'data_path': str(TESTSET), 'eval_data_path': str(TESTSET)},
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_model(model, test_dataloader, device)
