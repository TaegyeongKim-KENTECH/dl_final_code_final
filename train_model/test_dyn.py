import csv
import json
import os
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paths import CLIP_WEIGHTS, MODEL_SAVE_DIR, TESTSET
from clipfordetectiondata.datasets import TestDataset1
from models.clipnet_dyn import DynFakeDetector

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, dataloader, device, conf_threshold=None):
    """Evaluate with early-exit tracking; optionally override conf_threshold."""
    if conf_threshold is not None:
        model.conf_threshold = conf_threshold

    model.eval()

    predictions, labels, probabilities = [], [], []
    total_loss    = 0.0
    total_samples = 0
    artifact_only = 0

    folder_accuracies    = {}
    folder_probabilities = {}
    folder_labels        = {}
    folder_predictions   = {}

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Evaluating"
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            artifact_feat = model.artifact_branch.encode(inputs)
            pred_artifact = model.artifact_branch.classifier(artifact_feat)

            prob_art    = torch.sigmoid(pred_artifact)
            confidence = (prob_art - 0.5).abs() * 2

            if (confidence >= model.conf_threshold).all():
                outputs = pred_artifact
                artifact_only += targets.size(0)
            else:
                pred_semantic = model.semantic_branch(inputs)
                gate_logits   = model.gate(artifact_feat)
                weight        = torch.softmax(
                    gate_logits / model.temp, dim=-1
                )
                outputs = (weight[:, 0:1] * pred_artifact
                         + weight[:, 1:2] * pred_semantic)

            outputs = outputs.squeeze()
            loss     = criterion(outputs, targets.float())
            total_loss    += loss.item()
            total_samples += targets.size(0)

            predicted         = (outputs > 0.5).float()
            batch_probs       = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())
            probabilities.extend(batch_probs)

            for i, folder_name in enumerate(folder_names):
                if folder_name not in folder_accuracies:
                    folder_accuracies[folder_name] = {
                        "correct_0": 0, "total_0": 0,
                        "correct_1": 0, "total_1": 0,
                    }
                if targets[i].item() == 0:
                    folder_accuracies[folder_name]["total_0"] += 1
                    if predicted[i].item() == 0:
                        folder_accuracies[folder_name]["correct_0"] += 1
                else:
                    folder_accuracies[folder_name]["total_1"] += 1
                    if predicted[i].item() == 1:
                        folder_accuracies[folder_name]["correct_1"] += 1

                if folder_name not in folder_probabilities:
                    folder_probabilities[folder_name] = []
                    folder_labels[folder_name]        = []
                    folder_predictions[folder_name]   = []
                folder_probabilities[folder_name].append(batch_probs[i])
                folder_labels[folder_name].append(targets[i].item())
                folder_predictions[folder_name].append(predicted[i].item())

    accuracy          = accuracy_score(labels, predictions)
    average_precision = average_precision_score(labels, probabilities)
    avg_loss          = total_loss / len(dataloader)
    semantic_ratio    = 1.0 - artifact_only / max(total_samples, 1)

    folder_aps              = {}
    folder_total_accuracies = {}
    for folder_name in folder_probabilities:
        fp   = folder_probabilities[folder_name]
        fl   = folder_labels[folder_name]
        fprd = folder_predictions[folder_name]
        try:
            ap = average_precision_score(fl, fp)
        except ValueError:
            ap = 0.0
        total_correct  = sum(1 for l, p in zip(fl, fprd) if l == p)
        folder_aps[folder_name]              = ap
        folder_total_accuracies[folder_name] = total_correct / len(fl)

    return (
        accuracy, folder_accuracies, avg_loss,
        average_precision, folder_aps, folder_total_accuracies,
        semantic_ratio,
    )



def test_model(model, dataloader, device, conf_threshold=None):
    """Print test metrics and semantic-branch usage ratio."""
    (
        accuracy, folder_accuracies, loss,
        average_precision, folder_aps, folder_total_accuracies,
        semantic_ratio,
    ) = evaluate_model(model, dataloader, device, conf_threshold)

    print(f"\nOverall Test Loss            : {loss:.4f}")
    print(f"Overall Test Accuracy        : {accuracy:.4f}")
    print(f"Overall Test Average Precision: {average_precision:.4f}")
    print(f"Semantic branch called       : {semantic_ratio * 100:.1f}% of samples")
    print(f"(conf_threshold = {model.conf_threshold})")
    print("-" * 70)

    for folder_name, acc in folder_accuracies.items():
        acc0  = acc["correct_0"] / acc["total_0"] if acc["total_0"] > 0 else 0.0
        acc1  = acc["correct_1"] / acc["total_1"] if acc["total_1"] > 0 else 0.0
        ap    = folder_aps.get(folder_name, 0.0)
        total = folder_total_accuracies.get(folder_name, 0.0)
        print(
            f"Folder: {folder_name:30s} | "
            f"Acc(real)={acc0:.4f}  Acc(fake)={acc1:.4f}  "
            f"Total={total:.4f}  AP={ap:.4f}"
        )



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_save_path = MODEL_SAVE_DIR / "best_model_20260603-074224.pth"

    model = DynFakeDetector(
        pretrained_model_path=str(CLIP_WEIGHTS),
        normalize=True,
        next_to_last=False,
        conf_threshold=0.8,
    )
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)

    test_dataset = TestDataset1(
        is_train=False,
        args={"data_path": str(TESTSET), "eval_data_path": str(TESTSET)},
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_model(model, test_dataloader, device)

    print("\n" + "=" * 70)
    print("Threshold sweep")
    print("=" * 70)

    sweep_results = []

    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        (acc, _, _, ap, _, _, sem_ratio) = evaluate_model(
            model, test_dataloader, device, conf_threshold=thr
        )
        print(
            f"  conf_threshold={thr:.1f} | "
            f"Acc={acc:.4f}  AP={ap:.4f}  "
            f"semantic_called={sem_ratio * 100:.1f}%"
        )
        sweep_results.append({
            "conf_threshold":  thr,
            "accuracy":        round(acc, 6),
            "average_precision": round(ap, 6),
            "semantic_called": round(sem_ratio, 6),
        })

    json_path = MODEL_SAVE_DIR / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved → {json_path}")

    csv_path = MODEL_SAVE_DIR / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["conf_threshold", "accuracy", "average_precision", "semantic_called"]
        )
        writer.writeheader()
        writer.writerows(sweep_results)
    print(f"Sweep results saved → {csv_path}")
