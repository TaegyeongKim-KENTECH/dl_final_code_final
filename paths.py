from pathlib import Path

ROOT = Path(__file__).resolve().parent

TRAINSET = ROOT / "trainset"
VALSET = ROOT / "valset"
TESTSET = ROOT / "testset"
CLIP_WEIGHTS = ROOT / "weights" / "open_clip_pytorch_model.bin"
MODEL_SAVE_DIR = ROOT / "weights" / "model_save"
