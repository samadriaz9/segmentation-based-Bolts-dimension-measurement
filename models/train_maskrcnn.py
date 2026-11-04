"""Fine-tune Mask R-CNN on the bolt photographs.

Ultralytics YOLO and Roboflow models are not used. The assessment forbids both.
Mask R-CNN with a COCO-pretrained ResNet-50 FPN backbone is a standard instance
segmentation model, and the new head is trained on these images.

The test list is not read until the checkpoint and the score threshold have
been chosen on the validation list.
"""

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
WEIGHTS_PATH = MODEL_DIR / "maskrcnn_bolt.pth"
CONFIG_PATH = MODEL_DIR / "config.json"
METRICS_PATH = MODEL_DIR / "test_metrics.json"
LOG_PATH = MODEL_DIR / "train_log.json"

SEED = 42
MAX_EPOCHS = 24
MIN_EPOCHS = 10
PATIENCE = 6
BATCH_SIZE = 2
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE = 8
GAMMA = 0.1
MASK_THRESHOLD = 0.5
NUM_CLASSES = 2
CLASS_NAME = "Bolt"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BoltDataset(Dataset):
    def __init__(self, list_path, train):
        self.train = train
        self.samples = []
        for line in Path(list_path).read_text(encoding="utf-8").splitlines():
            relative = line.strip()
            if relative:
                image_path = DATASET / relative
                label_path = DATASET / "labels" / f"{image_path.stem}.txt"
                self.samples.append((image_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f"unreadable {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            coords = np.array([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= width
            coords[:, 1] *= height
            cv2.fillPoly(mask, [np.round(coords).astype(np.int32)], 1)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise RuntimeError(f"empty mask {label_path}")
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            x1, y1, x2, y2 = box
            box = np.array([width - x2, y1, width - x1, y2], dtype=np.float32)
        if self.train:
            scale = random.uniform(0.85, 1.15)
            image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.tensor(box, dtype=torch.float32).unsqueeze(0),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.from_numpy(mask.astype(np.uint8)).unsqueeze(0),
            "image_id": torch.tensor([index]),
        }
        return tensor, target, image_path.name


def collate(batch):
    images, targets, names = zip(*batch)
    return list(images), list(targets), list(names)


def build_model():
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, NUM_CLASSES)
    model.transform.min_size = (720,)
    model.transform.max_size = 1280
    return model


def move_targets(targets, device):
    moved = []
    for target in targets:
        moved.append({key: value.to(device) for key, value in target.items()})
    return moved


def mask_iou(prediction, ground_truth):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)
    union = np.logical_or(prediction, ground_truth).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(prediction, ground_truth).sum() / union)


def predict_split(model, loader, device):
    model.eval()
    records = []
    with torch.no_grad():
        for images, targets, names in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for output, target, name in zip(outputs, targets, names):
                ground_truth = target["masks"][0].numpy().astype(bool)
                scores = output["scores"].detach().cpu().numpy()
                masks = output["masks"].detach().cpu().numpy()
                detections = []
                for score, mask in zip(scores, masks):
                    binary = mask[0] >= MASK_THRESHOLD
                    detections.append({
                        "score": float(score),
                        "iou": mask_iou(binary, ground_truth),
                    })
                records.append({"file": name, "detections": detections})
    return records


def mean_iou_at(records, score_threshold):
    values = []
    for record in records:
        kept = [item for item in record["detections"] if item["score"] >= score_threshold]
        values.append(0.0 if not kept else max(kept, key=lambda item: item["score"])["iou"])
    return float(np.mean(values)) if values else 0.0


def precision_recall(records, score_threshold, iou_threshold=0.5):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for record in records:
        kept = [item for item in record["detections"] if item["score"] >= score_threshold]
        if not kept:
            false_negative += 1
            continue
        kept.sort(key=lambda item: item["score"], reverse=True)
        if kept[0]["iou"] >= iou_threshold:
            true_positive += 1
            false_positive += len(kept) - 1
        else:
            false_positive += len(kept)
            false_negative += 1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return precision, recall


def average_precision(records, iou_threshold):
    detections = []
    for image_index, record in enumerate(records):
        for item in record["detections"]:
            detections.append((item["score"], item["iou"], image_index))
    detections.sort(key=lambda item: item[0], reverse=True)
    matched = set()
    true_positive = []
    false_positive = []
    for _score, iou, image_index in detections:
        if iou >= iou_threshold and image_index not in matched:
            matched.add(image_index)
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)
    if not true_positive:
        return 0.0
    cumulative_true = np.cumsum(true_positive)
    cumulative_false = np.cumsum(false_positive)
    recall = cumulative_true / len(records)
    precision = cumulative_true / (cumulative_true + cumulative_false)
    average = 0.0
    for point in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= point]
        average += float(eligible.max()) if eligible.size else 0.0
    return average / 101.0


def summarize(records, score_threshold):
    precision, recall = precision_recall(records, score_threshold)
    thresholds = np.arange(0.50, 1.00, 0.05)
    aps = [average_precision(records, float(threshold)) for threshold in thresholds]
    per_image = []
    for record in records:
        kept = [item for item in record["detections"] if item["score"] >= score_threshold]
        best = max(kept, key=lambda item: item["score"]) if kept else None
        per_image.append({
            "file": record["file"],
            "score": None if best is None else round(best["score"], 4),
            "iou": 0.0 if best is None else round(best["iou"], 4),
        })
    return {
        "score_threshold": score_threshold,
        "mask_iou": round(mean_iou_at(records, score_threshold), 4),
        "mAP_50": round(aps[0], 4),
        "mAP_50_95": round(float(np.mean(aps)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "images": per_image,
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    seen = 0
    for images, targets, _names in loader:
        images = [image.to(device) for image in images]
        targets = move_targets(targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss { {key: float(value) for key, value in loss_dict.items()} }")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss) * len(images)
        seen += len(images)
    return total / max(seen, 1)


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available. Refusing to train Mask R-CNN on CPU.")
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    train_set = BoltDataset(DATASET / "train.txt", train=True)
    val_set = BoltDataset(DATASET / "val.txt", train=False)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, generator=generator)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate)
    model = build_model().to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    best_iou = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_records = predict_split(model, val_loader, device)
        val_iou = mean_iou_at(val_records, 0.5)
        row = {
            "epoch": epoch,
            "train_loss": round(loss, 4),
            "val_mask_iou_at_0.5": round(val_iou, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch} loss {row['train_loss']:.4f} val_iou {row['val_mask_iou_at_0.5']:.4f} {row['seconds']}s",
            flush=True,
        )
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), WEIGHTS_PATH)
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            print(f"stop at epoch {epoch}; best val iou {best_iou:.4f} at epoch {best_epoch}", flush=True)
            break
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    val_records = predict_split(model, val_loader, device)
    candidates = [round(value, 2) for value in np.arange(0.05, 0.96, 0.05)]
    score_threshold = max(candidates, key=lambda threshold: (mean_iou_at(val_records, threshold), threshold))
    print(f"val score threshold {score_threshold:.2f} iou {mean_iou_at(val_records, score_threshold):.4f}", flush=True)
    test_set = BoltDataset(DATASET / "test.txt", train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate)
    test_records = predict_split(model, test_loader, device)
    metrics = summarize(test_records, score_threshold)
    metrics.update({
        "split": "test",
        "checkpoint_epoch": best_epoch,
        "checkpoint_val_mask_iou_at_0.5": round(best_iou, 4),
        "images_in_split": len(test_set),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "history": history,
    })
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOG_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(
        f"TEST mask_iou {metrics['mask_iou']:.4f} mAP50 {metrics['mAP_50']:.4f} "
        f"mAP50_95 {metrics['mAP_50_95']:.4f} precision {metrics['precision']:.4f} recall {metrics['recall']:.4f}",
        flush=True,
    )
    config = {
        "model": "torchvision Mask R-CNN ResNet-50 FPN",
        "pretrained": "COCO MaskRCNN_ResNet50_FPN_Weights.DEFAULT",
        "why": "The assessment forbids Ultralytics YOLO and Roboflow. This is a COCO-pretrained instance segmentation model, fine-tuned on the bolt training images.",
        "class_id": 1,
        "class_name": CLASS_NAME,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "step_size": STEP_SIZE,
        "gamma": GAMMA,
        "max_epochs": MAX_EPOCHS,
        "checkpoint_rule": "highest validation mean mask IoU at score 0.5",
        "score_threshold_rule": "chosen on validation to maximize mean mask IoU, then frozen",
        "test_split": "scored once after checkpoint and threshold were frozen",
        "weights": str(WEIGHTS_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
