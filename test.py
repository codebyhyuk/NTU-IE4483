import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

from models.resnet import ResNet50 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="./checkpoints/resnet50_scratch.pth",
                   help="path to model checkpoint (.pth)")
    p.add_argument("--test_dir", type=str, required=True,
                   help="directory containing test images (flat folder of images)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_csv", type=str, default="submission.csv")
    p.add_argument("--num_classes", type=int, default=2,
                   help="number of classes for the loaded head (e.g., 2 for dog/cat)")
    p.add_argument("--use_gpu", action="store_true",
                   help="set to use CUDA if available")
    p.add_argument("--return_prob", action="store_true",
                   help="also output predicted probability of the chosen class")
    return p.parse_args()

class TestDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder

        self.files = sorted([f for f in os.listdir(folder)
                             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.folder, fname)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        img_id = os.path.splitext(fname)[0]
        return img, img_id

def main():
    args = parse_args()
    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_ds = TestDataset(args.test_dir, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── Model & checkpoint load ────────────────────────────────────────────────
    model = ResNet50(num_classes=args.num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    rows = []
    with torch.no_grad():
        for imgs, ids in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()

            if args.return_prob:
                max_probs = probs.max(dim=1)[0].cpu().numpy()
                for img_id, pred, p in zip(ids, preds, max_probs):
                    rows.append({"id": img_id, "label": int(pred), "prob": float(p)})
            else:
                for img_id, pred in zip(ids, preds):
                    rows.append({"id": img_id, "label": int(pred)})

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
