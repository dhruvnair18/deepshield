import os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# Optional: reduce thread oversubscription on Mac/conda
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DEVICE   = os.getenv("DEVICE","cpu")
BS       = int(os.getenv("BS","32"))
EPOCHS   = int(os.getenv("EPOCHS","5"))
LR       = float(os.getenv("LR","1e-3"))
IMG      = int(os.getenv("IMG","224"))
OUT      = os.getenv("OUT","models/effb0_df.pth")
TRAIN_DIR= os.getenv("TRAIN_DIR","data/train")
VAL_DIR  = os.getenv("VAL_DIR","data/val")

def make_dataloaders():
    tfm_train = transforms.Compose([
    transforms.ToTensor(),                              # move first to avoid PIL hue bug
    transforms.ColorJitter(0.10, 0.10, 0.10, 0.02),     # smaller hue jitter, safe on tensor
    transforms.RandomHorizontalFlip(),
    transforms.Resize((IMG, IMG)),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    ])

    tfm_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG, IMG)),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    ])


    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tfm_train)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=tfm_val)

    # IMPORTANT on macOS: use num_workers=0 to avoid spawn issues
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=0)
    return train_dl, val_dl

def evaluate(model, crit, val_dl):
    model.eval()
    tot=0; correct=0; loss_sum=0.0
    with torch.no_grad():
        for x,y in val_dl:
            x,y = x.to(DEVICE), y.float().to(DEVICE)
            logits = model(x).squeeze(1)
            loss = crit(logits, y)
            prob = torch.sigmoid(logits)
            pred = (prob>=0.5).float()
            correct += (pred==y).sum().item()
            loss_sum += loss.item()*x.size(0)
            tot += x.size(0)
    return (loss_sum/tot) if tot>0 else 0.0, (correct/tot) if tot>0 else 0.0

def main():
    torch.set_num_threads(1)

    train_dl, val_dl = make_dataloaders()

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss()
    opt   = optim.AdamW(model.parameters(), lr=LR)

    best=0.0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    for e in range(1, EPOCHS+1):
        model.train()
        for x,y in train_dl:
            x,y = x.to(DEVICE), y.float().to(DEVICE)
            logits = model(x).squeeze(1)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

        vl, va = evaluate(model, crit, val_dl)
        print(f"Epoch {e}: val_loss={vl:.4f} val_acc={va:.3f}")
        if va > best:
            best = va
            torch.save(model.state_dict(), OUT)
            print(f"Saved {OUT} (acc={va:.3f})")

if __name__ == "__main__":
    main()
