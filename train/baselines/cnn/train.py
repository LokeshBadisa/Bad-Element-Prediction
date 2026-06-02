import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ShareGPTBBoxDataset
from model import TwoBranchCNN

CONTEXT_SIZE = 224
CROP_SIZE    = 96
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms():
    context_tf = transforms.Compose([
        transforms.Resize((CONTEXT_SIZE, CONTEXT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    crop_tf = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return context_tf, crop_tf


class PairedTransformDataset(torch.utils.data.Dataset):
    def __init__(self, base, context_tf, crop_tf):
        self.base = base
        self.context_tf = context_tf
        self.crop_tf = crop_tf

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        return {
            'context': self.context_tf(s['context']),
            'crop':    self.crop_tf(s['crop']),
            'label':   torch.tensor(s['label'], dtype=torch.float32),
        }


def compute_metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return acc, prec, rec, f1


def save_plots(step_losses, epoch_val_losses, epoch_val_f1s, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Step-level training loss
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(step_losses, linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss per step')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'train_loss_steps.png', dpi=150)
    plt.close(fig)

    epochs = list(range(1, len(epoch_val_losses) + 1))

    # Val loss per epoch
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, epoch_val_losses, marker='o', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val loss')
    ax.set_title('Validation loss per epoch')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'val_loss_epochs.png', dpi=150)
    plt.close(fig)

    # Val F1 per epoch
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, epoch_val_f1s, marker='o', linewidth=1.5, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val F1')
    ax.set_title('Validation F1 per epoch')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'val_f1_epochs.png', dpi=150)
    plt.close(fig)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    context_tf, crop_tf = get_transforms()

    raw_train = ShareGPTBBoxDataset(args.train_json)
    raw_test  = ShareGPTBBoxDataset(args.test_json)
    train_ds  = PairedTransformDataset(raw_train, context_tf, crop_tf)
    test_ds   = PairedTransformDataset(raw_test,  context_tf, crop_tf)
    print(f'Train samples: {len(train_ds)}  Test samples: {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = TwoBranchCNN(pretrained=True).to(device)

    train_labels = torch.tensor([s[2] for s in raw_train.samples], dtype=torch.float32)
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)], device=device)
    print(f'pos_weight: {pos_weight.item():.2f}  (neg={int(n_neg)}, pos={int(n_pos)})')

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    step_losses      = []
    epoch_val_losses = []
    epoch_val_f1s    = []
    best_f1 = 0.0
    global_step = 0

    with open(log_path, 'w') as log:
        log.write('# epoch  step  train_loss\n')
        log.flush()

        for epoch in range(1, args.epochs + 1):
            # ---- train ----
            model.train()
            epoch_train_loss = 0.0
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]',
                             leave=False, dynamic_ncols=True)
            for batch in train_bar:
                context = batch['context'].to(device)
                crop    = batch['crop'].to(device)
                labels  = batch['label'].to(device)

                optimizer.zero_grad()
                logits = model(context, crop)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                global_step += 1
                step_losses.append(loss.item())
                epoch_train_loss += loss.item() * len(labels)
                log.write(f'{epoch}\t{global_step}\t{loss.item():.6f}\n')
                log.flush()
                train_bar.set_postfix(loss=f'{loss.item():.4f}')

            epoch_train_loss /= len(train_ds)
            scheduler.step()

            # ---- eval ----
            model.eval()
            val_loss = 0.0
            all_preds, all_true = [], []
            eval_bar = tqdm(test_loader, desc=f'Epoch {epoch}/{args.epochs} [eval] ',
                            leave=False, dynamic_ncols=True)
            with torch.no_grad():
                for batch in eval_bar:
                    context = batch['context'].to(device)
                    crop    = batch['crop'].to(device)
                    labels  = batch['label'].to(device)
                    logits  = model(context, crop)
                    val_loss += criterion(logits, labels).item() * len(labels)
                    preds = (torch.sigmoid(logits) >= 0.5).long()
                    all_preds.append(preds.cpu())
                    all_true.append(labels.long().cpu())

            val_loss /= len(test_ds)
            all_preds = torch.cat(all_preds)
            all_true  = torch.cat(all_true)
            acc, prec, rec, f1 = compute_metrics(all_preds, all_true)

            epoch_val_losses.append(val_loss)
            epoch_val_f1s.append(f1)

            log.write(
                f'# EPOCH {epoch} | train_loss={epoch_train_loss:.6f} '
                f'val_loss={val_loss:.6f} acc={acc:.4f} '
                f'prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}\n'
            )
            log.flush()

            print(f'Epoch {epoch:3d}/{args.epochs} | '
                  f'train_loss={epoch_train_loss:.4f}  val_loss={val_loss:.4f} | '
                  f'acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}')

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), save_path)
                print(f'  -> Saved best model (f1={best_f1:.3f})')

            save_plots(step_losses, epoch_val_losses, epoch_val_f1s, args.plots_dir)

    print(f'\nTraining complete. Best val F1: {best_f1:.3f}')
    print(f'Log: {log_path}  Plots: {args.plots_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json',  default='../../data_gen/trainset_sharegpt_data_may23.json')
    parser.add_argument('--test_json',   default='../../data_gen/testset_sharegpt_data_may23.json')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--save_path',   default='checkpoints/best_model.pth')
    parser.add_argument('--plots_dir',   default='plots')
    parser.add_argument('--log_path',    default='logs/training.txt')
    args = parser.parse_args()
    train(args)
