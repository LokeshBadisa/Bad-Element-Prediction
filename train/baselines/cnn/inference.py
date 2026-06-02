"""
Run inference with the best checkpoint and save wrong predictions to checkers/.
Each wrong prediction gets its own sub-folder with context.png, crop.png, and
a shared results.json containing actual/predicted labels and confidence.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset import ShareGPTBBoxDataset
from model import TwoBranchCNN

CONTEXT_SIZE = 224
CROP_SIZE = 96
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- load model ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # try alternate extension
        alt = ckpt_path.with_suffix('.pt' if ckpt_path.suffix == '.pth' else '.pth')
        if alt.exists():
            ckpt_path = alt
        else:
            raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    model = TwoBranchCNN(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f'Loaded checkpoint: {ckpt_path}')

    # ---- transforms (model input only, not for saving) ----
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

    # ---- dataset (raw paths + labels) ----
    raw_ds = ShareGPTBBoxDataset(args.test_json, transform=None)
    print(f'Test samples: {len(raw_ds)}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    wrong_count = 0

    with torch.no_grad():
        for idx in tqdm(range(len(raw_ds)), desc='Inference'):
            context_path, crop_path, label = raw_ds.samples[idx]

            context_img = Image.open(context_path).convert('RGB')
            crop_img = Image.open(crop_path).convert('RGB')

            ctx_tensor = context_tf(context_img).unsqueeze(0).to(device)
            crop_tensor = crop_tf(crop_img).unsqueeze(0).to(device)

            logit = model(ctx_tensor, crop_tensor)
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0

            if pred != label:
                sample_dir = out_dir / str(wrong_count)
                sample_dir.mkdir(exist_ok=True)

                # save originals (not the resized/normalised tensors)
                context_img.save(sample_dir / 'context.png')
                crop_img.save(sample_dir / 'crop.png')

                full_labels = {
                    str(k): v
                    for k, v in raw_ds.context_label_maps.get(context_path, {}).items()
                }
                results.append({
                    'id': wrong_count,
                    'actual': label,
                    'predicted': pred,
                    'confidence': round(prob, 4),
                    'context_src': str(context_path),
                    'crop_src': str(crop_path),
                    'full_labels': full_labels,
                })
                wrong_count += 1

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    total = len(raw_ds)
    fp = sum(1 for r in results if r['actual'] == 0 and r['predicted'] == 1)
    fn = sum(1 for r in results if r['actual'] == 1 and r['predicted'] == 0)
    print(f'\nWrong: {wrong_count}/{total}  |  FP: {fp}  FN: {fn}')
    print(f'Saved to {out_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--test_json',  default=str(Path(__file__).parent / '../../data_gen/testset_sharegpt_data_may23.json'))
    parser.add_argument('--output_dir', default='checkers')
    main(parser.parse_args())
