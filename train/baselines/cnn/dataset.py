import json
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def _parse_labels(content):
    """Parse label dict from ```json{0:benign, 1:malicious}``` (unquoted keys/values)."""
    match = re.search(r'\{([^}]+)\}', content)
    if not match:
        return {}
    labels = {}
    for item in match.group(1).split(','):
        item = item.strip()
        if ':' not in item:
            continue
        k, v = item.split(':', 1)
        labels[int(k.strip())] = v.strip().lower()
    return labels
    


class ShareGPTBBoxDataset(Dataset):
    """
    Yields one sample per (context image, crop image, label) pair.
    Labels: malicious=1, benign=0.
    """

    def __init__(self, json_path, transform=None):
        self.transform = transform
        self.samples = []  # list of (context_path, crop_path, label)
        self.context_label_maps = {}  # context_path -> {box_num: label_str}

        with open(json_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            images = entry['images']
            context_path = images[0]
            gpt_content = entry['messages'][-1]['content']
            label_map = _parse_labels(gpt_content)
            self.context_label_maps[context_path] = label_map

            for crop_path in images[1:]:
                box_num = int(Path(crop_path).stem)
                label_str = label_map.get(box_num)
                if label_str is None:
                    continue
                label = 1 if label_str == 'malicious' else 0
                self.samples.append((context_path, crop_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context_path, crop_path, label = self.samples[idx]
        context_img = Image.open(context_path).convert('RGB')
        crop_img = Image.open(crop_path).convert('RGB')
        if self.transform:
            context_img = self.transform(context_img)
            crop_img = self.transform(crop_img)
        return {'context': context_img, 'crop': crop_img, 'label': label}


