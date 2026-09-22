import os, random
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
from .adult import DataBundle


NORMALIZE = transforms.Normalize([.485, .456, .406], [.229, .224, .225])

def _transforms(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), NORMALIZE
    ]), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), NORMALIZE]
    )

def _split_indices(samples, seed):
    rng = random.Random(seed) 
    train = []
    valid = []

    for label in sorted({label for _, label in samples}):
        group = [i for i, (_, y) in enumerate(samples) if y == label]
        rng.shuffle(group)

        cut = round(.9 * len(group))
        train += group[:cut]
        valid += group[cut:]

    return sorted(train),sorted(valid)


class TinyImageNetVal(Dataset):
    def __init__(self, root, class_to_idx, transform):
        self.transform = transform
        self.samples = []

        root_path = Path(root)

        for line in open(root_path/"val_annotations.txt"):
            image, label, *_ = line.rstrip().split("\t")
            self.samples.append((root_path/"images"/image, class_to_idx[label]))

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self,index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), label

    
def get_imagefolder(seed, train_root, test_root, batch_size, image_size=224, num_workers=4, tiny_val=False):
    train_tf, eval_tf = _transforms(image_size)
    raw = datasets.ImageFolder(train_root)

    train = datasets.ImageFolder(train_root, train_tf)
    valid = datasets.ImageFolder(train_root, eval_tf)

    tr, va = _split_indices(raw.samples, seed)

    test = TinyImageNetVal(test_root, raw.class_to_idx,eval_tf) if tiny_val else datasets.ImageFolder(test_root, eval_tf)
    kw = dict(batch_size=batch_size,num_workers=num_workers,pin_memory=True)

    g = __import__('torch').Generator().manual_seed(seed)

    task_type = "multiclass" if len(raw.classes) > 2 else "binary"

    num_outputs = len(raw.classes) if task_type == "multiclass" else 1

    return DataBundle(
        DataLoader(Subset(train, tr), shuffle=True, generator=g, **kw),
        DataLoader(Subset(valid, va), shuffle=False, **kw),
        DataLoader(test, shuffle=False, **kw),
        3,
        num_outputs,
        task_type,
        tuple(raw.classes)
    )
