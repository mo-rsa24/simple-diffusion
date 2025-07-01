from pathlib import Path
import torch
import torchvision
from src.evaluation.composite_metrics import box_iou, color_consistency

def main(samples_dir: str, target_rgb=(255,0,0)):
    paths = sorted(Path(samples_dir).glob('sample_*.png'))
    scores = []
    for p in paths:
        img = torchvision.io.read_image(str(p)).float()
        # dummy masks/boxes; real implementation would load metadata
        mask = img.mean(0) > 128
        box = torch.tensor([5,5,23,23])
        scores.append(color_consistency(img/255., mask, target_rgb))
    print('mean color error:', sum(scores)/len(scores))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()
    main(args.dir)