from src.utils.transforms import transform, reverse_transform
from src.utils.calculations import get_noisy_image
import torch
from src.utils.visualization import plot
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw) # PIL image of shape HWC


x_start = transform(image).unsqueeze(0)
reverse_transform(x_start.squeeze())

plot(image, [get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])

print("done.")