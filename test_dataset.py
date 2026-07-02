from src.data.dataset import ScreenPhotoDataset, load_dataset
from src.data.transformer import train_transform

paths, labels = load_dataset("dataset")

dataset = ScreenPhotoDataset(
    paths,
    labels,
    transform=train_transform,
)

image, label = dataset[0]

print(image.shape)
print(image.dtype)
print(label)