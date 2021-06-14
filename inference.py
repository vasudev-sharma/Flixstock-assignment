import torch
from pathlib import Path
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FashionDataset
import os

def test_csv():
    pass


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load a model
    model_path = Path('model/model.h5')
    model = torch.load(str(model_path), map_location=device)

    # path of test data
    test_data_path = Path('data')
    list_test_images = os.listdir(str(test_data_path / 'test'))

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds_test = FashionDataset(test_data_path, root_dir='test', transform=test_transforms, samples=list_test_images)
    dl_test = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)
    print('Size of the dataset is ', len(ds_test))

    # get the model predicitons on the test dataset
    torch.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dl_test, 1):
            # Forward pass
            preds = torch.sigmoid(model(images))

            # threshold the values
            preds = torch.tensor(preds > 0.5, dtype=torch.float)
            print(preds)


    test_filename = 'test_attributes.csv'



    