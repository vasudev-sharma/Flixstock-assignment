import torch
from pathlib import Path
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FashionDataset

def test_csv():
    pass


if __name__ == '__main__':
    
    # load a model
    model_path = Path('model/model.h5')
    model = torch.load(str(model_path))

    # path of test data
    test_data_path = Path('data')

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds_test = FashionDataset(str(test_data_path), root_dir='test', transform=test_transforms)
    dl_test = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)


    # get the model predicitons on the test dataset
    torch.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dl_test, 1):
            # Forward pass
            preds = model(images)

            


    test_filename = 'test_attributes.csv'



    