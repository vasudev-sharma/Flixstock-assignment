import torch
from pathlib import Path
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FashionDataset
import numpy as np
import os
from model import get_model

def get_csv(filename):
    # TODO: Save the model predictions in CSV file
    pass


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load a model
    model_name = 'densenet121'
    model = get_model(model_name)
   
    model_path = Path('model/densenet121_model.h5')
    checkpoint = torch.load(str(model_path), map_location=device)
    model.load_state_dict(checkpoint)

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
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dl_test, 1):
            # Forward pass
            preds = torch.sigmoid(model(images))

            # threshold the values
            preds = np.array(preds.numpy() > 0.5, dtype=float)
            print(preds)
            


    
    # Create csv file for predictions
    test_filename = 'test_attributes.csv'

    get_csv(test_filename)



    