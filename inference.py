import torch
from pathlib import Path
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FashionDataset
import numpy as np
import os
from model import get_model
import pandas as pd

def get_csv(preds, csv_filename, filenames_images):
    # TODO: Save the model predictions in CSV file

    # Start and end indexes of neck attribute
    start_neck_idx = 0
    end_neck_idx = 6
    neck_targets = preds[:, start_neck_idx:end_neck_idx+1]
    
    # Neck attribute: label encoding of one-hot encoded vectors
    neck_targets_labels = np.where(neck_targets == 1)[1]

    # Start and end indexes of sleeve_length attribute
    start_sleeve_idx = end_neck_idx + 1
    end_sleeve_idx = 10

    sleeve_targets = preds[:, start_sleeve_idx:end_sleeve_idx+1]

    # Sleeve attribute: label encoding of one-hot encoded vectors
    sleeve_targets_labels = np.where(sleeve_targets == 1)[1]

    # Start of pattern attribute
    start_pattern_idx = end_sleeve_idx + 1
    pattern_targets = preds[:, start_pattern_idx:]

     # Pattern attribute: label encoding of one-hot encoded vectors
    sleeve_targets_labels = np.where(sleeve_targets == 1)[1]


    df_attributes = pd.read_csv(str(Path(csv_filename).parent / 'attributes.csv'))


    assert len(df_attributes) == len(csv_filename)
    df.to_csv(str(Path(csv_filename).parent / csv_filename))



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
        for batch_idx, (images, _, filenames_images) in enumerate(dl_test, 1):
            # Forward pass
            preds = torch.sigmoid(model(images))

            # threshold the values
            preds = np.array(preds.numpy() > 0.5, dtype=float)          
            


    
    # Create csv file for predictions
    test_filename = 'data/test_attributes.csv'

    get_csv(preds, test_filename, filenames_images)



    