import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class FashionDataset(Dataset):
  def __init__(self, root, root_dir='images', transform=None, target_transform=None, samples=None):
    self.root = root
    self.transform = transform
    self.target_transform = target_transform

    if samples is not None:
      self.samples = os.listdir(str(self.root / root_dir))
    else:
      self.samples = samples

    # Attributes dataframe
    self.df_attributes = pd.read_csv(str(self.root / 'attributes.csv'))

    # For each attribute, one hot encode it
    self.preprocess_targets() 

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, index):

    # shuffle the images list
    np.random.shuffle(self.samples)

    # name of the image
    filename_image = self.samples[index]

    try:
      image = Image.open(str(self.root / 'images' / filename_image)).convert('RGB')
    except Exception as e:
      print('Path of the image is', str(self.root / 'images' / filename_image))
      print('Unable to read the image')

    # retreive the specific row of given index
    df_row = self.df_attributes.loc[self.df_attributes['filename'] == filename_image]

    if self.transform is not None:
      image = self.transform(image)

    # Target
    target_start_idx = self.df_attributes.columns.get_loc('neck_0.0')
    target = torch.tensor(self.df_attributes.iloc[0].tolist()[target_start_idx:], dtype=torch.float32)

    # return target and labels
    return image, target
  
  def preprocess_targets(self):
    # Drop rows which have na
    self.df_attributes = self.df_attributes.dropna()

    # one hot encode the Neck attribute
    one_hot_neck = pd.get_dummies(self.df_attributes.neck, prefix='neck')

    # one hot encode the sleeve_length attribute
    one_hot_sleeve_length = pd.get_dummies(self.df_attributes.sleeve_length, prefix='sleeve_length')

    # one hot encode the patter attribute
    one_hot_pattern = pd.get_dummies(self.df_attributes.pattern, prefix='pattern')

    # concatenate the one hot encoded attributes to dataframe
    self.df_attributes = pd.concat([self.df_attributes, one_hot_neck, one_hot_sleeve_length, one_hot_pattern], axis=1)

  def getImagesList(self):
    return self.samples
