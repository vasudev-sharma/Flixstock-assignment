# Flixstock-assignment

## Problem statement: Predicting Deep Fashion Attributes
For more details, refer [here](https://github.com/vasudev-sharma/Flixstock-assignment/blob/master/Problem_Statement.pdf)

## Install dependencies
Inside a virtual environment ( `conda` / `virtualenv`), install the dependencies by running the command below in the terminal. <br>
```bash
$ pip install -r requirements.txt
```

## Usage:

1. **Preprocessing and training**<br>
  To train the model, refer the notebook [Assignment.ipynb](https://github.com/vasudev-sharma/Flixstock-assignment/blob/master/Assignment.ipynb).<br>

  **NOTE:-** It is recommend to leverage free Colab GPU resource, if personal GPU is not available, to train models faster.

2. **Experiment-Tracking**<br>
-> Weights and Biases: https://wandb.ai/vs74/Flixstock-assignment

3. **Inference** <br>
-> Run the script in the terminal.<br>
```bash 
$ python inference.py
```
-> This will generate [`test_attributes.csv`](https://github.com/vasudev-sharma/Flixstock-assignment/blob/master/data/test_attributes.csv) file under `data` directory for the test images located in `data/test` directory. 

## Models
Under the `models` directory, fine tuned models are available.

The following models were finetuned on the dataset<br>

a) [EfficientNet](https://github.com/vasudev-sharma/Flixstock-assignment/blob/master/model/densenet121_model.h5) <br>
b) [ResNet](https://github.com/vasudev-sharma/Flixstock-assignment/blob/master/model/efficientnet-b3.h5) <br>
c) [DenseNet](https://github.com/vasudev-sharma/Flixstock-assignment/blob/master/model/resnet50.h5<br>

## TODO
- [ ] Look at optimal ways of reducing bias (Don't drop the 'NA' values of [`attribute.csv`](https://github.com/vasudev-sharma/Flixstock-assignment/blob/master/data/attributes.csv) file)
