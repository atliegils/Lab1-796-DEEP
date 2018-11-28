# Lab1
Deep Learning [SC-T-796-DEEP] 
Reykjavik University

## Setup Environment [TODO]
To start things off, you will need to create a Notebook on your Google Drive. You can do this through the [Colab website](http://colab.research.google.com) 

Once there, you can paste the URL to this repository (https://github.com/atliegils/Lab1-796-DEEP) and pull the file to your Drive.

For improved performance, you should toggle the GPU accelerator.

`Edit` -> `Notebook Settings` -> `Hardware Accelerator` -> `GPU`

A lot of keyboard shortcuts are available and save time `Tools` -> `Keyboard Shortcuts` 

Now, that the environment is set up, we can start creating the model. 

For each of the following tasks, create a new section header cell in the notebook.

### Task 1 - Import packages
```python
import sys
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
```

## Data
We use the MNIST dataset to train our model

### Task 2 - Get Data
Load the dataset from MNIST

### Task 3 - Data Preprocessing
Split the data and convert the target to binary

## Model 
What architecture should look like

### Task 4 - Create Model
Construct the model using Keras

### Task 5 - Train Model
Use the preprocessed data to train the model

### Task 6 - Save / Load Model
Keep trained version in a file

Load a trained model from a file

### Evaluate Model
Measure the accuracy of your model
