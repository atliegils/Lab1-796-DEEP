![](https://www.ru.is/skin/basic9k/i/sitelogo.svg)

LAB 1
---

T-796-DEEP, Introduction to Deep Learning, 2018-3

---

## Setup Environment
To start things off, you will need to create a Notebook on your Google Drive. You can do this through the [Colab website](http://colab.research.google.com) 

Once there, you can select GitHub, paste the URL to this repository (https://github.com/atliegils/Lab1-796-DEEP) and pull the file to your Drive.

This notebook will be opened in a playground mode, so you will have to save a copy to your Drive or changes may be lost!

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

print('Python: %s' % sys.version)
```

Output should be the following:

```
Using TensorFlow backend.
Python: 3.6.7 (default, Oct 22 2018, 11:32:17) 
[GCC 8.2.0]
```

## Data
We use the MNIST dataset to train our model



### Task 2 - Get Data
Load the dataset from MNIST

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### Task 3 - Data Preprocessing
Split the data and convert the target to binary

```python
num_classes = 10
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

Output:

```
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
```

## Model 
What architecture should look like


### Task 4 - Create Model
Construct the model using Keras

```python
batch_size = 128
epochs = 12

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```

### Task 5 - Train Model
Use the preprocessed data to train the model

```python
```

### Task 6 - Save / Load Model
Save current model to a __.h5__ file

```python
model.save('my_model.h5')
del model  # deletes the existing model
```

Load the model from a file

```python
loaded_model = load_model('my_model.h5')
```

### Task 7 - Evaluate Model
Measure the accuracy of your model

```python
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
