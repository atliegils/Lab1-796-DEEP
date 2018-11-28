![](https://www.ru.is/skin/basic9k/i/sitelogo.svg)

# LAB 1

---

T-796-DEEP, Introduction to Deep Learning, 2018-3

---

## Task 0 - Setup Environment
Start by  creating a Notebook on your Google Drive. You do this through the [Colab website](http://colab.research.google.com) 

Once there, you can select GitHub tab, then paste the following URL:  (https://github.com/atliegils/Lab1-796-DEEP) and open the notebook in a seperate virtual enviroment assigned to your google account.

**This notebook will be opened in a playground mode, so you will have to save a copy to your Drive or changes may be lost!**

For improved performance, you should toggle the GPU accelerator.

`Edit` -> `Notebook Settings` -> `Hardware Accelerator` -> `GPU`

A lot of keyboard shortcuts are available and save time `Tools` -> `Keyboard Shortcuts` 

Now, that the environment is set up, we can started. 

For each of the following tasks, create a new section header cell in the notebook.

## Task 1 - Import packages
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

## Task 2 - Get Data

We use the MNIST dataset to train our model

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## Task 3 - Data Preprocessing

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

## Task 4 - Create Model
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

## Task 5 - Train Model
Use the preprocessed data to train the model.

Notice the speed difference using `GPU` or `TPU`  acceleration.

```python
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
```
Output:
```
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
60000/60000 [==============================] - 11s 181us/step - loss: 0.2695 - acc: 0.9178 - val_loss: 0.0550 - val_acc: 0.9824
Epoch 2/12
60000/60000 [==============================] - 9s 149us/step - loss: 0.0878 - acc: 0.9740 - val_loss: 0.0367 - val_acc: 0.9883
Epoch 3/12
60000/60000 [==============================] - 9s 149us/step - loss: 0.0656 - acc: 0.9807 - val_loss: 0.0328 - val_acc: 0.9892
Epoch 4/12
60000/60000 [==============================] - 9s 150us/step - loss: 0.0548 - acc: 0.9836 - val_loss: 0.0311 - val_acc: 0.9890
Epoch 5/12
60000/60000 [==============================] - 9s 150us/step - loss: 0.0476 - acc: 0.9854 - val_loss: 0.0293 - val_acc: 0.9899
Epoch 6/12
60000/60000 [==============================] - 9s 149us/step - loss: 0.0422 - acc: 0.9868 - val_loss: 0.0289 - val_acc: 0.9906
Epoch 7/12
60000/60000 [==============================] - 9s 149us/step - loss: 0.0369 - acc: 0.9889 - val_loss: 0.0290 - val_acc: 0.9908
Epoch 8/12
60000/60000 [==============================] - 9s 149us/step - loss: 0.0334 - acc: 0.9895 - val_loss: 0.0271 - val_acc: 0.9910
Epoch 9/12
60000/60000 [==============================] - 9s 149us/step - loss: 0.0330 - acc: 0.9896 - val_loss: 0.0337 - val_acc: 0.9891
Epoch 10/12
60000/60000 [==============================] - 9s 150us/step - loss: 0.0300 - acc: 0.9907 - val_loss: 0.0279 - val_acc: 0.9914
Epoch 11/12
60000/60000 [==============================] - 9s 150us/step - loss: 0.0277 - acc: 0.9911 - val_loss: 0.0266 - val_acc: 0.9920
Epoch 12/12
60000/60000 [==============================] - 9s 151us/step - loss: 0.0274 - acc: 0.9918 - val_loss: 0.0296 - val_acc: 0.9912
<keras.callbacks.History at 0x7f9aa51d9198>
```

## Task 6 - Save Model
Save current model to a __.h5__ file.   

```python
model.save('my_model.h5')
del model  # deletes the existing model
```

## Task 7 - Load Model

```python
loaded_model = load_model('my_model.h5')
```

## Task 7 - Evaluate Model

```python
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
