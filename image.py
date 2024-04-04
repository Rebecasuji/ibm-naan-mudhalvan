# Import required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Define the input size and number of classes
img_size = 224
num_classes = 2

# Create an instance of the ImageDataGenerator class for data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Create training data generator
train_generator = data_augmentation.flow_from_directory(
    directory=r'C:\Users\ELCOT\Desktop\Imageclassification_CNN_Python\Dataset\dogs_cats_sample_1000\train', #directory of the train dataset
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

# Create validation data generator
validation_generator = ImageDataGenerator().flow_from_directory(
    directory=r'C:\Users\ELCOT\Desktop\Imageclassification_CNN_Python\Dataset\dogs_cats_sample_1000\valid',  #directory of the validation dataset
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Test the model on a single image
test_image = keras.preprocessing.image.load_img(r'C:\Users\ELCOT\Desktop\Imageclassification_CNN_Python\Dataset\dogs_cats_sample_1000\dog.142.jpg', target_size=(img_size, img_size))    #path for the image to classify
test_image = keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
predictions = model.predict(test_image)
if predictions[0][0] > predictions[0][1]:
    print("Cat")
else:
    print("Dog")
