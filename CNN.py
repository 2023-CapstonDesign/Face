from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

img1 = image.load_img('pss.JPG', target_size=(150, 150))
img2 = image.load_img('userface.jpg', target_size=(150, 150))

x1 = image.img_to_array(img1)
x2 = image.img_to_array(img2)

x1 = np.expand_dims(x1, axis=0)
x2 = np.expand_dims(x2, axis=0)

images = np.vstack([x1, x2])
classes = model.predict(images, batch_size=10)

if classes[0] > 0.5:
    print("Person 1 and Person 2 are the same.\n")
    print("Password is Correct")
else:
    print("Person 1 and Person 2 are different.")
    print("Password is wrong")