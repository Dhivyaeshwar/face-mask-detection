import os
import cv2
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 1
DATASET_DIR = "data"
TEST_IMAGE_PATH = "test_images/test_sample_1.jpg"
MODEL_PATH = "mobilenetv2_mask_model.keras"  # Native Keras format

# ========== DATA LOADING ==========
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# ========== MODEL BUILDING ==========
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========== TRAINING ==========
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ========== SAVE MODEL ==========
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")

# ========== PREDICT FUNCTION ==========
def predict_image(image_path):
    model = load_model(MODEL_PATH)

    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    label = "With Mask" if np.argmax(prediction) == 0 else "Without Mask"

    # Show image with prediction
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (300, 300))
    color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
    cv2.putText(img_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Prediction", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========== RUN PREDICTION ==========
predict_image(TEST_IMAGE_PATH)
