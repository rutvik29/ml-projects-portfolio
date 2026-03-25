"""COVID-19 Detection from Chest X-Rays using Deep Learning.

Preprocessing + model evaluation with confusion matrix, ROC-AUC, F1.
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import cv2
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


IMG_SIZE = 224
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']


def preprocess_xray(image_path: str) -> np.ndarray:
    """Enhanced X-ray preprocessing with histogram equalization."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.equalizeHist(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return img_resized.astype(np.float32) / 255.0


def build_covid_model(num_classes: int = 3) -> Model:
    base = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=outputs)


if __name__ == "__main__":
    model = build_covid_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    model.summary()
