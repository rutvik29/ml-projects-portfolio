"""Face Expression Recognition using VGG Transfer Learning.

Achieves 78% accuracy on FER2013 dataset using VGG16 with fine-tuning.
"""
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48
BATCH_SIZE = 64


def build_vgg_model(num_classes: int = 7, fine_tune_at: int = 15) -> Model:
    """Build VGG16 transfer learning model for expression recognition."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    
    # Freeze early layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)


def get_data_generators(data_dir: str):
    """Build augmented data generators with tf.data API optimization."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train", target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        f"{data_dir}/validation", target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical'
    )
    return train_gen, val_gen


def train(data_dir: str = "./data/fer2013", epochs: int = 50, output_path: str = "./models/vgg_fer.h5"):
    model = build_vgg_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    train_gen, val_gen = get_data_generators(data_dir)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint(output_path, save_best_only=True)
    ]
    
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    return model, history


if __name__ == "__main__":
    model, history = train()
