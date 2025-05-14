import os
import numpy as np
import cv2
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv2D, LayerNormalization,
    Add, Reshape, GlobalAveragePooling2D, MultiHeadAttention, Embedding, Concatenate
)
from tensorflow.keras.optimizers import Adam
from concurrent.futures import ThreadPoolExecutor

# Paths and Constants

base_dir = r'C:\Users\raman\OneDrive\Desktop\Animal Proj\Dataset'
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

WIDTH, HEIGHT, CHANNEL = 224, 224, 3
PATCH_SIZE = 16
NUM_PATCHES = (WIDTH // PATCH_SIZE) ** 2
NUM_CLASSES = len(os.listdir(train_dir))
EPOCHS = 20
BATCH_SIZE = 32
VERBOSE = 1

# Image Processing Function
def process_block(img_path, width, height):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (width, height))
        img = img / 255.0
    return img

# Load Data in Parallel
def load_data_parallel(data_dir):
    X, y = [], []
    file_list = [(os.path.join(data_dir, dirname, file_name), dirname)
                 for dirname in os.listdir(data_dir)
                 for file_name in os.listdir(os.path.join(data_dir, dirname))]

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda f: (process_block(f[0], WIDTH, HEIGHT), f[1]), file_list)

    for img, label in results:
        if img is not None:
            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

# Prepare Data
X_train, y_train = load_data_parallel(train_dir)
X_test, y_test = load_data_parallel(test_dir)

# Encode Labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
np.save('label_encoder_classes.npy', label_encoder.classes_)

# ConvNeXt Block
def convnext_block(x, filters):
    x_dw = Conv2D(filters, kernel_size=7, padding='same', groups=filters)(x)
    x_dw = LayerNormalization()(x_dw)
    x_pw = Conv2D(filters * 4, kernel_size=1, padding='same', activation='gelu')(x_dw)
    x_pw = Conv2D(filters, kernel_size=1, padding='same')(x_pw)
    x = Add()([x, x_pw])
    x = LayerNormalization()(x)
    return x

# Transformer Encoder Block
def transformer_encoder(x, num_heads, ff_dim):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    ff_output = Dense(ff_dim, activation='gelu')(x)
    ff_output = Dense(x.shape[-1])(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization()(x)
    return x

# CaT Block
def cat_block(x, num_heads, ff_dim):
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = Reshape((h * w, c))(x)
    x = transformer_encoder(x, num_heads=num_heads, ff_dim=ff_dim)
    x = Reshape((h, w, c))(x)
    return x

# **Fixed: Four ConvNeXt Stages in CaTNet**
def build_catnet(input_shape):
    inputs = Input(shape=input_shape)

    # Stage 1
    x = Conv2D(64, kernel_size=4, strides=4, padding='same')(inputs)
    for _ in range(3):
        x = convnext_block(x, 64)

    # Stage 2
    x = Conv2D(96, kernel_size=2, strides=2, padding='same')(x)
    for _ in range(3):
        x = convnext_block(x, 96)

    # Stage 3
    x = Conv2D(128, kernel_size=2, strides=2, padding='same')(x)
    for _ in range(3):
        x = convnext_block(x, 128)

    # Stage 4
    x = Conv2D(256, kernel_size=2, strides=2, padding='same')(x)
    for _ in range(3):
        x = convnext_block(x, 256)

    # CaT Block
    x = cat_block(x, num_heads=4, ff_dim=384)
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs, x, name="CaTNet")
    return model

# ViT Model
def build_vit(input_shape):
    inputs = Input(shape=input_shape)
    patches = Conv2D(64, kernel_size=PATCH_SIZE, strides=PATCH_SIZE, padding='valid')(inputs)
    patches = Reshape((NUM_PATCHES, -1))(patches)
    projection = Dense(64, activation='linear')(patches)
    position_embedding = Embedding(input_dim=NUM_PATCHES, output_dim=64)(tf.range(NUM_PATCHES))
    x = projection + position_embedding
    for _ in range(8):
        x = transformer_encoder(x, num_heads=4, ff_dim=128)
    x = GlobalAveragePooling2D()(Reshape((int(NUM_PATCHES**0.5), int(NUM_PATCHES**0.5), 64))(x))
    model = Model(inputs, x, name="ViT_Model")
    return model

# **Fixed: Proper Model Instantiation**
inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL))
catnet_model = build_catnet((HEIGHT, WIDTH, CHANNEL))
vit_model = build_vit((HEIGHT, WIDTH, CHANNEL))

catnet_features = catnet_model(inputs)
vit_features = vit_model(inputs)

# Concatenate Features
fused_features = Concatenate()([catnet_features, vit_features])

# **Add Dense and Dropout Layers**
x = Dense(512, activation='relu')(fused_features)
x = Dropout(0.5)(x)  # Dropout layer with 50% drop probability
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Another Dropout layer

# Final Output Layer
output = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the final model
feature_extractor = Model(inputs, output, name="FeatureFusion")

# **Fixed: Extract Features Correctly**
X_train_fused = feature_extractor.predict(X_train)
X_test_fused = feature_extractor.predict(X_test)

# Train an XGBoost Classifier
clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
clf.fit(X_train_fused, y_train, eval_set=[(X_train_fused, y_train), (X_test_fused, y_test)], verbose=True)

# Predict
y_pred = clf.predict(X_test_fused)

# Compute Accuracy
accuracy = np.mean(y_pred == y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Training vs Validation Loss
results = clf.evals_result()
plt.plot(results['validation_0']['mlogloss'], label='Train Loss')
plt.plot(results['validation_1']['mlogloss'], label='Test Loss')
plt.legend()
plt.title("Train vs Test Loss")
plt.show()

# Save the model if needed
feature_extractor.save("hybrid_model_savedmodel", save_format="tf")
print("Training Ended")

