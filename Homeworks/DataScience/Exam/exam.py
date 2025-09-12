# Step 1
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
# Step 2
import plotly.express as px
import pandas as pd
# Step 3
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. --------------------------------------------------------------------
# Datasets
train = fetch_20newsgroups(subset='train')  # Training Dataset
test = fetch_20newsgroups(subset='test')  # Test Dataset

# Vectorizar
max_features = 50000
num_classes = len(train.target_names)
vectorizer = TfidfVectorizer(
    stop_words='english', max_features=max_features, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")

# Input Vectors
train_X = vectorizer.fit_transform(train.data)
test_X = vectorizer.transform(test.data)

# Etiquetas
train_y = train.target
test_y = test.target

# 2. --------------------------------------------------------------------

class_names = train.target_names
train_df = pd.DataFrame({
    "Categoria": [class_names[i] for i in train_y]
})

fig = px.histogram(train_df, x="Categoria",
                   title="Distribución de categorías en Train")
fig.update_xaxes(tickangle=45)
fig.show()

train_lengths = [len(text.split()) for text in train.data]
len_df = pd.DataFrame({"Longitud": train_lengths})

fig = px.box(len_df, y="Longitud",
             title="Distribución de longitudes de documentos (Train)")
fig.show()

# 3. --------------------------------------------------------------------
k_train_X = tf.convert_to_tensor(train_X.toarray(), dtype=tf.float32)
k_test_X = tf.convert_to_tensor(test_X.toarray(), dtype=tf.float32)

model = models.Sequential([
    layers.Input(shape=(max_features,)),
    layers.Dense(256, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax'),
])

# 4. --------------------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=6e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1)
]

history = model.fit(k_train_X, train_y,
                    validation_data=(k_test_X, test_y),
                    batch_size=256, epochs=20,
                    callbacks=callbacks)

# 5. --------------------------------------------------------------------
df = pd.DataFrame(history.history)
df["epoch"] = range(1, len(df)+1)

fig_loss = px.line(df, x="epoch", y=[c for c in df.columns if c in ["loss", "val_loss"]],
                   title="Curvas de pérdida")
fig_loss.update_layout(xaxis_title="Época", yaxis_title="Loss")
fig_loss.show()

# Curvas de accuracy
fig_acc = px.line(df, x="epoch", y=[c for c in df.columns if c in ["accuracy", "val_accuracy"]],
                  title="Curvas de accuracy")
fig_acc.update_layout(xaxis_title="Época", yaxis_title="Accuracy")
fig_acc.show()

# Curva de learning rate (si existe en history)
lr_keys = [c for c in df.columns if c.lower() in ["lr", "learning_rate"]]
if lr_keys:
    fig_lr = px.line(
        df, x="epoch", y=lr_keys[0], title="Learning rate por época")
    fig_lr.update_layout(xaxis_title="Época", yaxis_title="Learning rate")
    fig_lr.show()
