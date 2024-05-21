import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

# Ignorar advertencias
warnings.filterwarnings('ignore')

# Cargar los datos
df = pd.read_csv('/content/drive/My Drive/supermarket.csv')

# Convertir la columna HighSales a binaria
df['HighSales'] = df['Total'].apply(lambda x: 1 if x > 300 else 0)

# Seleccionar caracteristicas y objetivo
features = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Payment', 'Rating']
X = pd.get_dummies(df[features], drop_first=True)
y = df['HighSales']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarizar las caracteristicas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Crear el modelo
nn_model = Sequential()
nn_model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(16, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = nn_model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=0)

# Evaluar el modelo
loss, accuracy_nn = nn_model.evaluate(X_test, y_test, verbose=0)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype("int32")

print("Neural Network Accuracy:", accuracy_nn)
print("Neural Network Classification Report:\n", classification_report(y_test, y_pred_nn))

# Matriz de confusion
cm_nn = confusion_matrix(y_test, y_pred_nn)

# Graficar la matriz de confusion
plot_confusion_matrix(cm_nn, "Confusion Matrix - Neural Network")

# Graficar la precision a lo largo de las epocas
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
