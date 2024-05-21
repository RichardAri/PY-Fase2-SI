import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
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

# Inicializar y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Realizar predicciones
y_pred_knn = knn.predict(X_test)

# Evaluar el modelo
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# Matriz de confusion
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Graficar la matriz de confusion
plot_confusion_matrix(cm_knn, "Confusion Matrix - KNN")
