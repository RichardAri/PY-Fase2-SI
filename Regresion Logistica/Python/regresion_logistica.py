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

from sklearn.linear_model import LogisticRegression

# Inicializar y entrenar el modelo de regresion logística
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Realizar predicciones
y_pred_log_reg = log_reg.predict(X_test)

# Evaluar el modelo
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy:", accuracy_log_reg)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Matriz de confusion
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)

# Graficar la matriz de confusion
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(cm_log_reg, "Confusion Matrix - Logistic Regression")
