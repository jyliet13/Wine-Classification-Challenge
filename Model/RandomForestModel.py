#%%
# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
# import graphviz
#%%
df = pd.read_csv('../data/dataset.csv')
#%%
# Split the data into features (X) and target (y)
X = df.drop(columns=['Class'], axis=1)
y = df['Class']
#%%
# Verificar las dimensiones
print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

# Mostrar los primeros registros
print(X.head())
print(y.head())
#%%
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%%
# Verificar tamaños
print("Tamaño de X_train:", X_train.shape)
print("Tamaño de X_test:", X_test.shape)
#%%
param_dist = {
    "n_estimators": randint(50, 200),  # Número de árboles (50 a 200)
    "max_depth": randint(5, 20),       # Profundidad máxima (5 a 20)
    "min_samples_split": randint(2, 10) # Mínimo de muestras por división
}
#%%
# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#%%
# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)
#%%
# Aplicar RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf_classifier, param_distributions=param_dist, n_iter=20, cv=5, scoring="accuracy", random_state=42
)
#%%
# Entrenar el modelo con búsqueda de hiperparámetros
random_search.fit(X_train, y_train)
#%%
# Imprimir los mejores parámetros encontrados
print("Mejores hiperparámetros:", random_search.best_params_)
#%%
# Evaluar el modelo optimizado
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
#%%
# Make predictions
y_pred = rf_classifier.predict(X_test)
#%%
# Sample prediction
sample = X_test.iloc[0:1]  # Keep as DataFrame to match model input format
prediction = rf_classifier.predict(sample)

#%%
# Export the first three decision trees from the forest
# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
