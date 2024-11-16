# 📚 Introducción a Scikit-learn y la Regresión Lineal  

## 🧠 ¿Qué es Scikit-learn?  
**Scikit-learn** es una biblioteca de Python diseñada para el **Machine Learning**.  
- Contiene herramientas para **entrenar modelos**, hacer predicciones y analizar datos.  
- Incluye algoritmos para clasificación, regresión, clustering, reducción de dimensiones, y más.  
- **Fácil de usar**: Ideal tanto para principiantes como para expertos en ciencia de datos.  

## 📈 ¿Qué es la regresión lineal?  
La **regresión lineal** es una técnica de Machine Learning que:  
- Encuentra la **relación entre una variable dependiente (\(y\))** y una o más **variables independientes (\(x\))**.  
- Sirve para **hacer predicciones**, como predecir ventas, precios, o tendencias.  
- Utiliza la fórmula: y = mx + b
- Donde:  
  - (m): Es la pendiente, que muestra cómo cambia (y) según (x).  
  - (b): Es la intersección, el valor de (y) cuando (x = 0).  

  ## ⚙️ Implementación de Regresión Lineal con Scikit-learn  

### 1️⃣ **Instalar Scikit-learn**  

# Asegúrate de tener Scikit-learn instalado en tu entorno:  

`bash
pip install scikit-learn`

### 2️⃣ Importar librerías necesarias

`from sklearn.linear_model import LinearRegression  # Modelo de regresión lineal
from sklearn.model_selection import train_test_split  # División de datos
from sklearn.metrics import mean_squared_error  # Métrica de evaluación`

### 3️⃣ Preparar los datos

# Imagina que tienes un dataset con valores de entrada 𝑋 y salida 𝑦

# Variables independientes y dependientes
`X = [[1], [2], [3], [4], [5]]  # Entrada
y = [2, 4, 6, 8, 10]           # Salida`

### 4️⃣ Dividir los datos en entrenamiento y prueba

# Dividimos los datos para entrenar el modelo y probarlo
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`

### 5️⃣ Entrenar el modelo

# Creamos y entrenamos el modelo de regresión lineal:

# Crear el modelo
`model = LinearRegression()`

# Entrenar el modelo con los datos de entrenamiento
`model.fit(X_train, y_train)`

### 6️⃣ Hacer predicciones

# Usamos el modelo para predecir valores en los datos de prueba:

# Hacer predicciones
`y_pred = model.predict(X_test)`

`print("Predicciones:", y_pred)`

### 7️⃣ Evaluar el modelo

# Calculamos el error promedio cuadrático (MSE) para medir el rendimiento del modelo:

# Evaluar el modelo

`error = mean_squared_error(y_test, y_pred)
print("Error promedio cuadrático:", error)`

📊 Código completo

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Datos
X = [[1], [2], [3], [4], [5]]  # Entrada
y = [2, 4, 6, 8, 10]           # Salida

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
error = mean_squared_error(y_test, y_pred)

# Resultados
print("Predicciones:", y_pred)
print("Error promedio cuadrático:", error)

### 🚀 ¿Por qué usar Scikit-learn para regresión lineal?

✅ Simplicidad: Hace que implementar algoritmos sea rápido y sencillo.
✅ Optimización: Los algoritmos están optimizados para ser rápidos y eficientes.
✅ Integración: Funciona perfectamente con otras librerías como Pandas y NumPy.

