# 🏦 Predicción de Churn de Clientes Bancarios

[![Made with Python](https://img.shields.io/badge/Made%20with-Python%203.10-blue.svg)](https://www.python.org/)
[![Data Science Bootcamp](https://img.shields.io/badge/Proyecto-Bootcamp-green)](#)
[![Status](https://img.shields.io/badge/Status-Completo-brightgreen)](#)

---

Este proyecto analiza los datos de **10,000 clientes de un banco** para predecir si un cliente abandonará la institución (`churn`), a partir de variables demográficas y de comportamiento.  

Se construyen y comparan modelos de **Machine Learning** (Regresión Logística y Random Forest), incluyendo versiones **balanceadas**, para enfrentar el desbalance de clases típico en problemas de churn.

---

## 📌 Objetivo

Desarrollar un modelo de clasificación que permita:

- Identificar clientes con alto riesgo de abandono.
- Comparar el desempeño de distintos algoritmos (Regresión Logística vs Random Forest).
- Evaluar el impacto del **desbalance de clases (≈80% se quedan, ≈20% se van)**.
- Proponer una solución que pueda apoyar estrategias de **retención de clientes**.

---

## 🛠️ Herramientas utilizadas

- `Python`
- `Pandas`, `NumPy`
- `scikit-learn` (model_selection, LogisticRegression, RandomForestClassifier, métricas)
- `Matplotlib`
- `Jupyter Notebook`

---

## 📊 Contenido del análisis

- ✔ Carga y exploración de un dataset de **10,000 clientes** con **13 características relevantes**.
- ✔ Limpieza de datos:
  - Eliminación de columnas irrelevantes (`RowNumber`, `CustomerId`, `Surname`).
  - Manejo de valores faltantes en `Tenure` mediante imputación.
- ✔ Codificación de variables categóricas con **One-Hot Encoding (OHE)**:
  - `Geography`
  - `Gender`
- ✔ Separación de datos en:
  - Conjunto de entrenamiento, validación y prueba.
- ✔ Entrenamiento de modelos:
  - Regresión Logística (sin balanceo y con `class_weight='balanced'`).
  - Random Forest (sin balanceo y con `class_weight='balanced'`).
- ✔ Evaluación con:
  - `Accuracy`, `Precision`, `Recall`, `F1-score`
  - Curvas **ROC** y métricas **AUC**.

---

## 📈 Resultados clave

- El dataset presenta un **desbalance de clases** (~80% clientes que se quedan, ~20% clientes que se van), lo cual afecta especialmente a modelos lineales como la Regresión Logística.
- La **Regresión Logística balanceada** mejora su `F1` respecto a la versión sin balanceo, pero mantiene un **AUC ≈ 0.692**, lo que limita su capacidad discriminativa global.
- El modelo de **Random Forest balanceado** logra:
  - Mejor equilibrio entre `precision` y `recall`.
  - Un **AUC ≈ 0.84**, superior al de la Regresión Logística.
  - Mejor identificación de la clase minoritaria (clientes que abandonan).

En resumen, **Random Forest balanceado** es el modelo más robusto para este problema.

---

## 🧠 Conclusión

- Se recomienda **implementar el modelo de Random Forest balanceado** como modelo principal para:
  - Detectar clientes en riesgo de churn.
  - Priorizar acciones de retención (campañas, beneficios, contacto personalizado).
- La **Regresión Logística balanceada** puede utilizarse como modelo complementario cuando se requiera:
  - Mayor interpretabilidad de coeficientes.
  - Explicar el impacto de cada variable en el riesgo de abandono.
- El uso conjunto de métricas (`F1` + `AUC`) permite tomar decisiones más sólidas sobre qué modelo desplegar en producción y cómo diseñar estrategias de retención.

---

## 📁 Estructura del proyecto

```text
Churn-Clientes-Banco/

├── Src/
│   └── Modelo_Churn_Banco.py          # Código fuente limpio con el pipeline de ML
│
├── Notebooks/
│   └── Proyecto10.ipynb         # Notebook con el desarrollo paso a paso
│
├── Data/
│   └── Churn.csv                      # Dataset del proyecto (ajusta el nombre si es distinto)
│
├── requirements.txt                   # Librerías necesarias
├── .gitignore
└── README.md

## 👨‍💻 Autor

Axel López

🔗 LinkedIn
✉️ axellpzlin@gmail.com
🎯 Proyecto de portafolio - Bootcamp de Ciencia de Datos
