# Clasificación de Expresión Génica con Machine Learning

![R](https://img.shields.io/badge/R-4.x-276DC3?logo=r&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Análisis de un conjunto de datos de expresión génica (RNA-Seq) mediante técnicas de machine learning supervisadas y no supervisadas, desarrollado como actividad grupal del Máster en Bioinformática (UNIR).

El dataset contiene el perfil de expresión de miles de genes en 801 muestras, clasificadas en 5 tipos tumorales (codificados como CFB, AGH, CGC, CHC, HPB). El objetivo es reducir la dimensionalidad, identificar estructura en los datos y construir clasificadores capaces de predecir el tipo de tumor a partir del perfil de expresión génica.

## Contenido

- **Preprocesamiento**: control de calidad, filtrado de genes de baja varianza, escalado.
- **Reducción de dimensionalidad**: PCA, t-SNE (Rtsne) y UMAP (uwot) para visualizar la separación entre clases.
- **Clustering no supervisado**: k-means y clustering jerárquico.
- **Selección de variables**: regularización LASSO (glmnet) para identificar genes más informativos.
- **Clasificación supervisada**: kNN, Naive Bayes y SVM, evaluados con validación cruzada 10-fold.
- **Evaluación de modelos**: curvas ROC y Precision-Recall por clase (AUC > 0.97, AUPRC > 0.86).

## Estructura del repositorio

```
├── R/
│   └── clasificacion_expresion_genica.R   # Script principal del análisis
├── data/
│   ├── gene_expression.csv                # Matriz de expresión génica
│   ├── classes.csv                        # Etiquetas de clase por muestra
│   └── column_names.txt                   # Nombres de genes (columnas)
├── LICENSE
└── README.md
```

## Dependencias (R)

tidyverse · caret · factoextra · randomForest · Rtsne · uwot · gridExtra · cluster · glmnet · PRROC · pROC

## Uso

1. Instalar R y las dependencias listadas arriba.
2. Ejecutar `R/clasificacion_expresion_genica.R` de forma secuencial.

## Autoría

Trabajo grupal — Máster en Bioinformática (UNIR):
Samuel Salazar Diaz, Samuel David Espitia Contreras, Luisa Ospina Londoño, Miller Esneyder Vargas Santiago.

## Licencia

Distribuido bajo licencia MIT — ver [LICENSE](LICENSE).
