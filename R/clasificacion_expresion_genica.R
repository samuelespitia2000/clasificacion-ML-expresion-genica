###################################################################################
###################################################################################
###                                                                             ###
###  ANÁLISIS DE UN CONJUNTO DE DATOS DE ORIGEN BIOLÓGICO MEDIANTE TÉCNICAS DE  ###
###               MACHINE LEARNING SUPERVISADAS Y NO SUPERVISADAS               ###
###                                                                             ###
###  SAMUEL SALAZAR DIAZ, SAMUEL DAVID ESPITIA CONTRERAS, LUISA OSPINA LONDOÑO  ###
###                      Y MILLER ESNEYDER VERGAS SANTIAGO                      ###
###                                                                             ###
###################################################################################
###################################################################################

# Configuración del entorno de trabajo
setwd("/home/samuel/Escritorio/Maestria_Bioinformatica/Actividades/Algoritmos_e_IA/actividad_3/Data")

# Librerias necesarias
library(tidyverse)
library(caret)
library(factoextra)
library(randomForest)
library(Rtsne)
library(uwot)
library(gridExtra)
library(cluster)
library(glmnet)
library(caret)
library(PRROC)
library(pROC)

# Carga de dataset
columnas <- read.table("column_names.txt")
genes <- read.csv2("gene_expression.csv", header = FALSE)
clases <- read.csv2("classes.csv", header = FALSE)

# Asignación de nombres columnas de los genes
colnames(genes) <- columnas[,1]

# Asignación de nombres a las filas
colnames(clases) <- c("sample", "clase") # asignamos nombre a las columnas del df de las clases
genes <- cbind(clases, genes) # unimos todo en un solo df

# Convertimos los registros en as.numeric
genes <- genes %>%
  mutate(across(-c(sample, clase), as.numeric))

# Comprobación de NA
colSums(is.na(genes)) # no tine valores NA

# Conteo de ceros por columna
ceros_por_columna <- colSums(genes[, -c(1, 2)] == 0, na.rm = TRUE)
ceros_por_columna 

# Calculamos la varianza 
varianzas <- apply(genes[, -c(1, 2)], 2, var)
table(varianzas < 0.05)
names(varianzas)[varianzas < 0.05] # nombres de los genes con varianza menor a 0.05
genes_var_bajas <- names(varianzas)[varianzas < 0.05] # guardamos los nombres

# Filtrado de genes con baja varianza
genes_fil <- genes %>% 
  select(-all_of(genes_var_bajas)) # eliminamos genes con varianza menor a 0.05

# Realizamos este filtrado de los genes que presentan una varianza menor a 0.05 con el 
# objetivo de reducir la dimensionalidad del conjunto de datos y eliminar aquellos que
# no aportan información relevante para el análisis posterior.

# Escalar los genes
X_scaled <- scale(genes_fil %>% select(-clase, -sample))

###########################################################################
###########################################################################
###                                                                     ###
###                      APENDIZAJE NO SUPERVIZADO                      ###
###                                                                     ###
###########################################################################
###########################################################################

### REDUCCIÓN DE DIMENSIONALIDAD
##----------------------------------------------------------
##  1. t-Distributed Stochastic Neighbor Embedding (t-SNE)  
##----------------------------------------------------------
# Este modelo fue seleccionado debido a su capacidad para preservar las relaciones locales 
# de datos de alta dimensionalidad en un mapeo de baja dimensionalidad y, en el caso de 
# nuestra base de datos, que cuenta con 496 genes como variables y 801 registros, esto es 
# algo importante a tener en cuenta a la hora de seleccionar el método. Este método no tiene 
# parámetros para establecer lo que hace mucho mas sencillo su uso.
#
# Las ventajas que tiene t-SNE son que permite visualizar datos de alta dimensionalidad en 
# un espacio de menor dimensión, facilitando la identificación de patrones y agrupamientos 
# en los datos, y es eficiente para identificar la estructura de los datos, lo que puede ser 
# beneficioso para detectar subgrupos o clústeres dentro de los datos biológicos.
# 
# Las desventajas son que t-SNE puede ser computacionalmente intensivo, especialmente para 
# conjuntos de datos grandes y su componente estocástico puede llevar a resultados diferentes 
# en ejecuciones sucesivas, lo que puede dificultar la reproducibilidad de los resultados.

set.seed(2003) 
tsne <- Rtsne(X=X_scaled, dims=2) # Realizamos t-SNE
tsne_result <- data.frame(tsne$Y) # Convertimos a data frame

# Graficamos
ggplot(tsne_result, aes(x = X1, y = X2, color = genes_fil$clase)) +
  geom_point(size = 2, alpha = 0.85) +
  scale_color_manual(values = c("red", "blue", "green", "orange", "purple")) +
  labs(title = "t-SNE", x = "Dim 1", y = "Dim 2", color = "clase") +
  theme_classic() +
  theme(plot.title=element_text(hjust=0.5))

# Como se puede observar en la gráfica resultantes este método ha logrado diferenciar claramente las 
# diferentes clases del dataset, a pesar de algunas muestras que se encuentran mezcladas entre sí.

##-----------------------------------------------------------
##  2. Unifrom Manifold Approximation and Projection (UMAP)  
##-----------------------------------------------------------
# Este modelo fue seleccionado debido a su capacidad para preservar tanto las relaciones 
# locales como globales de los datos en un espacio de menor dimensión. Lo que es importante 
# con datos biológicos, donde las relaciones entre diferentes tipos de células o condiciones 
# pueden ser complejas. Los parámetros seleccionados se establecieron despues de realizar 
# diferentes pruebas decidimos conservar esa configuración pero no descartamos la posibilidad 
# de que otros parámetros puedan funcionar mejor.
# 
# Las ventajas que tiene esta metodología son que es eficiente en términos computacionales, lo 
# que permite trabajar con conjuntos de datos grandes y complejos. También busca definir la 
# estructura del espacio topológico y la representación adecuada de este espacio topológico en
# el espacio de menor dimensión, resultando efectivo para diversos tipos de datos.
# 
# Las desventajas que presenta este modelo es la elección de los parámetros como el número de 
# vecinos cercanos (n_neighbours), tamaño del espacio de salida (n_componentes), distancia 
# mínima permitida entre puntos, distancia mínima permitida entre puntos (min_dist) y cantidad
# de puntos que están conectados fuertemente (local_connectivity), que pueden afectar 
# significativamente los resultados, lo que requiere una cuidadosa selección y ajuste.

set.seed(2003) 
umap.results <- umap(X_scaled, n_neighbors=0.3 * nrow(X_scaled),
                     n_components = 2, min_dist = 0.2, local_connectivity=3, 
                     ret_model = TRUE, verbose = TRUE) # Realizamos UMAP

umap.df <- data.frame(umap.results$embedding) # Convertimos a data frame
m_dist <- dist(umap.df) # Matriz de distancias basada en UMAP

# Graficamos
ggplot(umap.df, aes(x = X1, y = X2, color = genes_fil$clase)) +
  geom_point(size = 2, alpha = 0.85) +
  scale_color_manual(values = c("red", "blue", "green", "orange", "purple")) +
  labs(title = "UMAP", x = "X1", y = "X2", color = "clase") +
  theme_classic() +
  theme(plot.title=element_text(hjust=0.5))

# Al igula que con el modelo anterior podemos ver como las diferentes clases del dataset 
# se encuentran bien diferenciadas.Por tanto, consideramos que UMAP ha sido efectivo para 
# reducir la dimensionalidad de los datos y preservar las relaciones importantes entre 
# las muestras.

### CLUSTERIZACIÓN
##-----------------------------------------
##  3. Clustering no jerárguico - K-means  
##-----------------------------------------
set.seed(2003) 
kms <- kmeans(X_scaled, centers = 5) # Realizamos k-means con 5 clusters
table(kms$cluster) # Tamaño de cada cluster
table(kms$cluster, genes_fil$clase) # Tabla de contingencia clusters vs clases reales

umap.k <- umap(X_scaled, n_neighbors=0.3 * nrow(X_scaled),
                     n_components = 2, min_dist = 0.2, local_connectivity=3, 
                     ret_model = TRUE, verbose = TRUE) # Realizamos UMAP

# Data frame con resultados UMAP y clusters k-means
df_umap <- data.frame(umap.k$embedding, TrueClass = genes_fil$clase, 
                      Cluster = as.factor(kms$cluster))

# Graficamos
ggplot(df_umap, aes(x = X1, y = X2, color = genes_fil$clase)) +
  geom_point(size = 2, alpha = 0.85) +
  scale_color_manual(values = c("red", "blue", "green", "orange", "purple")) +
  labs(title = "Clustering no jerárquico con k-means (UMAP)", x = "X1", y = "X2", color = "clase") +
  theme_minimal() +
  theme(plot.title=element_text(hjust=0.5))

##-----------------------------------------
##  4. Clustering jerárquico - Dendograma  
##-----------------------------------------
# Seleccionamos este método de clustering jerárquico porque nos permite visualizar las 
# relaciones entre las muestras en forma de árbol invertido, lo que facilita la identificación 
# de grupos, sin la necesidad de especificar previamente el número de grupos, lo cual es 
# especialmente útil en el contexto de datos biológicos de alta dimensionalidad.
#
# Una de las principales ventajas del clustering jerárquico es que ofrece diferentes métodos 
# de enlace, como single, complete, average y Ward, lo que permite organizar los clústeres 
# de distintas maneras según el criterio de similitud utilizado. Esto brinda mayor flexibilidad 
# para identificar patrones y subestructuras en los datos.
# 
# Sin embargo, una de las desventajas del clustering jerárquico es su costo computacional, 
# ya que puede volverse poco eficiente para conjuntos de datos muy grandes. Además, el método 
# es sensible a la métrica de distancia y al método de enlace seleccionado.

set.seed(2003) 
dist_matrix <- dist(umap.df) # Matriz de distancias basada en UMAP

# Realizar el clustering con diferentes metodos de agrupamiento
hclust_single <- hclust(dist_matrix, method = "single")
hclust_complete <- hclust(dist_matrix, method = "complete")
hclust_average <- hclust(dist_matrix, method = "average")
hclust_ward <- hclust(dist_matrix, method = "ward.D2")

colores <- c("red", "blue", "green", "orange", "purple") # Definimos colores

# single: conecta puntos cercanos y puede encontrar clusters largos y delgados, 
#         PERO a veces conecta muchos puntos formando cadenas, lo que puede ser
#         poco útil
clust_single <- fviz_dend(hclust_single, 
                          cex = 0.5,
                          k = 5,
                          palette = colores,
                          main = "Single",
                          xlab = "Índice de Observaciones",
                          ylab = "Distancia") + theme_classic() +
  theme_classic()

# complete: crea clusters compactos y bien definidos, PERO es sensible a puntos 
#           extremos (outliers), que pueden distorsionar los clusters
clust_complete <- fviz_dend(hclust_complete, 
                            cex = 0.5,
                            k = 5,
                            palette = colores,
                            main = "Complete",
                            xlab = "Índice de Observaciones",
                            ylab = "Distancia") + 
  theme_classic()

# average:  encuentra un equilibrio entre single y complete, PERO a veces no es 
#           tan bueno para clusters con tamaños muy diferentes
clust_average <- fviz_dend(hclust_average, 
                           cex = 0.5,
                           k = 5,
                           palette = colores,
                           main = "Average",
                           xlab = "Índice de Observaciones",
                           ylab = "Distancia") + 
  theme_classic()

# Ward: crea clusters redondeados y homogéneos similares a los que genera 
#       k-means, PERO no funciona tan bien si los clusters tienen formas raras 
#       o tamaños muy diferentes
clust_ward <- fviz_dend(hclust_ward, 
                        cex = 0.5,
                        k = 5,
                        palette = colores,
                        main = "Ward",
                        xlab = "Índice de Observaciones",
                        ylab = "Distancia") + 
  theme_classic()

grid.arrange(clust_single, clust_complete, clust_average, clust_ward,
             nrow = 2,
             top = "Dendogramas con diferentes métodos de agrupamiento")

###########################################################################
###########################################################################
###                                                                     ###
###                      APENDIZAJE NO SUPERVIZADO                      ###
###                                                                     ###
###########################################################################
###########################################################################

# Preparación de los datos
set.seed(2003)
x <- as.matrix(X_scaled)
y <- factor(genes_fil$clase)

# Ajuste del modelo LASSO con validación cruzada
lasso_model <- cv.glmnet(
  x,
  y,
  family = "multinomial", # Clasificación multiclase (5 clases) 
  alpha = 1 # Penalización L1 (LASSO)
)

# Extracción de los coeficientes asociados al lambda óptimo 
selected_genes <- coef(lasso_model, s = "lambda.min") 

# Conversión a data frame
coef_matrices <- lapply(selected_genes, as.matrix) # Convertir cada clase a matriz
coef_all <- do.call(cbind, coef_matrices) # Combinar todas las matrices en una sola
selected_genes_df <- as.data.frame(coef_all) # Convertir a data frame

# Identificación de coeficientes distintos de cero
non_zero_indices <- selected_genes_df[
  selected_genes_df[,1] != 0,
  ,
  drop = FALSE
]

dim(non_zero_indices) # Dimensión del data frame de coeficientes no nulos
non_zero_indices # Coeficientes seleccionados 

# Extracción de los nombres de las variables seleccionados (-intercepto)
names <- rownames(selected_genes_df)[
  selected_genes_df$lambda.min != 0 &
    rownames(selected_genes_df) != "(Intercept)"
]

length(names) # Número total de variables seleccionadas
names # Variables seleccionadas por LASSO

# Construcción del nuevo data frame reducido
data <- genes_fil %>%
  dplyr::select(clase, names)

# Asignación de nombres de fila (muestra)
rows <- genes_fil$sample 
rownames(data) <- rows
str(data)

# Reordenación de columnas
names <- colnames(data)[-1]
data <- data %>% dplyr::select(names, clase)

# División del conjunto de datos en entrenamiento y prueba
# Creamos partición del 80% entrenamiento y 20% prueba
trainIndex <- createDataPartition(
  genes_fil$clase,
  p = 0.8,
  list = FALSE
)

# Nos aseguramos de que la variable respuesta sea un factor
genes_fil$clase <- as.factor(genes_fil$clase)

# Generamos los conjuntos de entrenamiento y prueba
trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]

##---------------------------------
##  1. k-Nearest Neighbours (kNN)  
##---------------------------------

# Knn fue elegido para procesar este conjunto de datos, teniendo en cuenta que la expresion 
# genica consta de datos no lineales y de alta dimensionalidad, caracteristicas que este 
# algoritmo no asume, como tampoco impone relaciones lineales, sino que clasifica segun 
# la proximidad en el espacio de expresión, basandose en el agrupamiento que presentan las 
# muestras del conjunto de datos, en el que segun el tipo de tumor, presentan niveles de
# expresiín similar.

knnModel <- train(
  clase ~ .,
  data = trainData,
  method = "knn", # Algoritmo kNN
  trControl = trainControl(method = "cv", number = 10), # Validación cruzada de 10 folds
  preProcess = c("center", "scale"), # Estandarización de los datos
  tuneLength = 30
)
knnModel

# Gráfica del rendimiento del modelo
plot(knnModel)
# con el cross validation del modelo, se puedo observar que el numero de vecinos optimo es 11



# Predicciones de clase en el conjunto de prueba
predictions_knn <- predict(knnModel, newdata = testData)
predictions_knn

# Definimos como factor la variable clase de testData
testData$clase <- factor(testData$clase)

# Matriz de confusión
confusionMatrix(predictions_knn, testData$clase)
table(testData$clase)

# Probabilidades predichas para cada clase
probabilities_knn <- predict(knnModel, newdata = testData, type = "prob")
probabilities_knn

# Curva PR
# Calculos para la curva
clases <- levels(testData$clase)

pr_list <- lapply(clases, function(cl) {
  pr.curve(
    scores.class0  = probabilities_knn[, cl],
    weights.class0 = testData$clase == cl,
    curve = TRUE
  )
})

# Extraemos AUPRC por clase
auc_pr <- sapply(pr_list, function(x) x$auc.integral)

# Definimos colores (uno por clase)
names(colores) <- clases

# Graficamos la primera curva
plot(
  pr_list[[1]],
  col  = colores[clases[1]],
  lwd  = 2, 
  main = "Curvas Precision–Recall (kNN, one-vs-rest)",
  legacy.axes = TRUE,
  auc.main = FALSE
)

# Añadimos el resto de curvas
for (i in 2:length(pr_list)) {
  plot(
    pr_list[[i]],
    col = colores[clases[i]],
    lwd = 2,
    add = TRUE
  )
}

# Leyenda con AUPRC
legend(
  "bottomleft",
  legend = paste0(
    clases,
    " (AUPRC = ",
    round(auc_pr, 3),
    ")"
  ),
  col = colores,
  lwd = 2,
  bty = "n"
)
# Las curvas PR se calcularon mediante la estrategia uno versus el resto, es decir en cada curva
# se considera la clase como positiva frente a las demás, a manera generar se observa un buen 
# rendimiento, con valores AUPRC mayores a 0.86, donde se resalta que la clase AGH obtuvo una
# separación completa, sin falsos positivos ni falsos negativos, posiblemente por caracteristicas
# o patrones muy distintos al resto, la clase HPB presentó un desempeño inferior al resto, 
# es decir que existe una mayor dificultad a la hora de su discriminación, posiblemente por
# similitudes biologicas con las otras clases.

# Curva ROC
clases <- levels(testData$clase)

# Colores (uno por clase)
names(colores) <- clases

# Calcular curvas ROC (one-vs-rest)
roc_list <- lapply(clases, function(cl) {
  roc(
    response = testData$clase == cl,   
    predictor = probabilities_knn[, cl], # probabilidad clase cl
    quiet = TRUE
  )
})

# Graficamos la primera curva
plot(
  roc_list[[1]],
  col  = colores[clases[1]],
  lwd  = 2,
  main = "Curvas ROC (kNN, one-vs-rest)",
  legacy.axes = TRUE
)

# Añadimos el resto
for (i in 2:length(roc_list)) {
  plot(
    roc_list[[i]],
    col = colores[clases[i]],
    lwd = 2,
    add = TRUE
  )
}

# Línea diagonal (clasificador aleatorio)
abline(a = 0, b = 1, lty = 2, col = "gray")

# calculo de AUC y grafica
auc_roc <- sapply(roc_list, auc)

# Leyenda con AUPRC
legend(
  "bottomright",
  legend = paste0(
    clases,
    " (AUC = ",
    round(auc_roc, 3),
    ")"
  ),
  col = colores,
  lwd = 2,
  bty = "n"
)

# Las curvas ROC tambien fueron calculadas independientemente para cada clase, considerando
# cada clase como positiva frente a las demas, se evidencia un buen rendimiento general, con 
# AUC mayores a 0.974, lo que se considera un excelente rendimiento, logrando una buena 
# discriminación de los datos, resalta la clase AGH por su rendimiento perfecto, asociado 
# posiblementre a caracteristicas biologicas diferenciada a las demas clases.

##-----------------------
##  2. Naive Bayes (NB)  
##-----------------------
nb_model <- train(
  clase ~ .,
  data = trainData,
  method = "nb", # Algoritmo Naive Bayes
  trControl = trainControl(method = "cv", number = 10) # Validación cruzada de 10 folds
)
nb_model

# Predicción de clases del modelo
predictions_nb <- predict(nb_model, newdata = testData)
predictions_nb

### Evaluación mediante matriz de confusión
confusionMatrix(predictions_nb, testData$clase)

# Obtención de probabilidades de clase
probabilities_nb <- predict(nb_model, newdata = testData, type = "prob")
probabilities_nb

# Curva PR
clases <- levels(testData$clase)

pr_list2 <- lapply(clases, function(cl) {
  pr.curve(
    scores.class0  = probabilities_nb[, cl],
    weights.class0 = testData$clase == cl,
    curve = TRUE
  )
})

# Extraemos AUPRC por clase
auc_pr <- sapply(pr_list2, function(x) x$auc.integral)

# Definimos colores (uno por clase)
names(colores) <- clases

# Graficamos la primera curva
plot(
  pr_list2[[1]],
  col  = colores[clases[1]],
  lwd  = 2, 
  main = "Curvas Precision–Recall (Naive Bayes)",
  legacy.axes = TRUE,
  auc.main = FALSE
)

# Añadimos el resto de curvas
for (i in 2:length(pr_list2)) {
  plot(
    pr_list2[[i]],
    col = colores[clases[i]],
    lwd = 2,
    add = TRUE
  )
}

# Leyenda con AUPRC
legend(
  "bottomleft",
  legend = paste0(
    clases,
    " (AUPRC = ",
    round(auc_pr, 3),
    ")"
  ),
  col = colores,
  lwd = 2,
  bty = "n"
)

# Curva ROC
clases <- levels(testData$clase)

# Colores (uno por clase)
names(colores) <- clases

# Calcular curvas ROC (one-vs-rest)
roc_list2 <- lapply(clases, function(cl) {
  roc(
    response = testData$clase == cl,   
    predictor = probabilities_nb[, cl], # probabilidad clase cl
    quiet = TRUE
  )
})

# Graficamos la primera curva
plot(
  roc_list2[[1]],
  col  = colores[clases[1]],
  lwd  = 2,
  main = "Curvas ROC (Naive Bayes)",
  legacy.axes = TRUE
)

# Añadimos el resto
for (i in 2:length(roc_list2)) {
  plot(
    roc_list2[[i]],
    col = colores[clases[i]],
    lwd = 2,
    add = TRUE
  )
}

# Línea diagonal (clasificador aleatorio)
abline(a = 0, b = 1, lty = 2, col = "gray")

# calculo de AUC y grafica
auc_roc2 <- sapply(roc_list2, auc)

# Leyenda con AUPRC
legend(
  "bottomright",
  legend = paste0(
    clases,
    " (AUC = ",
    round(auc_roc2, 3),
    ")"
  ),
  col = colores,
  lwd = 2,
  bty = "n"
)

##--------------------------------------------
##  3. Máquinas de Vectores de Soporte (SVM)  
##--------------------------------------------
svmModelLineal <- train(
  clase ~ .,
  data = trainData,
  method = "svmLinear", # Algoritmo SVM con kernel lineal 
  trControl = trainControl(method = "cv", number = 10, classProbs = TRUE),
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    C = seq(0.01, 2, length.out = 20)
  )
)

svmModelLineal
plot(svmModelLineal)

## ===============================================================
##  Predicciones y matriz de confusión
## ===============================================================
predictions_svm <- predict(svmModelLineal, newdata = testData)
predictions_svm

### Evaluación del rendimiento mediante matriz de confusión
### Permite calcular Accuracy, Sensitivity y Specificity
confusionMatrix(predictions_svm, testData$clase)

### Predicción de probabilidades de clase en el conjunto de prueba
### Estas probabilidades se utilizan posteriormente para construir curvas ROC
probabilities_svm_linear <- predict(
  svmModelLineal,
  newdata = testData,
  type = "prob"
)

### Visualización de las probabilidades predichas
probabilities_svm_linear

## ---------------------------------------------------------
### Cálculo de curvas PR para cada modelo
### ---------------------------------------------------------

clases <- levels(testData$clase)

pr_list_svm <- lapply(clases, function(cl) {
  pr.curve(
    scores.class0  = probabilities_svm_linear[, cl],
    weights.class0 = testData$clase == cl,
    curve = TRUE
  )
})

### ---------------------------------------------------------
### Representación gráfica conjunta de la curva PR
### ---------------------------------------------------------

library(PRROC)

# Extraemos AUPRC por clase
auc_pr_svm <- sapply(pr_list_svm, function(x) x$auc.integral)

# Definimos colores (uno por clase)
colores <- c("red", "blue", "green", "orange", "purple")
names(colores) <- clases

# Graficamos la primera curva
plot(
  pr_list_svm[[1]],
  col  = colores[clases[1]],
  lwd  = 2, 
  main = "Curvas Precision–Recall (SVM, one-vs-rest)",
  legacy.axes = TRUE,
  auc.main = FALSE
)

# Añadimos el resto de curvas
for (i in 2:length(pr_list_svm)) {
  plot(
    pr_list_svm[[i]],
    col = colores[clases[i]],
    lwd = 2,
    add = TRUE
  )
}

# Leyenda CON AUPRC
legend(
  "bottomleft",
  legend = paste0(
    clases,
    " (AUPRC = ",
    round(auc_pr_svm, 3),
    ")"
  ),
  col = colores,
  lwd = 2,
  bty = "n"
)

# Curva ROC

library(pROC)

# Clases
clases <- levels(testData$clase)

# Colores (uno por clase)
colores <- c("red", "blue", "green", "orange", "purple")
names(colores) <- clases

# Calcular curvas ROC (one-vs-rest)
roc_list_svm <- lapply(clases, function(cl) {
  roc(
    response = testData$clase == cl,     # TRUE/FALSE
    predictor = probabilities_svm_linear[, cl], # probabilidad clase cl
    quiet = TRUE
  )
})

# Graficar curvas ROC
# Graficamos la primera curva
plot(
  roc_list_svm[[1]],
  col  = colores[clases[1]],
  lwd  = 2,
  main = "Curvas ROC (SVM, one-vs-rest)",
  legacy.axes = TRUE
)

# Añadimos el resto
for (i in 2:length(roc_list_svm)) {
  plot(
    roc_list_svm[[i]],
    col = colores[clases[i]],
    lwd = 2,
    add = TRUE
  )
}

# Línea diagonal (clasificador aleatorio)
abline(a = 0, b = 1, lty = 2, col = "gray")

# calculo de AUC y grafica
auc_roc_svm <- sapply(roc_list_svm, auc)

legend(
  "bottomright",
  legend = paste0(
    clases,
    " (AUC = ",
    round(auc_roc_svm, 3),
    ")"
  ),
  col = colores,
  lwd = 2,
  bty = "n"
)










