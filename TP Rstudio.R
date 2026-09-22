########## Trabajo Práctico Integrador - Módulo R ##########

##### Parte 1: Elección del dataset y la técnica #####

# Para este trabajo utilizaremos el dataset California Housing Prices, 
# disponible en Kaggle. El conjunto de datos contiene información sobre 
# distintos distritos de California, incluyendo variables relacionadas con las
# características de las viviendas, la población y el entorno socioeconómico.

# La pregunta que me interesa contestar es:
# ¿En qué medida las características de las viviendas y de su entorno permiten
# explicar y predecir el valor mediano de las viviendas en los distintos
# distritos de California?

# Elijo este dataset porque presenta un número considerable de observaciones
# y múltiples variables explicativas, lo que permite realizar un análisis
# exploratorio genuino y estudiar las relaciones entre las características de
# los distritos y el valor de las viviendas. Además, la diversidad de variables
# permite analizar posibles relaciones entre predictores y explorar la
# presencia de correlaciones y otros problemas que pueden afectar a una
# regresión lineal.

### Paquetes a utilizar

library(tidyverse)
library(ggplot2)
library(glmnet)

### Importamos el archivo en formato ".csv"

data <- read_csv("data/housing.csv")

### Variables

# 1) longitude = longitud geográfica del distrito.
# 2) latitude = latitud geográfica del distrito.
# 3) housing_median_age: edad mediana de las viviendas del distrito.
# 4) total_rooms: cantidad total de habitaciones en el distrito.
# 5) total_bedrooms: cantidad total de dormitorios en el distrito.
# 6) population: población total del distrito.
# 7) households: cantidad total de hogares del distrito.
# 8) median_income: ingreso mediano de los hogares del distrito.
# 9) median_house_value: valor mediano de las viviendas del distrito.
# 10) ocean_proximity: categoría que indica la proximidad del distrito al océano.

## Técnicas seleccionadas

# La técnica principal utilizada en este trabajo será la regresión lineal,
# complementada con métodos de regularización Ridge y LASSO.
# La regresión lineal establecerá un buen punto de partida,mientras que Ridge y
# LASSO permitirán evaluar alternativas orientadas a mejorar la capacidad 
# predictiva y, en el caso de LASSO, realizar selección de variables.

##### Parte 2: Análisis exploratorio de datos (EDA) #####

## Estructura General: ¿Qué tienen los datos?

glimpse(data)

sapply(data, class)

# Rta: 9 variables numéricas y una variable almacenada como texto.

# Cantidad de observaciones
nrow(data)

# Cantidad de variables
ncol(data)

## Valores Faltantes

# Evaluamos si tenemos valores faltantes
colSums(is.na(data))

# Rta: Como podemos observar, la única variable que contiene información NA es 
# la variable "total_bedrooms" que contiene 207 observaciones vacías.

na_prop <- colMeans(is.na(data))

na_prop

# Como podemos ver, las observaciones faltantes de la variable "total_bedrooms"
# representan aproximadamente el 1% del total de observaciones.

# Dado que los valores faltantes representan aproximadamente el 1% de las
# observaciones, se removió dichas observaciones para trabajar con una
# muestra completa, sin una pérdida significativa de información.

data2 <- na.omit(data) # Dataset limpio de N/A

# Verificamos que efectivamente hayamos limpiado los valores faltantes

colSums(is.na(data2))

## Distribuciones

data_long <- data2 %>%
  select(
    median_house_value,
    median_income,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "value"
  ) %>%
  mutate(
    value = if_else(
      variable == "median_house_value",
      value / 1000,
      value
    )
  )

ggplot(data_long, aes(x = value)) +
  geom_histogram(
    bins = 40,
    fill = "steelblue",
    color = "white"
  ) +
  facet_wrap(
    ~ variable,
    scales = "free",
    ncol = 2
  ) +
  labs(
    title = "Distribución de las principales variables",
    subtitle = "El valor de las viviendas se expresa en miles de USD",
    x = NULL,
    y = "Cantidad de distritos"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 11),
    strip.text = element_text(size = 11, face = "bold"),
    panel.grid.minor = element_blank()
  )

## Variable median_house_value

# En el gráfico de la variable median_house_value se observa que hay un pico 
# para los valores cercanos a USD 500.000.


# Cuantificamos cuántos casos toman exactamente el valor de USD 500.000.

sum(data2$median_house_value == 500000)

mean(data2$median_house_value == 500000)

# Podemos observar que los 27 valores encontrados representan un 0,132% del
# total de las observaciones.

# Investigando un poco los valores máximos de la variable, tendremos:

sort(unique(data2$median_house_value), decreasing = TRUE)[1:20]

# Observamos que existe un valor de USD 500.001, por lo que USD 500.000
# no constituye el máximo de la variable.

data2 %>%
  filter(median_house_value >= 499000) %>%
  count(median_house_value, sort = TRUE)

# Resulta particularmente llamativo que 958 observaciones tomen exactamente
# el valor de USD 500.001, mientras que solo 27 observaciones toman el valor
# de USD 500.000. Esto sugiere la existencia de un límite superior en el
# registro de la variable median_house_value.

# Visualmente, podemos observar esta concentración en:

ggplot(data2, aes(x = median_house_value)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  coord_cartesian(xlim = c(400000, 500001)) +
  labs(
    title = "Distribución de los valores más altos de las viviendas",
    x = "Valor mediano de la vivienda (USD)",
    y = "Cantidad de distritos"
  ) +
  theme_minimal()

## Total_rooms

summary(data2$total_rooms)

quantile(
  data2$total_rooms,
  probs = c(0.01, 0.25, 0.5, 0.75, 0.99)
)

sort(data2$total_rooms, decreasing = TRUE)[1:20]

data2 %>%
  arrange(desc(total_rooms)) %>%
  select(
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    median_house_value,
    ocean_proximity
  ) %>%
  slice_head(n = 10)

# La variable total_rooms presenta una marcada asimetría positiva.
# La mediana es de 2.127 habitaciones en el distrito, mientras que la media 
# alcanza 2.636. El percentil 99 se ubica en 11.244 habitaciones y el máximo
# alcanza las 39.320.

# Al analizar las observaciones con mayor cantidad de habitaciones,
# se observa que también presentan valores elevados de población,
# hogares y cantidad de dormitorios. Por lo tanto, los valores extremos
# parecen corresponder a distritos de gran tamaño y no se identifican
# como errores evidentes de registro.

# En consecuencia, no consideramos necesario realizar modificaciones sobre
# estas observaciones.

## Total_bedrooms

summary(data2$total_bedrooms)

quantile(
  data2$total_bedrooms,
  probs = c(0.01, 0.25, 0.5, 0.75, 0.99)
)

sort(data2$total_bedrooms, decreasing = TRUE)[1:20]

data2 %>%
  arrange(desc(total_bedrooms)) %>%
  select(
    total_bedrooms,
    total_rooms,
    population,
    households,
    median_income,
    median_house_value,
    ocean_proximity
  ) %>%
  slice_head(n = 10)

# La variable total_bedrooms presenta una marcada asimetría positiva.
# La mediana es de 435 dormitorios en el distrito, mientras que la media
# alcanza 538. El percentil 99 se ubica en 2.221 dormitorios y el máximo
# alcanza los 6.445.

# Al analizar las observaciones con mayor cantidad de dormitorios,
# se observa que también presentan valores elevados de habitaciones,
# población y hogares. Por lo tanto, los valores extremos parecen
# corresponder a distritos de gran tamaño y no se identifican como
# errores evidentes de registro.

# En consecuencia, no consideramos necesario realizar modificaciones
# sobre estas observaciones.

## Population

summary(data2$population)

quantile(
  data2$population,
  probs = c(0.01, 0.25, 0.5, 0.75, 0.99)
)

sort(data2$population, decreasing = TRUE)[1:20]

data2 %>%
  arrange(desc(population)) %>%
  select(
    population,
    households,
    total_rooms,
    total_bedrooms,
    median_income,
    median_house_value,
    ocean_proximity
  ) %>%
  slice_head(n = 10)

# La variable population presenta una marcada asimetría positiva.
# La mediana es de 1.166 habitantes por distrito, mientras que la media
# alcanza 1.425. El percentil 99 se ubica en 5.809 habitantes y el máximo
# alcanza los 35.682.

# Al analizar las observaciones con mayor población, se observa que estas
# presentan valores elevados de hogares, habitaciones y dormitorios.
# Esto sugiere que los valores extremos corresponden a distritos de gran
# tamaño por lo que no los identificamos como errores evidentes de registro.

# En consecuencia, no consideramos necesario realizar modificaciones
# sobre estas observaciones.

## Households

summary(data2$households)

quantile(
  data2$households,
  probs = c(0.01, 0.25, 0.5, 0.75, 0.99)
)

sort(data2$households, decreasing = TRUE)[1:20]

# La variable households presenta una marcada asimetría positiva.
# La mediana es de 409 hogares por distrito, mientras que la media
# alcanza 499. El percentil 99 se ubica en 1.986 hogares y el máximo
# alcanza los 6.082.

# Los valores extremos resultan consistentes con los observados en las
# variables population, total_rooms y total_bedrooms, ya que corresponden
# a distritos con una mayor cantidad de habitantes y viviendas.

# En consecuencia, no consideramos necesario realizar modificaciones
# sobre estas observaciones.

## Median_income

summary(data2$median_income)

quantile(
  data2$median_income,
  probs = c(0.01, 0.25, 0.5, 0.75, 0.99)
)

sort(data2$median_income, decreasing = TRUE)[1:20]

data2 %>%
  arrange(desc(median_income)) %>%
  select(
    median_income,
    median_house_value,
    total_rooms,
    total_bedrooms,
    population,
    households,
    ocean_proximity
  ) %>%
  slice_head(n = 10)

sum(data2$median_income == 15.0001)

mean(data2$median_income == 15.0001)

data2 %>%
  filter(median_income == 15.0001) %>%
  count(median_house_value, sort = TRUE)

# La variable median_income presenta una marcada asimetría positiva.
# La mediana es de 3,54, mientras que la media alcanza 3,87.
# El percentil 99 se ubica en 10,60 y el máximo alcanza 15,0001.

# El valor máximo de 15,0001 se repite en 48 observaciones,
# equivalentes aproximadamente al 0,24% de la muestra. La concentración
# exacta en este valor sugiere la existencia de un límite superior en
# el registro de la variable.

# Al analizar estas 48 observaciones, se observa que 45 de ellas
# también presentan el valor máximo registrado de median_house_value
# (500001), lo que representa el 93,75% de los casos.

# Este patrón resulta relevante para el análisis posterior, ya que
# muestra una concentración de observaciones en los valores superiores
# de ambas variables. No obstante, dado que no se identifican errores
# evidentes de registro, se mantienen estas observaciones en el dataset.

## Longitude y Latitude

ggplot(data2, aes(x = longitude, y = latitude)) +
  geom_point(alpha = 0.3, size = 1) +
  labs(
    title = "Distribución geográfica de los distritos",
    x = "Longitud",
    y = "Latitud"
  ) +
  theme_minimal()

## Ocean_proximity

table(data2$ocean_proximity)

prop.table(table(data2$ocean_proximity))

ggplot(data2, aes(
  x = ocean_proximity,
  y = median_house_value
)) +
  geom_boxplot() +
  labs(
    title = "Valor de las viviendas según proximidad al océano",
    x = "Proximidad al océano",
    y = "Valor mediano de la vivienda (USD)"
  ) +
  theme_minimal()

# El gráfico muestra diferencias importantes en la distribución del valor
# de las viviendas según la categoría de proximidad al océano.

# Los distritos clasificados como INLAND presentan, en general, valores
# de las viviendas inferiores a los observados en las restantes categorías.
# Por otro lado, las categorías NEAR BAY y NEAR OCEAN presentan valores
# centrales más elevados.

# La categoría ISLAND presenta valores elevados, aunque debe interpretarse
# con cautela debido a que cuenta únicamente con 5 observaciones.

# Estas diferencias sugieren que la ubicación respecto del océano puede
# ser una variable relevante para explicar el valor de las viviendas y,
# por lo tanto, será incorporada al análisis posterior.

### Relaciones bivariadas

## Median_income vs Median_house_value

ggplot(data2, aes(
  x = median_income,
  y = median_house_value
)) +
  geom_point(alpha = 0.3) +
  geom_smooth(
    method = "lm",
    se = TRUE
  ) +
  scale_y_continuous(
    labels = scales::label_number(big.mark = ".")
  ) +
  labs(
    title = "Relación entre ingreso mediano y valor de la vivienda",
    x = "Ingreso mediano",
    y = "Valor mediano de la vivienda (USD)"
  ) +
  theme_minimal()

# El gráfico muestra una relación positiva entre median_income y
# median_house_value: en general, los distritos con mayores ingresos
# presentan mayores valores medianos de las viviendas.

# Sin embargo, la relación presenta mucha dispersión. No parece ajustarse
# perfectamente a una relación lineal. Además, se observa una concentración
# importante de observaciones en el valor máximo registrado de
# median_house_value (USD 500.001), que ya había sido identificada en el
# análisis univariado.

# La recta de regresión estimada continúa creciendo por encima de este valor.
# Sin embargo, no existen observaciones de median_house_value por encima
# de USD 500.001, por lo que esta parte de la recta corresponde a una
# extrapolación y no será considerada para interpretar la relación observada.

## Total_rooms vs Median_house_value

ggplot(data2, aes(
  x = total_rooms,
  y = median_house_value
)) +
  geom_point(alpha = 0.3) +
  geom_smooth(
    method = "lm",
    se = TRUE
  ) +
  scale_y_continuous(
    labels = scales::label_number(big.mark = ".")
  ) +
  labs(
    title = "Relación entre cantidad de habitaciones y valor de la vivienda",
    x = "Cantidad total de habitaciones",
    y = "Valor mediano de la vivienda (USD)"
  ) +
  theme_minimal()

# El gráfico muestra una relación positiva entre total_rooms y
# median_house_value: en términos generales, los distritos con una mayor
# cantidad de habitaciones presentan mayores valores medianos de las viviendas.

# Sin embargo, se observa una elevada dispersión de los valores de la vivienda
# para niveles similares de total_rooms, lo que indica que la cantidad de
# habitaciones por sí sola no permite explicar completamente las diferencias
# observadas en el valor de las viviendas.

# Al igual que en el análisis de median_income, se observa una concentración
# importante de observaciones en el valor máximo registrado de
# median_house_value (USD 500.001).

cor_data <- data2 %>%
  select(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    median_house_value
  )

cor(cor_data)

# La matriz de correlaciones muestra una elevada asociación entre las
# variables relacionadas con el tamaño de los distritos. En particular,
# se observan correlaciones superiores a 0,85 entre total_rooms,
# total_bedrooms, population y households.

# También se observa una elevada correlación negativa entre longitude
# y latitude (-0.925), asociada a la estructura geográfica de las
# observaciones.

# Por otro lado, median_income presenta la mayor correlación lineal
# con median_house_value entre las variables numéricas (0.688),
# consistente con la relación positiva observada previamente.

# La correlación alta entre varios predictores constituye un aspecto
# relevante para la especificación del modelo y motiva el análisis
# posterior mediante métodos de regularización.

cor_matrix <- cor(cor_data)

cor_long <- as.data.frame(cor_matrix) %>%
  rownames_to_column("variable_1") %>%
  pivot_longer(
    cols = -variable_1,
    names_to = "variable_2",
    values_to = "correlation"
  )

variables <- colnames(cor_data)

cor_long <- cor_long %>%
  mutate(
    variable_1 = factor(variable_1, levels = variables),
    variable_2 = factor(variable_2, levels = variables)
  )

ggplot(cor_long, aes(
  x = variable_1,
  y = variable_2,
  fill = correlation
)) +
  geom_tile() +
  scale_y_discrete(limits = rev(variables)) +
  scale_fill_gradient2(
    low = "steelblue",
    mid = "white",
    high = "firebrick",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  labs(
    title = "Matriz de correlaciones entre variables numéricas",
    x = NULL,
    y = NULL,
    fill = "Correlación"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

##### Parte 3: Aplicación de la técnica e interpretación #####

### Modelo OLS de referencia

## Transformamos la variable ocean_proximity en una variable categórica

data2 <- data2 %>%
  mutate(
    ocean_proximity = factor(ocean_proximity)
  )

str(data2$ocean_proximity)

ols <- lm(
  median_house_value ~ .,
  data = data2
)

options(scipen = 999)

summary(ols)

## Comentarios

# 1) Todos los coeficientes son estadísticamente significativos
# 2) El R^2 del modelo estimado sobre la muestra completa es 0,6465,
# por lo que el modelo presenta un ajuste de 64,65% sobre esta muestra.
# 3) El R^2 ajustado (0,6463) es prácticamente igual al R^2 (0,6465),
# por lo que la corrección por el número de variables incluidas tiene
# un efecto muy pequeño sobre el ajuste del modelo.

## ¿Por qué no alcanza con OLS?

# La matriz de correlaciones mostró una elevada asociación entre varias
# variables explicativas, particularmente entre aquellas relacionadas con
# el tamaño de los distritos.

# Si bien el modelo OLS presenta un R^2 de 0,6465, la presencia de
# predictores altamente correlacionados puede afectar la estabilidad e
# interpretación de los coeficientes individuales.

# Por este motivo, se incorporan métodos de regularización como Ridge y
# LASSO. Ridge incorpora una penalización que contrae los coeficientes,
# lo que puede ayudar a estabilizar su estimación cuando existen
# predictores altamente correlacionados.

# LASSO incorpora una penalización similar, pero además puede llevar
# algunos coeficientes exactamente a cero, permitiendo seleccionar
# variables.

## División de la muestra en train y test

set.seed(1234)

n <- nrow(data2)

train_id <- sample(
  seq_len(n),
  size = floor(0.8 * n)
)

train <- data2[train_id, ]
test <- data2[-train_id, ]

nrow(train)
nrow(test)

# Armamos las matrices train

x_train <- model.matrix(
  median_house_value ~ .,
  data = train
)[, -1]

y_train <- train$median_house_value

# Armamos las matrices test

x_test <- model.matrix(
  median_house_value ~ .,
  data = test
)[, -1]

y_test <- test$median_house_value

# Dimensiones de las matrices

dim(x_train)
dim(x_test)

## Estimación Ridge

# glmnet permite estimar una familia de modelos que combina las penalizaciones
# Ridge y LASSO mediante el parámetro alpha.

# En glmnet, alpha = 0 corresponde a Ridge y alpha = 1 corresponde a LASSO.

# Por otra parte, dividimos las observaciones de train en 10 grupos
# (nfolds = 10) para realizar validación cruzada y seleccionar el valor
# de lambda que minimiza el error de predicción dentro del conjunto de
# entrenamiento.

lambda_grid <- exp(
  seq(
    log(10^8),
    log(100),
    length.out = 200
  )
)

ridge_cv <- cv.glmnet(
  x_train,
  y_train,
  alpha = 0,
  nfolds = 10,
  lambda = lambda_grid
)

plot(ridge_cv)

# Valor de lambda que minimiza el error cuadrático medio
# de validación cruzada.
ridge_cv$lambda.min

# Valor de lambda asociado a la regla de una desviación estándar.
ridge_cv$lambda.1se

# Rango de valores de lambda evaluados.
range(ridge_cv$lambda)

## Predicciones sobre el conjunto de prueba

pred_ridge <- predict(
  ridge_cv,
  newx = x_test,
  s = "lambda.min"
)

## Evaluación predictiva de Ridge

rmse_ridge <- sqrt(mean((y_test - pred_ridge)^2))

mae_ridge <- mean(abs(y_test - pred_ridge))

r2_ridge <- 1 - sum((y_test - pred_ridge)^2) /
  sum((y_test - mean(y_test))^2)

rmse_ridge
mae_ridge
r2_ridge

## Estimación OLS sobre train

ols_train <- lm(
  median_house_value ~ .,
  data = train
)

## Predicciones OLS sobre test

pred_ols <- predict(
  ols_train,
  newdata = test
)

## Evaluación predictiva de OLS

rmse_ols <- sqrt(mean((y_test - pred_ols)^2))

mae_ols <- mean(abs(y_test - pred_ols))

r2_ols <- 1 - sum((y_test - pred_ols)^2) /
  sum((y_test - mean(y_test))^2)

rmse_ols
mae_ols
r2_ols

## Estimación LASSO

lambda_grid_lasso <- exp(
  seq(
    log(max(lambda_grid)),
    log(1),
    length.out = 250
  )
)

# Reestimamos LASSO utilizando la nueva grilla y validación cruzada
# de 10 folds. alpha = 1 indica que estamos utilizando la penalización
# LASSO.

lasso_cv <- cv.glmnet(
  x_train,
  y_train,
  alpha = 1,
  nfolds = 10,
  lambda = lambda_grid_lasso
)

# Gráfico del error de validación cruzada para los distintos valores
# de lambda.

plot(lasso_cv)

# Lambda que minimiza el error de validación cruzada.

lasso_cv$lambda.min

# Lambda correspondiente a la regla de una desviación estándar.

lasso_cv$lambda.1se

# Verificamos el rango de valores de lambda utilizados.

range(lasso_cv$lambda)

## Predicciones LASSO y evaluación fuera de muestra

# Utilizamos lambda.min, es decir, el valor de lambda que minimiza
# el error de validación cruzada.
pred_lasso <- predict(
  lasso_cv,
  newx = x_test,
  s = "lambda.min"
)

# RMSE: mide el tamaño promedio de los errores de predicción,
# penalizando más los errores grandes.
rmse_lasso <- sqrt(mean((y_test - pred_lasso)^2))

# MAE: mide el error absoluto promedio y es más fácil de interpretar
# porque está expresado en las mismas unidades que el precio.
mae_lasso <- mean(abs(y_test - pred_lasso))

# R² fuera de muestra: indica qué proporción de la variabilidad
# observada en el conjunto de prueba es explicada por las predicciones.
r2_lasso <- 1 - sum((y_test - pred_lasso)^2) /
  sum((y_test - mean(y_test))^2)

rmse_lasso
mae_lasso
r2_lasso

coef(lasso_cv, s = "lambda.min")

## Comparación de la selección de variables entre lambda.min y lambda.1se

# Coeficientes utilizando lambda.min:
coef_min <- coef(lasso_cv, s = "lambda.min")

# Coeficientes utilizando lambda.1se:
coef_1se <- coef(lasso_cv, s = "lambda.1se")

# Mostramos ambos conjuntos de coeficientes para comparar
coef_min
coef_1se

# Cantidad de coeficientes distintos de cero
# (sin contar el intercepto)
sum(coef_min[-1, ] != 0)
sum(coef_1se[-1, ] != 0)

## Identificación de las variables seleccionadas por LASSO con lambda.1se

# Extraemos los coeficientes correspondientes a lambda.1se
coef_1se <- coef(lasso_cv, s = "lambda.1se")

# Mostramos únicamente los coeficientes que fueron llevados a cero
# por la penalización LASSO.
coef_1se[coef_1se[, 1] == 0, , drop = FALSE]

##### Comparación de modelos: RMSE #####

# Creamos una tabla con el RMSE de cada modelo.
# El RMSE permite comparar directamente el error de predicción
# sobre el mismo conjunto de test.

comparison <- data.frame(
  Modelo = c("OLS", "Ridge", "LASSO"),
  RMSE = c(rmse_ols, rmse_ridge, rmse_lasso)
)

# Gráfico de puntos.
# Se utiliza un punto en lugar de barras porque las diferencias entre
# los modelos son muy pequeñas y no queremos exagerarlas visualmente.

ggplot(comparison, aes(x = Modelo, y = RMSE)) +
  geom_point(size = 4) +
  geom_text(
    aes(label = round(RMSE, 0)),
    vjust = -1,
    size = 4
  ) +
  scale_y_continuous(
    limits = c(66900, 67050),
    breaks = seq(66900, 67050, by = 25)
  ) +
  labs(
    title = "Error de predicción fuera de muestra",
    subtitle = "RMSE sobre el conjunto de prueba",
    x = NULL,
    y = "RMSE (USD)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11),
    axis.text = element_text(size = 10),
    axis.title.y = element_text(size = 11)
  )

##### Diagnóstico de residuos: modelo OLS #####

# Calculamos los residuos del modelo OLS sobre el conjunto de prueba.
# El residuo es la diferencia entre el valor observado y el valor predicho.
ols_residuals <- y_test - pred_ols

residuals_data <- data.frame(
  fitted = as.numeric(pred_ols),
  residuals = as.numeric(ols_residuals)
)

# Gráfico de residuos versus valores predichos.
# Este gráfico permite identificar posibles patrones en los residuos
# y observar si su dispersión cambia a medida que aumenta el valor predicho.

ggplot(residuals_data, aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.25, size = 1.2) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Residuos vs. valores predichos",
    subtitle = "Modelo OLS sobre el conjunto de prueba",
    x = "Valor predicho de la vivienda (USD)",
    y = "Residuo (USD)"
  ) +
  theme_minimal()

# Los residuos se concentran principalmente alrededor de cero,
# aunque presentan una dispersión considerable.

# Para valores predichos elevados se observa una menor cantidad de
# observaciones y una concentración de residuos negativos.

# También se observa una estructura diagonal en la parte superior derecha,
# asociada a las observaciones cuyo valor de median_house_value alcanza
# el máximo registrado de USD 500.001.

# En conjunto, el gráfico muestra que los errores de predicción no se
# distribuyen de manera completamente homogénea a lo largo de los valores
# predichos.

comparison_final <- data.frame(
  Modelo = c("OLS", "Ridge", "LASSO"),
  RMSE = c(rmse_ols, rmse_ridge, rmse_lasso),
  MAE = c(mae_ols, mae_ridge, mae_lasso),
  R2 = c(r2_ols, r2_ridge, r2_lasso)
)

comparison_final