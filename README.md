# TP Integrador – Econometría y Machine Learning

## Descripción

Este repositorio contiene el trabajo práctico integrador realizado para la materia. El objetivo es analizar y predecir el valor medio de las viviendas utilizando distintas técnicas de regresión y comparar su desempeño predictivo.

## Dataset

Se utiliza el **California Housing Dataset**, que contiene información socioeconómica y habitacional de distintos distritos de California.

La variable objetivo es el **valor medio de las viviendas** (`median_house_value`).

## Técnicas aplicadas

Se estiman y comparan distintos modelos de regresión:

* Regresión lineal por Mínimos Cuadrados Ordinarios (OLS).
* Ridge Regression.
* LASSO.
* Elastic Net.

El desempeño de los modelos se evalúa mediante métricas de error predictivo, incluyendo **RMSE**, **MAE** y **R²**.

## Archivos

* `TP_Integrador.R`: script completo para reproducir el análisis.
* `Reporte.pdf`: informe final con la metodología, resultados, gráficos e interpretación.

## Cómo ejecutar el proyecto

1. Clonar o descargar este repositorio.
2. Abrir `TP_Integrador.R` en RStudio.
3. Instalar los paquetes necesarios indicados en el script, en caso de no tenerlos instalados.
4. Ejecutar el script desde el inicio hasta el final.

El script contiene las instrucciones necesarias para realizar el análisis y utiliza una semilla fija (`set.seed()`) cuando corresponde, permitiendo reproducir los resultados.
