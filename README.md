# Construcción de un clasificador automático de reportes sobre la situación de Adultos Mayores en el Municipio de Vicente López

## Objetivos
- Entrenar un modelo que permita automatizar el taggeo de reportes en texto crudo elaborado por voluntarios del Municipio de Vicente López
- Diseñar una metodología de evaluación continua y diagnóstico periódico del modelo

## Estructura del repositorio

- `/data/`: reportes con texto crudo y taggeos manuales
- `/data/proc`: archivos procesados y que son subproductos e insumo de otros procesos
- `/fasttext`: archivos procesados y formateados específicamente para entrenar fasttext
- `/fasttext/src/`: código
- `/fasttext/models`: objetos serializados de cada modelo
- `/fasttext/notebooks`: notebooks con evaluaciones de modelos
