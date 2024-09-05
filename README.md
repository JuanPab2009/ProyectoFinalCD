# ProyectoFinalCD

En este repositorio vamos a incluir todo el progreso de la clase de Proyecto de Ciencia de Datos.

# Predicción de Resultados de Partidos de Fútbol en La Liga utilizando Regresión Logística

## Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar un modelo predictivo utilizando **regresión logística** para predecir el resultado de los partidos de fútbol de la **Liga Española (La Liga)**. Utilizando datos estadísticos obtenidos de la página **FBref**, el modelo intentará predecir si un equipo ganará, empatará o perderá un partido. El proyecto explora la capacidad de los algoritmos de **machine learning** para identificar patrones en los datos históricos y generar predicciones útiles para aficionados, analistas deportivos y casas de apuestas.

## Estructura del Proyecto

- **data/**: Este directorio contiene los conjuntos de datos de La Liga extraídos de FBref.
- **notebooks/**: Contiene los notebooks de Jupyter con el código para la limpieza de datos, exploración y entrenamiento del modelo de regresión logística.
- **models/**: Aquí se guardan los modelos entrenados.
- **results/**: Incluye las evaluaciones del modelo, como las matrices de confusión y los reportes de clasificación.
- **README.md**: Archivo de documentación del proyecto.
- **requirements.txt**: Archivo con las dependencias necesarias para reproducir el entorno de desarrollo.
  
## Datos
Los datos se obtienen desde la página [FBref](https://fbref.com/en/comps/12/La-Liga-Stats) y se refieren a las estadísticas de la temporada actual de La Liga. Los datos incluyen:
- Goles a favor y en contra
- Posesión del balón
- Tiros a puerta
- Estadísticas defensivas y ofensivas de los equipos

### Instalación
Para poder ejecutar el proyecto en tu entorno local, sigue los siguientes pasos:

1. Clona este repositorio:
   ```bash
   git clone https://github.com/usuario/repositorio_prediccion_laliga.git
    ```
### Instala las dependencias necesarias:
pip install -r requirements.txt

### Descarga los datos desde la página de FBref e inclúyelos en el directorio data/.

### Abre el notebook principal en el directorio notebooks/:
jupyter notebook notebooks/modelo_prediccion.ipynb

### Ejecuta las celdas para:
#### - Realizar el preprocesamiento de los datos.
#### - Entrenar el modelo de regresión logística.
#### - Evaluar el rendimiento del modelo con los conjuntos de datos de prueba.

##### Los resultados del modelo de regresión logística incluyen una matriz de confusión y un informe de clasificación.
###### Estos resultados se encuentran en el directorio results/.

##### Las contribuciones son bienvenidas. Si deseas mejorar el modelo o agregar nuevas características,
##### puedes crear un pull request o abrir un issue en el repositorio.

### Licencia:
##### Este proyecto está licenciado bajo la MIT License.