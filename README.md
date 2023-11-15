# Algoritmo de Imputación de Datos Sistemas SCADA:

<figure  align="center">
<img src='/assets/chec.png' width="800"> 
<figcaption>Jaivaná</figcaption>
</figure>



# Prerrequisitos para la Instalación:

Para instalar y ejecutar la solución, se deben cumplir los siguientes prerrequisitos:

- **Python:** Es necesario tener instalado Python versión 3.8.10 o superior.
- **git:** Es necesario tener instalado git o a la hora de clonar el repositorio, descargarlo en un archivo zip.

# Instrucciones de Instalación:

Para instalar el software, abra su interfaz de línea de comandos preferida y siga los pasos detallados a continuación:

Clone el repositorio en la máquina en la cual vaya a hacer el despliegue

```
$ git clone https://github.com/deathperminut/Chec_Data_project.git
```

Acceda al directorio del proyecto con el siguiente comando:

```
$ cd Chec_Data_project
```

Cree un entorno virtual (En caso de que la máquina en la que se ejecute la solución, también se ejecuten otros scripts de python):

```
$ python3 -m venv venv
```

Active el entorno virtual:

```
$ source venv/bin/activate
```

Instale los requerimientos:

```
$ pip install -r Project/requirements.txt
```

Descomprima cada uno de los archivos zip en la carpeta Project/Model_vae/Pesos

Lance el servicio:

```
$ python3 Project/main.py
```

