# Taller 4

Universidad El Bosque
 Mariana Lizarazo.

## Estructura del repositorio

```
lista_4/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/                  # CSV de los experimentos y de clientes
├── src/lista4/
│   ├── data.py            # carga de datos (load_csv)
│   ├── models.py          # regresión lineal y logística, resúmenes por modelo
│   └── metrics.py         # riesgos empíricos, errores de clasificación, correlación
└── notebooks/
    ├── ejercicio_1.ipynb  # regresión
    ├── ejercicio_2.ipynb  # colinealidad en tres archivos
    ├── ejercicio_3.ipynb  # cambio de escala de x2
    └── ejercicio_4.ipynb  # abandono de clientes y fuga de información
```

El código reutilizable está en `src/lista4/`; los cuadernos solo llaman a esas funciones.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          
pip install -r requirements.txt
pip install -e .                
```

## Cómo ejecutar los cuadernos

1. Abrir la carpeta del proyecto en VS Code (o ejecutar `jupyter lab`).
2. Abrir un cuaderno de `notebooks/` y seleccionar el kernel de Python donde se instalaron las dependencias.
3. Ejecutar todas las celdas (*Run All*).

Los cuadernos deben abrirse desde la carpeta `notebooks/`, porque cargan los datos y el paquete con rutas relativas a ella.

## Notas

- Pérdida cuadrática: R(h) = (1/n) Σ (yᵢ − h(xᵢ))².
- Regresión logística: `LogisticRegression(solver="lbfgs", C=1e6, max_iter=20000)` y umbral 0.5.
- Ejercicio 4: `ajuste_comercial_30d` usa datos posteriores a t0, por lo que se excluye de `variables_disponibles`.