"""Exportación de figuras calculadas en las notebooks. No carga ni dibuja imágenes previas.

El código de cada gráfico permanece junto a su tabla en la notebook.
Esta función solo guarda la figura, su tabla y su descripción, y la muestra.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def guardar_y_mostrar(fig, datos, carpeta, nombre, descripcion, notebook, seccion):
    """Guarda PNG + SVG + CSV desde objetos de esta ejecución y devuelve una fila de índice."""
    if not isinstance(datos, pd.DataFrame) or datos.empty:
        raise ValueError(f'{nombre}: se necesita una tabla no vacía con los datos del gráfico.')
    if len(fig.axes) != 1:
        raise ValueError(f'{nombre}: utilizar un único gráfico por figura.')
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(carpeta / f'{nombre}.png', dpi=170, bbox_inches='tight')
    fig.savefig(carpeta / f'{nombre}.svg', bbox_inches='tight')
    datos.to_csv(carpeta / f'{nombre}_datos.csv', index=False)
    (carpeta / f'{nombre}.txt').write_text(descripcion + '\n', encoding='utf-8')
    plt.show()
    plt.close(fig)
    print(descripcion)
    print(f'Datos de esta figura: {nombre}_datos.csv')
    return {'notebook': notebook, 'seccion': seccion, 'figura': f'{nombre}.png',
            'vector': f'{nombre}.svg', 'datos': f'{nombre}_datos.csv',
            'filas': len(datos), 'descripcion': descripcion}


def guardar_indice(registros, carpeta):
    """Índice de figuras realmente generadas por la notebook en esta ejecución."""
    tabla = pd.DataFrame(registros)
    if tabla.empty or tabla.figura.duplicated().any():
        raise ValueError('El índice debe contener figuras generadas y nombres únicos.')
    carpeta = Path(carpeta)
    tabla.to_csv(carpeta / 'indice_figuras.csv', index=False)
    return tabla[['notebook', 'seccion', 'figura', 'datos', 'filas']]
