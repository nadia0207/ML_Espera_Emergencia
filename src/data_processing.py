import numpy as np
import pandas as pd

def card_tipo(df,umbral_categoria = 10, umbral_continua = 30):
    """
    Analiza la cardinalidad de cada columna del DataFrame y sugiere el tipo de variable.

    Parameters
    ----------
    df : pd.DataFrame - DataFrame a analizar.
    umbral_categoria : int, optional (Por defecto 10)
    umbral_continua  : float, optional (Por defecto 30)

    Returns
    -------
    pd.DataFrame con las columnas: Card, %_Card, Tipo, tipo_sugerido.
        - Card: número de valores únicos
        - %_Card: porcentaje de valores únicos sobre el total
        - Tipo: tipo de dato original
        - tipo_sugerido: sugerencia de tipo (Binaria, Categorica, Numerica discreta, Numerica continua)
    """

    # Primera parte: Preparo el dataset con cardinalidades, % variación cardinalidad, y tipos
    df_temp = pd.DataFrame([df.nunique(), df.nunique()/len(df) * 100, df.dtypes]) # Cardinaliad y porcentaje de variación de cardinalidad
    df_temp = df_temp.T # Como nos da los valores de las columnas en columnas, y quiero que estas sean filas, la traspongo
    df_temp = df_temp.rename(columns = {0: "Card", 1: "%_Card", 2: "Tipo"}) # Cambio el nombre de la transposición anterior para que tengan más sentido, y uso asignación en vez de inplace = True (esto es arbitrario para el tamaño de este dataset)

    # Corrección para cuando solo tengo un valor
    df_temp.loc[df_temp.Card == 1, "%_Card"] = 0.00

    # Creo la columna de sugerenica de tipo de variable, empiezo considerando todas categóricas pero podría haber empezado por cualquiera, siempre que adapte los filtros siguientes de forma correspondiente
    df_temp["tipo_sugerido"] = "Categorica"
    df_temp.loc[df_temp["Card"] == 2, "tipo_sugerido"] = "Binaria"
    df_temp.loc[df_temp["Card"] >= umbral_categoria, "tipo_sugerido"] = "Numerica discreta"
    df_temp.loc[df_temp["%_Card"] >= umbral_continua, "tipo_sugerido"] = "Numerica continua"
    # Ojo los filtros aplicados cumplen con el enunciado pero no siguen su orden y planteamiento

    return df_temp

# ========================================================================================

def elimina_columna(df, *columnas):
    """
    Elimina una o varias columnas de un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame - DataFrame del que se eliminarán las columnas.
    *columnas : str - Nombres de las columnas a eliminar. Se pueden pasar múltiples columnas.

    Returns
    -------
    pd.DataFrame - DataFrame sin las columnas especificadas.

    Example
    -------
    >>> elimina_columna(df, 'Visit ID', 'Patient ID', 'Hospital ID')
    """   

    return df.drop(columns = list(columnas)) 

#=========================================================================================
#======== Categorizar tiempo de espera hasta médico =======
def categoriza_espera(df):
    """
    Crea una variable categórica ordinal a partir del tiempo de espera hasta ser atendido por un médico.

    Los tramos definidos son:
        1 → Corta    : 0  a 15 minutos
        2 → Media    : 15 a 45 minutos
        3 → Larga    : 45 a 90 minutos
        4 → Muy larga: más de 90 minutos

    Parameters
    ----------
    df : pd.DataFrame - DataFrame que debe contener la columna 'Time to Medical Professional (min)'.

    Returns
    -------
    pd.DataFrame - DataFrame con la nueva columna 'categoria_espera' de tipo entero (1-4).

    Example
    -------
    >>> categoriza_espera(df)
    """

    bins = [0, 15, 45, 90, float('inf')] #(90 - hasta infinito)
    labels = [1, 2, 3, 4]
    df = df.copy()
    df['categoria_espera'] = pd.cut(
        df['Time to Medical Professional (min)'],
        bins=bins,
        labels=labels
    ).astype(int)
    return df





