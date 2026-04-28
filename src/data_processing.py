
def elimina_columnas_id(df):
    columnas = ['Visit ID','Patient ID','Hospital ID']
    df_final = df.drop(columns = columnas)
    return df_final
    
def elimina_Total_Wait_Time(df):
    df_final = df.drop(columns = 'Total Wait Time (min)')
    return df_final

def elimina_Visit_Date(df):
    df_final = df.drop(columns = 'Visit Date')   
    return df_final

