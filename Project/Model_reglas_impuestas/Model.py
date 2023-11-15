#Packages:
import pyodbc
import numpy as np
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
import json
import datetime
#from Model_vae import model_vae
#import Model_vae.model_vae as vae
from Model_vae.model_vae import VaeModel

###FUNCTIONS:
def cumple_condicion(cadena):
    """ 
    DESCRIPTION: Function to filter the circuits comply with the nomenclature
    --------------------------------------------------------------------------
    --------------------------------------------------------------------------
    PAREMETERS:
    - cadena (str): string with the first five characters of the circuit code
    --------------------------------------------------------------------------
    --------------------------------------------------------------------------
    RETURN:
    Return a boolean variable indicating if the code comply with the nomenclature or  no.
    
    
    """
    if len(cadena) < 5:
        return False
    
    if cadena[0].isdigit() and cadena[1].isalpha():
        return True
    
    if cadena[:3].isalpha() and cadena[3:5].isdigit():
        return True
    
    return False

#FUNCTIONS:
def convert_variables_to_float(Dataframe):
    """ 
    DESCRIPTION: Function to become the interest variables in float type.
    -----------------------------------------------------------------------
    -----------------------------------------------------------------------
    PARAMETERS:
    - Dataframe (pd.DataFrame): Dataframe with the variables to transform.
    -----------------------------------------------------------------------
    -----------------------------------------------------------------------
    RETURN:
    - Dataframe (pd.DataFrame): The same dataframe with the transformed variables.
    
    
    """
    variables_to_convert=["VA","VB","VC","IA","IB","IC","P","Q"]
    Dataframe[variables_to_convert]=Dataframe[variables_to_convert].astype(float)
    return Dataframe

def getUmbrals(Dataframe):
    """ 
    DESCRIPTION: Function to get the comparation umbral.
    -------------------------------------------------
    -------------------------------------------------
    PARAMETERS:
    - Dataframe (pd.DataFrame): Dataframe with the interest variables.
    -------------------------------------------------
    -------------------------------------------------
    RETURN:
    None
    
    
    """
    umbral[0] = None
    # Interest variables:
    variables_interes = ['VA', 'VB', 'VC', 'IA', 'IB', 'IC', 'P','Q']
    # Filter the rows in the which the interest variables are not Null.
    filtered_df = Dataframe.dropna(subset=variables_interes, thresh=len(variables_interes))
    if filtered_df.shape[0] != 0:
        filtered_df['APARENTE_ORIGINAL'] = ((filtered_df['P'].astype(float)**2)+(filtered_df['Q'].astype(float)**2))**(1/2)
        P=Dataframe['VA'].astype(float) * Dataframe['IA'].astype(float) * 0.9 + Dataframe['VB'].astype(float) * Dataframe['IB'].astype(float) * 0.9 + Dataframe['VC'].astype(float) * Dataframe['IC'].astype(float) * 0.9
        Q=Dataframe['VA'].astype(float) * Dataframe['IA'].astype(float) * 0.435 + Dataframe['VB'].astype(float) * Dataframe['IB'].astype(float) * 0.435 + Dataframe['VC'].astype(float) * Dataframe['IC'].astype(float) * 0.435
        filtered_df['APARENTE_CALCULADA'] = ((P**2)+(Q**2))**(1/2)
        filtered_df['ERROR_POTENCIAS'] = (filtered_df['APARENTE_CALCULADA']-filtered_df['APARENTE_ORIGINAL']).abs() / (filtered_df['APARENTE_ORIGINAL']).abs() 
        data = float(filtered_df['ERROR_POTENCIAS'].median())
        if (np.isnan(data) or data == np.inf):
            umbral[0] = None
        else:
            umbral[0] = data



def initialCounts(ListBarras):
    """ 
    DESCRIPTION: Function to start the global variables of the algorithm.
    ---------------------------------------------------------------------
    ---------------------------------------------------------------------
    PARAMETERS:
    - ListaBarras (List): List with all the barras and red circuits will pass for the algorithm.
    ---------------------------------------------------------------------
    ---------------------------------------------------------------------
    RETURN:
    None
    
    """
    global barra;
    global variable;
    global porcent_error;
    global Dataframe_Signos;
    global umbral;
    
    porcent_error=0.15
    # Definir las columnas y los índices
    columnas = [
        'VA_NULOS_INICIALES', 'VA_NULOS_FINALES','VB_NULOS_INICIALES','VB_NULOS_FINALES','VC_NULOS_INICIALES','VC_NULOS_FINALES','IA_NULOS_INICIALES', 
        'IA_NULOS_FINALES','IB_NULOS_INICIALES', 'IB_NULOS_FINALES','IC_NULOS_INICIALES', 'IC_NULOS_FINALES' ,'P_NULOS_INICIALES', 'P_NULOS_FINALES',
        'Q_NULOS_INICIALES', 'Q_NULOS_FINALES','IA_NEGATIVO_CORREGIDO','IB_NEGATIVO_CORREGIDO','IC_NEGATIVO_CORREGIDO','VA_NEGATIVO_CORREGIDO','VB_NEGATIVO_CORREGIDO',
        'VC_NEGATIVO_CORREGIDO','CORRECCION_ESCALA_VA','CORRECCION_ESCALA_VB','CORRECCION_ESCALA_VC','CORRECCION_ESCALA_P','CORRECCION_ESCALA_Q',"VA_IMPUTADO_MEDIANA","VA_IMPUTADO_NIVEL_TENSION",
        "VB_IMPUTADO_MEDIANA","VB_IMPUTADO_NIVEL_TENSION","VC_IMPUTADO_MEDIANA","VC_IMPUTADO_NIVEL_TENSION",
        "P_CAMBIADO_CERO","Q_CAMBIADO_CERO","P_CAMBIADO_POR_CALCULADO","Q_CAMBIADO_POR_CALCULADO","CABEZERAS_CON_CIRCUITO_SIN_REGISTRO",
        "IA_IMPUTADO_POR_ANALISIS_NODAL","IB_IMPUTADO_POR_ANALISIS_NODAL","IC_IMPUTADO_POR_ANALISIS_NODAL","CABEZERAS_CON_CIRCUITO_SIN_REGISTRO_O_SIGNOS_DE_POTENCIAS_INCORRECTOS","NO_ES_POSIBLE_IDENTIFICAR_FLUJO_POR_POTENCIAS","V_CAMBIADO_CERO_POR_POTENCIAS_CERO","I_CAMBIADO_CERO_POR_POTENCIAS_CERO"]
    # Crear el DataFrame con valores iniciales en cero
    data = [[0 for _ in columnas] for _ in ListBarras]
    umbral = [None]
    variable=[None]
    barra=[None]
    Dataframe_Signos=[None]

def convertToDate(data):
    """ 
    DESCRIPTION: Function to become the date column to datetime type.
    ------------------------------------------------------------------
    ------------------------------------------------------------------
    PARAMETERS:
    - data (pd.DataFrame): Dataframe with the column to transform.
    ------------------------------------------------------------------
    ------------------------------------------------------------------
    RETURN:
    - data (pd.DataFrame): The samedataframe with the transformed column.
    
    """
    data['TIEMPO_AJUSTADO'] = pd.to_datetime(data['TIEMPO_AJUSTADO'])
    return data

def getUniqueTime(data):
    """ 
    DESCRIPTION: Function to get the unique dates of the dataframe.
    ---------------------------------------------------------------
    ---------------------------------------------------------------
    PARAMETERS:
    - data (pd.DataFrame): Dataframe with the column TIEMPO_AJUSTADO
    ---------------------------------------------------------------
    ---------------------------------------------------------------
    RETURN:
    unique_dates (pd.DataFrame): Unique Dates.
    """
    return data['TIEMPO_AJUSTADO'].unique()


# Función para convertir valores negativos en positivos
def convertir_positivo(valor):
    """ 
    DESCRIPTION: Function to become the negative interest variables to positive variables.
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
    PARAMETERS:
    - valor (float): Value of the interest variable.
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
    RETURN:
    - valor (float): Positive value of the interest variable.
    """
    if valor is None:
        return valor
    elif valor < 0:
        return abs(valor)
    else:
        return valor
# Función que rescala los valores que no estan en una escala adecuada
def convertir_escala(valor):
    """ 
    DESCRIPTION: Function to correct the scale of the interest variables.
    ---------------------------------------------------------------------
    ---------------------------------------------------------------------
    PARAMETERS:
    - valor (float): Value of the possible variable to correct.
    ---------------------------------------------------------------------
    ---------------------------------------------------------------------
    RETURN:
    - valor (float): Rescale variable of the interest variable.
    """
    if valor is None:
        return valor
    elif variable[0] == 'VA' and valor > 1000:
        return valor / 1000

    elif variable[0] == 'VB' and valor > 1000:
        return valor / 1000

    elif variable[0] == 'VC' and valor > 1000:
        return valor / 1000

    elif variable[0] == 'P' and valor > 100:
        return valor / 1000

    elif variable[0] == 'Q' and valor > 100:
        return valor / 1000
        
    else:
        return valor





def Absolute(Datafilter):
    """ 
    DESCRIPTION: Function to become the interest variables to positive.
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): The same dataframe with the interest variables positive.
    """
    variable[0] = 'IA'
    Datafilter['IA'] = Datafilter['IA'].map(convertir_positivo);
    variable[0] = 'IB'
    Datafilter['IB'] = Datafilter['IB'].map(convertir_positivo);
    variable[0] = 'IC'
    Datafilter['IC'] = Datafilter['IC'].map(convertir_positivo);
    variable[0] = 'VA'
    Datafilter['VA'] = Datafilter['VA'].map(convertir_positivo);
    variable[0] = 'VB'
    Datafilter['VB'] = Datafilter['VB'].map(convertir_positivo);
    variable[0] = 'VC'
    Datafilter['VC'] = Datafilter['VC'].map(convertir_positivo);
    return Datafilter

def ReScale(Datafilter):
    """ 
    DESCRIPTION: Function to reescale the interest variables.
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): The same dataframe with the rescale interest variables.
    """

    variable[0] = 'VA'
    Datafilter['VA'] = Datafilter['VA'].map(convertir_escala);
    variable[0] = 'VB'
    Datafilter['VB'] = Datafilter['VB'].map(convertir_escala);
    variable[0] = 'VC'
    Datafilter['VC'] = Datafilter['VC'].map(convertir_escala);
    variable[0] = 'P'
    Datafilter['P'] = Datafilter['P'].map(convertir_escala);
    variable[0] = 'Q'
    Datafilter['Q'] = Datafilter['Q'].map(convertir_escala);
    return Datafilter

def CheckVoltage(Datafilter):
    """ 
    DESCRIPTION: Function to impute voltage per imposed rulers.
    ------------------------------------------------------------
    ------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the variables to impute.
    ------------------------------------------------------------
    ------------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): The same dataframe with the imputed variables.
    """
    if Datafilter['VA'].count() == 0:
        pass

    else:

        mediana=Datafilter["VA"].median()
        max=Datafilter["VA"].max()
        min=Datafilter["VA"].min()
        dif=abs(max-min)
        if max==0:
                Datafilter["VA"].fillna(mediana,inplace=True)
        else:
                per_er=abs(dif/max)
                if per_er>0.30:
                        Datafilter["VA"].fillna(max,inplace=True)
                else:
                        Datafilter["VA"].fillna(mediana,inplace=True)
    
    if Datafilter['VB'].count() == 0:
        pass

    else:
        mediana=Datafilter["VB"].median()
        max=Datafilter["VB"].max()
        min=Datafilter["VB"].min()
        dif=abs(max-min)
        if max==0:
                Datafilter["VB"].fillna(mediana,inplace=True)
        else:
                per_er=abs(dif/max)
                if per_er>0.30:
                        Datafilter["VB"].fillna(max,inplace=True)
                else:
                        Datafilter["VB"].fillna(mediana,inplace=True)
    
    if Datafilter['VC'].count() == 0:
        pass

    else:
        mediana=Datafilter["VC"].median()
        max=Datafilter["VC"].max()
        min=Datafilter["VC"].min()
        dif=abs(max-min)
        if max==0:
                Datafilter["VC"].fillna(mediana,inplace=True)
        else:
                per_er=abs(dif/max)
                if per_er>0.30:
                        Datafilter["VC"].fillna(max,inplace=True)
                else:
                        Datafilter["VC"].fillna(mediana,inplace=True)
    
    return Datafilter
    

def CurrentVoltageCero(Datafilter):
    """ 
    DESCRIPTION: Function to replace the powers (P and Q) to zero, in the cases
    in the which all the currents are zero or/and all the voltages are zero.
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): DataFrame with the interest variables.
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    RETURN:
    Datafilter (pd.DataFrame): The same dataframe with the changes done.
    """
    indices_cero = list(Datafilter.loc[((Datafilter['IA'] == 0) & (Datafilter['IB'] == 0) & (Datafilter['IC'] == 0)) | 
                ((Datafilter['VA'] == 0) & (Datafilter['VB'] == 0) & (Datafilter['VC'] == 0))].index)
    Datafilter['P'].loc[indices_cero] = 0 ## Replace the powers by zero
    Datafilter['Q'].loc[indices_cero] = 0 ## Replace the powers by zero
    return Datafilter

def getSign(Number):
    """ 
    DESCRIPTION: Function to get the sign of a number.
    ------------------------------------------------------------
    ------------------------------------------------------------
    PARAMETERS:
    - Number (float): Number to get the sign.
    ------------------------------------------------------------
    ------------------------------------------------------------
    RETURN:
    - (int): 1 if is a positive number and -1 if is a negative number.
    """
    if ( float(Number) < 0 ):
       return -1
    else:
       return 1

def findLastRegister(df):
    """ 
    Description: Function to get the last good signs of the powers (P and Q).
    -------------------------------------------------------------------------
    -------------------------------------------------------------------------
    PARAMETERS:
    - df (pd.DataFrame): Dataframe with the interest variables.
    -------------------------------------------------------------------------
    -------------------------------------------------------------------------
    RETURN:
    None
    """
    CIRCUITOS_UNICOS = df['CIRCUITO'].unique()
    lista=[]
    for CIRCUITO in CIRCUITOS_UNICOS:
        registros_asociados = df[df['CIRCUITO'] == CIRCUITO]
        registros_filtrados = registros_asociados[(registros_asociados['P'].notna()) & (registros_asociados['P'] != 0) & (registros_asociados['Q'].notna()) & (registros_asociados['Q'] != 0)]
        registros_ordenados = registros_filtrados.sort_values(by=['TIEMPO_AJUSTADO'], ascending=False)
        if (registros_ordenados.shape[0]!=0):
            lista.append({'CIRCUITO':CIRCUITO,'P':getSign(registros_ordenados.iloc[0].loc['P']),'Q':getSign(registros_ordenados.iloc[0].loc['Q'])})
        else:
            lista.append({'CIRCUITO':CIRCUITO,'P':0,'Q':0}) 

    Dataframe_Signos[0] = pd.DataFrame(lista)

def correctSigno(Datafilter):
    """ 
    DESCRIPTION: Function to correct the signs of the powers.
    ---------------------------------------------------------
    ---------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): DataFrame with the interest variables.
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): The same dataframe with the sign of the powers corrected.
    
    """
    for index in Datafilter.index:
        CIRCUITO = str(Datafilter['CIRCUITO'].loc[index])
        if (pd.isna(Datafilter["P"].loc[index]) == False  and pd.isna(Datafilter["Q"].loc[index]) == False):
            if (Dataframe_Signos[0][Dataframe_Signos[0]['CIRCUITO'] == CIRCUITO]['P'].iloc[0] != 0):
                Datafilter['P'].loc[index] = Datafilter['P'].loc[index] * Dataframe_Signos[0][Dataframe_Signos[0]['CIRCUITO'] == CIRCUITO]['P'].iloc[0]
                Datafilter['Q'].loc[index] = Datafilter['Q'].loc[index] * Dataframe_Signos[0][Dataframe_Signos[0]['CIRCUITO'] == CIRCUITO]['Q'].iloc[0]
    return Datafilter


def Verificar_columnas_para_imputar(Datafilter,Columnas_Nulos):
    """ 
    DESCRIPTION: Function to get the columns only have one lost data.
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    - Columnas_Nulos (List): List with the columns that have NULL data.
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    RETURN:
    - columnas_perdido (List): List with the names of the columns that have only one lost data.
    """
    df = Datafilter.copy()
    columnas_perdido = []

    for columna in Columnas_Nulos:
        if df[columna].isnull().sum() == 1:
            columnas_perdido.append(columna)

    return columnas_perdido


def replaceCurrentsGeneral(df, currents_columns, target_row):
    """ 
    DESCRIPTION: Function to impute currents.
    ------------------------------------------
    ------------------------------------------
    PARAMETERS:
    - df (pd.DataFrame): Dataframe with the interest variables.
    - current_columns (List): List with the names of the columns of the currents.
    - target_row: Index of the circuit with the major value.
    -----------------------------------------------------------------------------
    -----------------------------------------------------------------------------
    RETURN: 
    - df (pd.DataFrame): Dataframe with the interest variables with some imputations.
    """
    # Sum all rows except the target row to find the missing value for each current column
    for col in currents_columns:
        null_positions = df[col].isna()
        first_null_position = null_positions.idxmax()
    if (first_null_position == target_row):
        df.at[target_row,col] = df[col].sum()
    else:
        df.at[first_null_position,col] =abs(df[col].loc[target_row] - df.loc[df.index != target_row, col].sum())
    return df

def getTargetRow(Datafilter,columnas):
    """ 
    DESCRIPTION: Function to get the index of the circuit with the majour current.
    ----------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    - columnas (List): List with the columns of the variables of currents.
    """
    df = Datafilter.copy()
    idx_mayores = df[columnas].idxmax()
    return idx_mayores[columnas[0]]



def analizar_fases_completas(Datafilter):
    """ 
    DESCRIPTION: Function to find the full columns, it mean, the columns without Null Values.
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest columns.
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    RETURN:
    columnas_completas (List): List with the full columns.
    """
    # Interest columns:
    columns_of_interest = ['IA', 'IB', 'IC']

    # EFind columns without Null values:
    columns_without_null = Datafilter[columns_of_interest].columns[Datafilter[columns_of_interest].notnull().all()]

    # Become in a list
    columns_list = columns_without_null.tolist()

    columnas_completas = []
    ## Verify no all the values of the full columns are zero.
    for fase in columns_list:
        if (Datafilter[fase].sum() != 0):
            columnas_completas.append(fase)
    return columnas_completas

def obtener_mayor(valor1, valor2):
    """ 
    DESCRIPTION: Function to get the majour between two numbers.
    ------------------------------------------------------------
    ------------------------------------------------------------
    PARAMETERS:
    - valor1 (float): Number 1.
    - valor2 (float): Number 2.
    ------------------------------------------------------------
    ------------------------------------------------------------
    RETURN:
    - (float): Majour number.
    """
    if valor1 > valor2:
        return valor1
    else:
        return valor2

def contar_mayores(arr):
    """ 
    DESCRIPTION: Function to get a flag that indicate if a element of a array is majour that a porcent_error or no.
    ---------------------------------------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------
    PARAMETERS:
    - arr (list): List with the elements to verify.
    ---------------------------------------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------
    RETURN:
    - contador (Boolean): Indicator flag.
    """
    contador = False
    for num in arr:
        if(np.isnan(num) == False):
            if num > porcent_error:
                return True
    return contador

def calculate_error(Datafilter,fases_completas):
    """ 
    DESCRIPTION: Function to verify if the error is majour to the umbral in some variable.
    ----------------------------------------------------
    ----------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    - fases_completas (List): List with the name of the full columns.
    ----------------------------------------------------
    ----------------------------------------------------
    RETURN:
    - Flag (Boolean): Indicator Flag.
    """
    ##General Method:
    errores=[]
    Datafilter=convert_variables_to_float(Datafilter)
    for fase in fases_completas:
        errores.append(abs(Datafilter[fase].loc[Datafilter.index[0]] - Datafilter[fase].loc[Datafilter.index[1]]) / obtener_mayor(Datafilter[fase].loc[Datafilter.index[0]],Datafilter[fase].loc[Datafilter.index[1]]))
    ##There is some error majour to the umbral?
    flag = contar_mayores(errores)
    return flag

def calcular_error_general(dataframe,columnas):
    """ 
    DESCRIPTION: Function to calculate the general error.
    -------------------------------------------------
    -------------------------------------------------
    PARAMETERS:
    - dataframe (pd.DataFrame): Dataframe with the interest variables.
    - columnas (List): Interest columns.
    -------------------------------------------------
    -------------------------------------------------
    RETURN:
    - flag (Boolean): Indicator Flag.
    """
    dataframe[["IA","IB","IC","VA","VB","VC","P","Q"]]=dataframe[["IA","IB","IC","VA","VB","VC","P","Q"]].astype(float)
    df = dataframe.copy()
    mayores = df[columnas].max(axis=0)
    idx_mayores = df[columnas].idxmax()
    misma_fila = set(list(idx_mayores))
    if len(misma_fila) == 1:  
        suma_restantes = df[columnas].sum(axis=0) - mayores
        errores = ((mayores - suma_restantes).abs() /  suma_restantes.abs())
        flag = contar_mayores(errores)
        return flag
    else:
        return True

def replaceCurrentsCase2(Datafilter):
    """ 
    DESCRIPTION: Function to replace currents in the case of two circuits in a cabecera.
    ---------------------------------------
    ---------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    --------------------------------------------------
    --------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables and the current replaced.
    
    
    
    """

    etiquetas=['IA','IB','IC']
    for etiqueta in etiquetas:
        mediana=Datafilter[etiqueta].median()
        if Datafilter[etiqueta].isnull().sum() == 1:
            Datafilter[etiqueta].fillna(mediana,inplace=True) 
    return Datafilter


def columnas_con_nulos(lista_sin_nulos):
    """ 
    DESCRIPTION: Function to identify columns without Null values.
    ---------------------------------------------------------------
    ---------------------------------------------------------------
    PARAMETERS:
    - lista_sin_nulos (List): List with the name of the columns that do not have Null values.
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    RETURN:
    - (List): List with the name of the columns that have Null values.
    """
    lista_completa = ['IA','IB','IC']
    return [elemento for elemento in lista_completa if elemento not in lista_sin_nulos]


def CheckPower(Datafilter):
    """ 
    DESCRIPTION: Function to calculate powers with your sign if is necesary.
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables and the calculated powers.
    """

    ## Check if are all the current and volage values
    # Interest columnns:
    columnas_verificar = ['IA', 'IB', 'IC', 'VA', 'VB' , 'VC']
    # Verify if the rows don´t have Null values in the specified columns:
    filas_sin_nulos = Datafilter[columnas_verificar].notna().all(axis=1)
    # Add a columns to the dataframe to indicate if the row has all the values no Null.
    Datafilter_copy=Datafilter.copy()
    Datafilter_copy['sin_nulos'] = filas_sin_nulos
    for index in Datafilter.index:
        ##Verify if is the sign of the circuit
        CIRCUITO = Datafilter['CIRCUITO'].loc[index]

        if Datafilter_copy['sin_nulos'].loc[index] and Dataframe_Signos[0][Dataframe_Signos[0]['CIRCUITO'] == CIRCUITO]['P'].iloc[0] != 0:
            variables_a_verificar=['P','Q']
            hay_valor_nulo = Datafilter[variables_a_verificar].loc[index].isnull().any().any() ## VERIFICAMOS SI P O Q ESTAN PERDIDAS
            ##Calculate the aparent power
            Datafilter=convert_variables_to_float(Datafilter)
            P=( Datafilter['VA'].loc[index] * Datafilter['IA'].loc[index] * 0.9 + Datafilter['VB'].loc[index] * Datafilter['IB'].loc[index] * 0.9 + Datafilter['VC'].loc[index] * Datafilter['IC'].loc[index] * 0.9 ) / 1000
            Q=( Datafilter['VA'].loc[index] * Datafilter['IA'].loc[index] * 0.435 + Datafilter['VB'].loc[index] * Datafilter['IB'].loc[index] * 0.435 + Datafilter['VC'].loc[index] * Datafilter['IC'].loc[index] * 0.435 ) /1000
            if hay_valor_nulo == True:
                ##Replace if there is some Null value
                Datafilter['P'].loc[index]=P
                Datafilter['Q'].loc[index]=Q
            else:
                Potencia_Aparente_calculada=(P**2 + Q**2)**(1/2)
                Potencia_Aparente_Registrada=(Datafilter['P'].loc[index]**2 + Datafilter['Q'].loc[index]**2)**(1/2)
                ##Calculate the error percent.
                Porcentaje_error= abs(Potencia_Aparente_calculada - Potencia_Aparente_Registrada) / Potencia_Aparente_Registrada
                if (np.isnan(Porcentaje_error) == False and umbral[0] != None ):
                    if (Porcentaje_error > umbral[0]):
                        Datafilter['P'].loc[index]=P
                        Datafilter['Q'].loc[index]=Q
    
    Datafilter = correctSigno(Datafilter)
    return Datafilter

def partition_dataframe_column(df, column_name):
    """ 
    DESCRIPTION: Function to find the possibles groups of a combination in a nodal analysis.
    ----------------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------------
    PARAMETERS:
    - df (pd.DataFrame): Dataframe with the interest variables.
    - column_name (str): String with the name of the column.
    ----------------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------------
    RETURN:
    - group1_rows (List): List with the indices of the rows of the group 1.
    - group2_rows (List): List with the indices of the rows of the group 2.
    """
    arr = df[column_name].values
    total_sum = sum(arr)
    n = len(arr)

    dp = np.zeros((n + 1, int(total_sum * 100) + 1), dtype=bool)
    dp[0][0] = True

    for i in range(1, n + 1):
        for j in range(int(arr[i - 1] * 100), int(total_sum * 100) + 1):
            dp[i][j] = dp[i - 1][j]
            if arr[i - 1] <= j / 100:
                dp[i][j] = dp[i][j] or dp[i - 1][int((j - arr[i - 1] * 100))]

    target_sum = total_sum / 2
    while not dp[n][int(target_sum * 100)]:
        target_sum -= 0.01

    group1_indices = []
    group2_indices = []
    i = n
    j = int(target_sum * 100)
    while i > 0 and j > 0:
        if dp[i][j] and not dp[i - 1][j]:
            group1_indices.append(i - 1)
            j -= int(arr[i - 1] * 100)
        else:
            group2_indices.append(i - 1)
        i -= 1

    group1_indices.reverse()
    group2_indices.reverse()

    group1_rows = df.iloc[group1_indices]
    group2_rows = df.iloc[group2_indices]

    return group1_rows, group2_rows

def getSuperior(dato_1,dato_2):
    """ 
    DESCRIPTION: Function to get the majour between two numbers.
    ------------------------------------------------------------
    ------------------------------------------------------------
    PARAMETERS:
    - dato_1 (float): Number 1.
    - dato_2 (float): Number 2.
    ------------------------------------------------------------
    ------------------------------------------------------------
    RETURN:
    - (float): Majour number.
    """
    if dato_1>=dato_2:
        return dato_1
    else:
        return dato_2

def NodalAnalysisWithoutFullPhases(Datafilter,FasesImputar):
    """ 
    DESCRIPTION: Function to do the nodal analysis without full phases.
    ------------------------------------------------------------
    ------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    - FasesImputar (List): List with the names of the columns to impute.
    ------------------------------------------------------------
    ------------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables with some imputed variables.
    
    """
    ##Get the circuits
    Circuit_unique = list(Datafilter['CIRCUITO'].unique())
    ## Get the sign of the powers
    filtered_df_signos = Dataframe_Signos[0][Dataframe_Signos[0]['CIRCUITO'].isin(Circuit_unique)]
    # Condition to filter
    condicion_entrada = filtered_df_signos['P'] == 1
    condicion_salida = filtered_df_signos['P'] == -1
    # Get the values of 'columna_name' that comply with the condition
    valores_filtrados_entrada = filtered_df_signos.loc[condicion_entrada, 'CIRCUITO'].tolist()
    valores_filtrados_salida = filtered_df_signos.loc[condicion_salida, 'CIRCUITO'].tolist()
    if (len(valores_filtrados_entrada) == 0 or len(valores_filtrados_salida) == 0 ):
        
        return Datafilter
    else:
        Dataframe_filtrados_entrada = Datafilter[Datafilter['CIRCUITO'].isin(valores_filtrados_entrada)]
        Dataframe_filtrados_salida = Datafilter[Datafilter['CIRCUITO'].isin(valores_filtrados_salida)]
        Datafilter = nodalCurrents(Datafilter,Dataframe_filtrados_entrada,Dataframe_filtrados_salida,FasesImputar)
        return Datafilter
        


def checkNodalAnalysis(Datafilter,fase_completa):
    """ 
    DESCRIPTION:
    -------------------------------------------------
    -------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    - fase_completa (str): String with the name of the column of the full phase.
    -------------------------------------------------
    -------------------------------------------------
    RETURN:
    - [Datafilter, flag, Dataframe_filtrados_entrada, Dataframe_filtrados_salida] (List): Original Datafilter, indicator flag, Dataframe with the information os the input circuits,
    Dataframe with the information os the output circuits
    
    """
    ##Get the circuits
    Circuit_unique = list(Datafilter['CIRCUITO'].unique())
    ## Get the sign of the powers
    filtered_df_signos = Dataframe_Signos[0][Dataframe_Signos[0]['CIRCUITO'].isin(Circuit_unique)]
    # Condition to filter
    condicion_entrada = filtered_df_signos['P'] == 1
    condicion_salida = filtered_df_signos['P'] == -1
    valores_filtrados_entrada = filtered_df_signos.loc[condicion_entrada, 'CIRCUITO'].tolist()
    valores_filtrados_salida = filtered_df_signos.loc[condicion_salida, 'CIRCUITO'].tolist()
    if (len(valores_filtrados_entrada) == 0 or len(valores_filtrados_salida) == 0 ):
        return [Datafilter,True,[],[]]
    else:
        Dataframe_filtrados_entrada = Datafilter[Datafilter['CIRCUITO'].isin(valores_filtrados_entrada)]
        Dataframe_filtrados_salida = Datafilter[Datafilter['CIRCUITO'].isin(valores_filtrados_salida)]
        ##Sum by columns for every phase
        Suma_entradas = Dataframe_filtrados_entrada[fase_completa].sum()
        Suma_salidas = Dataframe_filtrados_salida[fase_completa].sum()
        try:
            error_percent = (Suma_entradas - Suma_salidas).abs() / Suma_salidas.abs()
            flag = any(valor > porcent_error for valor in list(error_percent))  ## TRUE if is majour to the umbral
            return [Datafilter, flag, Dataframe_filtrados_entrada, Dataframe_filtrados_salida] 
        except Exception as e:
            banderas = []
            for i in fase_completa:
                if (Suma_salidas[i] == 0 and Suma_entradas[i] == 0):
                    bandera = False
                elif Suma_salidas[i] == 0:
                    bandera=True
                else:
                    error=abs(Suma_entradas[i]-Suma_salidas[i]/abs(Suma_salidas))
                    
                    bandera = True
                banderas.append(bandera)
    
                flag = True if True in banderas else False
                return [Datafilter, flag, Dataframe_filtrados_entrada, Dataframe_filtrados_salida]
   

def nodalCurrents(Datafilter,group_1,group_2,fases_restantes):
    """ 
    DESCRIPTION: Function to impute data in the possible phases.
    ---------------------------------------------------------------
    ---------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    - group_1 (pd.DataFrame): DataFrame with the interest variables of the group 1.
    - group_2 (pd.DataFrame): DataFrame with the interest variables of the group 2.
    - fases_restantes (List): List with the names of the columns to impute data.
    ---------------------------------------------------------------
    ---------------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables and the possible imputed data.
    
    """ 
    ##Here the rest phases only have a Null data.
    for fase in fases_restantes:
        ##Get the index of the Null Value
        missing_indice = Datafilter.index[Datafilter[fase].isnull()].tolist()[0]
        ##Get the two dataframe of the combination
        df_group_1_sum =  Datafilter[fase].loc[group_1.index].sum() ## Get the elements of a group 1
        df_group_2_sum =  Datafilter[fase].loc[group_2.index].sum() ## Get the elements of a group 2
        ## Check in what group is the Null value:
        if missing_indice in group_1.index :
            ## If the index is in the group 1:
            Datafilter.at[missing_indice, fase] = abs(df_group_2_sum - df_group_1_sum)
        elif missing_indice in group_2.index:
            ## If the index is in the group 2:
            Datafilter.at[missing_indice, fase] = abs(df_group_1_sum - df_group_2_sum)
        else:
            
            if(df_group_1_sum > df_group_2_sum):
                Datafilter.at[missing_indice,fase] = abs(df_group_1_sum - df_group_2_sum)
                indice=list(Dataframe_Signos[0][Dataframe_Signos[0]['CIRCUITO'] == Datafilter['CIRCUITO'].loc[missing_indice]].index)[0]
                Dataframe_Signos[0]['P'].loc[indice] = -1
                Dataframe_Signos[0]['Q'].loc[indice] = -1
                

            elif (df_group_1_sum < df_group_2_sum):
                Datafilter.at[missing_indice,fase] = abs(df_group_1_sum - df_group_2_sum)
                indice=list(Dataframe_Signos[0][Dataframe_Signos[0]['CIRCUITO'] == Datafilter['CIRCUITO'].loc[missing_indice]].index)[0]
                Dataframe_Signos[0]['P'].loc[indice] = 1
                Dataframe_Signos[0]['Q'].loc[indice] = 1
                
        
    return Datafilter

def checkFases(group_1,group_2,Datafilter,fases_restantes):
    """ 
    DESCRIPTION: Function to check if the found combination makes sense.
    -------------------------------------------------------
    -------------------------------------------------------
    PARAMETERS:
    - group_1 (pd.DataFrame): DataFrame with the interest variables of the group 1.
    - group_2 (pd.DataFrame): DataFrame with the interest variables of the group 2.
    - Datafilter (pd.DataFrame): DataFrame with the interest variables.
    - fases_restantes (List): List with the names of the columns to impute data.
    -------------------------------------------------------
    -------------------------------------------------------
    RETURN:
    - (Boolean): Boolean variable indicating if the found combination makes sense or no.
    
    """
    

    for fase in fases_restantes:
        ##Dataframes of the combination
        df_group_1_sum =  Datafilter[fase].loc[group_1.index].sum() ## OBTENEMOS LOS ELEMENTOS DE UNA LADO SUMADOS SIN NULOS
        df_group_2_sum =  Datafilter[fase].loc[group_2.index].sum() ## OBTENEMOS LOS ELEMENTOS DE UNA LADO SUMADOS SIN NULOS
        ## MIRAMOS EN CUAL DE LOS 2 GRUPOS ESTA EL VALOR PERDIDO
        ##OBTENEMOS EL DATO MAYOR PARA CALCULAR EL % DE ERROR
        Mayor = getSuperior(df_group_1_sum,df_group_2_sum)
        Error = (df_group_1_sum - df_group_2_sum) / Mayor
        if porcent_error > Error:
            return True
        
        
    return False



def VerifyPowerWithoutRegister(Datafilter):
    """ 
    DESCRIPTION: Function to verify if could regist the sign of the powers for the circuits.
    ----------------------------------------------
    ----------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    ----------------------------------------------
    ----------------------------------------------
    RETURN:
    - (int): Amount of index that comply with the condition.
    
    """
    ##Get the circuits
    Circuit_unique = list(Datafilter['CIRCUITO'].unique())
    ## Get the sign of the powers
    filtered_df_signos = Dataframe_Signos[0][Dataframe_Signos[0]['CIRCUITO'].isin(Circuit_unique)]
    ##There is some circuit with a power without sign?
    Circuitos_sin_potencias = filtered_df_signos['P'] == 0
    # Condition
    condicion = lambda x: x == True

    # Index of the elements that comply with the condition:
    indices = [i for i, elemento in enumerate(Circuitos_sin_potencias) if condicion(elemento)]
    return len(indices)



def PowerZero(Datafilter):
    """ 
    DESCRIPTION: Function to replace some currents or/and some voltage per cero, by powers in zero.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables and some powers replaced by zero.
    """
    indices_cero = list(Datafilter.loc[((Datafilter['P'] == 0) & (Datafilter['Q'] == 0))].index)
    Datafilter_ceros = Datafilter.loc[indices_cero] ## Replace the powers by zero
    Indices_interes_caso_1 = list(Datafilter_ceros.loc[(
                            (Datafilter_ceros['VA'].notna()) & (Datafilter_ceros['VB'].notna()) & (Datafilter_ceros['VC'].notna()) &
                            (Datafilter_ceros['VA'] != 0) & (Datafilter_ceros['VB'] != 0) & (Datafilter_ceros['VC'] != 0)
                            & ((Datafilter_ceros['IA'].isna()) | (Datafilter_ceros['IA'] == 0))
                            & ((Datafilter_ceros['IB'].isna()) | (Datafilter_ceros['IB'] == 0))
                            & ((Datafilter_ceros['IC'].isna()) | (Datafilter_ceros['IC'] == 0))
                            )].index)
    Indices_interes_caso_2 = list(Datafilter_ceros.loc[(
                            (Datafilter_ceros['IA'].notna()) & (Datafilter_ceros['IB'].notna()) & (Datafilter_ceros['IC'].notna()) &
                            (Datafilter_ceros['IA'] != 0) & (Datafilter_ceros['IB'] != 0) & (Datafilter_ceros['IC'] != 0)
                            & ((Datafilter_ceros['VA'].isna()) | (Datafilter_ceros['VA'] == 0))
                            & ((Datafilter_ceros['VB'].isna()) | (Datafilter_ceros['VB'] == 0))
                            & ((Datafilter_ceros['VC'].isna()) | (Datafilter_ceros['VC'] == 0))
                            )].index)
    
    if(len(Indices_interes_caso_1) != 0):
        for index in Indices_interes_caso_1:
            Datafilter['IA'].loc[index] = 0
            Datafilter['IB'].loc[index] = 0
            Datafilter['IC'].loc[index] = 0
    if(len(Indices_interes_caso_2) != 0):
        for index in Indices_interes_caso_1:
            Datafilter['VA'].loc[index] = 0
            Datafilter['VB'].loc[index] = 0
            Datafilter['VC'].loc[index] = 0
    return Datafilter


def process_date(date,dataframe):
    """ 
    DEESCRIPTION: Function to impute currents, voltages or/and powers by zero.
    --------------------------------------------------------
    --------------------------------------------------------
    PARAMETERS:
    - dataframe (pd.DataFrame): Dataframe with the interest variables.
    - date: Date of interest.
    --------------------------------------------------------
    --------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables and some imputed variables.
    
    """
    ##Values of that date
    Datafilter = dataframe[dataframe['TIEMPO_AJUSTADO'] == date]

    # Filter the rows in the which the character number six of the circuit code is iqual to six:
    df_circuits_no_nodal = Datafilter[Datafilter['CIRCUITO'].str[5] == 'B']

    # Create a new  DataFrame 'df_no_b' with the rows that don´t comply with the condition
    Datafilter = Datafilter[Datafilter['CIRCUITO'].str[5] != 'B']
    

    try:
        ##Impute voltages of phase:
        Datafilter = CheckVoltage(Datafilter)
        ##Impute the powers in the which the voltage and the current are iqual to zero
        Datafilter = CurrentVoltageCero(Datafilter)
        ##Impute currents or voltajes in the case the powers are iqual to zero
        Datafilter = PowerZero(Datafilter)
        ##Query the amount of circuits
        cantidad_circuitos = Datafilter.shape[0]
        if cantidad_circuitos == 1:
            Datafilter = Case_1_circuit(Datafilter)
        elif cantidad_circuitos == 2:
            Datafilter = Case_2_circuit(Datafilter)
        elif cantidad_circuitos == 3:
            Datafilter = Case_3_circuit(Datafilter)
        elif cantidad_circuitos > 3:
            Datafilter = Case_n_circuit(Datafilter)
        Datafilter=pd.concat([Datafilter,df_circuits_no_nodal],axis=0)
        return Datafilter
        
    except Exception as e:
        Datafilter=pd.concat([Datafilter,df_circuits_no_nodal],axis=0)
        return Datafilter

    

def Case_1_circuit(Datafilter):
    """ 
    DEESCRIPTION: Function to do the nodal and power analysis in the case of one circuit.
    --------------------------------------------------------
    --------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    --------------------------------------------------------
    --------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables and some imputed variables.
    """
    
    BoolPower = VerifyPowerWithoutRegister(Datafilter)
    if (BoolPower == 1):
        return Datafilter
    try:
        ##Change the values of power, by your magnitude because the sign is not important because is only one register.
        Datafilter['P']=Datafilter['P'].abs()
        Datafilter['Q']=Datafilter['Q'].abs()
    except:
        pass
    ##Identify what variables are Null, If there is not one current: 
    variables_a_verificar=['VA','VB','VC','IA','IB','IC']
    hay_valor_nulo = Datafilter[variables_a_verificar].isnull().any().any()
    if hay_valor_nulo == False:
        ### Verify if colud get the sign of the power
        
        variables_a_verificar=['P','Q']
        hay_valor_nulo = Datafilter[variables_a_verificar].isnull().any().any()
        ### Calculate the aparent power 
        Datafilter=convert_variables_to_float(Datafilter)
        P= (Datafilter['VA'] * Datafilter['IA'] * 0.9 + Datafilter['VB'] * Datafilter['IB'] * 0.9 + Datafilter['VC'] * Datafilter['IC'] * 0.9 ) / 1000
        Q= (Datafilter['VA'] * Datafilter['IA'] * 0.435 + Datafilter['VB'] * Datafilter['IB'] * 0.435 + Datafilter['VC'] * Datafilter['IC'] * 0.435) / 1000
        
        if hay_valor_nulo == False:
            Potencia_Aparente_calculada=(P**2 + Q**2)**(1/2)
            Potencia_Aparente_Registrada=(Datafilter['P']**2 + Datafilter['Q']**2)**(1/2)
            ##Calculate the error percent.
            Porcentaje_error= float((Potencia_Aparente_calculada - Potencia_Aparente_Registrada).abs() / Potencia_Aparente_Registrada)
            if (umbral[0] != None and np.isnan(Porcentaje_error) != True):
                if (Porcentaje_error > umbral[0]):
                    Datafilter['P']=P 
                    Datafilter['Q']=Q 
                    Datafilter=correctSigno(Datafilter); ## Only change de sign
                else:
                    Datafilter=correctSigno(Datafilter); ## Correct the sign according to the last register
            else:
                Datafilter=correctSigno(Datafilter); ## Correct the sign according to the last register
        else:
            Datafilter['P']=P
            Datafilter['Q']=Q
            Datafilter=correctSigno(Datafilter)
    return Datafilter  
    




def Case_2_circuit(Datafilter):
    """ 
    DEESCRIPTION: Function to do the nodal and power analysis in the case of two circuits.
    --------------------------------------------------------
    --------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    --------------------------------------------------------
    --------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables and some imputed variables.
    """
    indices_cero = list(Datafilter.loc[((Datafilter['IA'] == 0) & (Datafilter['IB'] == 0) & (Datafilter['IC'] == 0))].index)
    if len(indices_cero) == 2:
        return Datafilter
    elif len(indices_cero) == 1:
        ###Take advantage of the first case
        circuito_cero = Datafilter.loc[indices_cero] ## This indicates that doesn´t have influence in the nodal analysis, is open, it mean has only one circuit
        circuito_no_cero = Datafilter.drop(indices_cero,axis=0) ## Handle as the case of one circuit
        circuito_no_cero = Case_1_circuit(circuito_no_cero)
        Datafilter = pd.concat([circuito_no_cero,circuito_cero],ignore_index=True)
        return Datafilter
    else:
        ##  Find how many pairs od currents in your respectuve phase is possible to compare.
        fases_completas = analizar_fases_completas(Datafilter)
        if len(fases_completas) == 0:
            ##  Check if for all the circuits was possible get the sign of the power.
            BoolPower = VerifyPowerWithoutRegister(Datafilter)
            if (BoolPower > 0):
                pass
            else:
                Columnas_Nulos = ['IA','IB','IC']
                ## Get the columns that only have one lost data, because only in this case is possible to impute.
                Columnas_Para_imputar = Verificar_columnas_para_imputar(Datafilter,Columnas_Nulos)
                if (len(Columnas_Para_imputar)==0):
                    pass 
                else:
                    Datafilter = NodalAnalysisWithoutFullPhases(Datafilter,Columnas_Para_imputar)
                    
        elif len(fases_completas) == 1:
            ##Claculate the error percent between the currents of the same phase.
            flag = calculate_error(Datafilter,fases_completas)
            if flag:
                ##Generate the report
                pass
            else:
                ##Case in the which can impute Null Values; One input and one output
                Datafilter = replaceCurrentsCase2(Datafilter);
        elif len(fases_completas) == 2:
            flag = calculate_error(Datafilter,fases_completas)
            if flag:
                pass
            else:
                ##Case in the which can impute Null Values; One input and one output
                Datafilter = replaceCurrentsCase2(Datafilter);
        elif len(fases_completas) == 3:
            flag = calculate_error(Datafilter,fases_completas)
            if flag:
                pass
        
        Datafilter = CheckPower(Datafilter) ## Check if is possible impute powers by historic calcules.
        return Datafilter



def Case_3_circuit(Datafilter):
    """ 
    DESCRIPTION: Function to do the nodal and power analysis in the case of three circuits.
    --------------------------------------------------------
    --------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    --------------------------------------------------------
    --------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables and some imputed variables.
    """
    
    ## Check if is possible to simplify using the last cases:
    indices_cero = list(Datafilter.loc[((Datafilter['IA'] == 0) & (Datafilter['IB'] == 0) & (Datafilter['IC'] == 0))].index) ## Check if there is some column with the data all in zero.
    if len(indices_cero) == 3:
        return Datafilter
    elif len(indices_cero) == 2:
        ##Use the case of one only circuit:
        circuito_cero = Datafilter.loc[indices_cero] ## This indicates that doesn´t have influence in the nodal analysis, is open, it mean has only one circuit
        circuito_no_cero = Datafilter.drop(indices_cero,axis=0) ## Handle as the case of one only circuit
        circuito_no_cero = Case_1_circuit(circuito_no_cero)
        Datafilter = pd.concat([circuito_no_cero,circuito_cero],ignore_index=True)
        return Datafilter
    elif len(indices_cero) == 1:
        ##Use the case of one only circuit
        circuito_cero = Datafilter.loc[indices_cero] ## This indicates that doesn´t have influence in the nodal analysis, is open, it mean has only one circuit
        circuito_no_cero = Datafilter.drop(indices_cero,axis=0) ## Handle as the case of two circuits
        circuito_no_cero = Case_2_circuit(circuito_no_cero)
        Datafilter = pd.concat([circuito_no_cero,circuito_cero],ignore_index=True)
        return Datafilter
    else:
        ##  Find how many pairs od currents in your respectuve phase is possible to compare.
        fases_completas = analizar_fases_completas(Datafilter)
        if len(fases_completas) == 0:
            ## Check if for every circuit was possible to find the sign of the power.
            BoolPower = VerifyPowerWithoutRegister(Datafilter)
            if (BoolPower > 1):
                pass
            else:
                Columnas_Nulos = ['IA','IB','IC']
                ## Get the columns that only have one lost data, because only in this case is possible to impute.
                Columnas_Para_imputar = Verificar_columnas_para_imputar(Datafilter,Columnas_Nulos)
                if (len(Columnas_Para_imputar)==0):
                    pass
                else:
                    Datafilter = NodalAnalysisWithoutFullPhases(Datafilter,Columnas_Para_imputar)
        ## Do a nodal analysis assuming that major of the currents is iqual to the sum of the others currents.
        elif len(fases_completas) == 1 :
            ##Calculate the error percnet of the currents of the same phase and the case in the which there is some with the full phase.
            flag = calcular_error_general(Datafilter,fases_completas)
            if flag:
                pass
            else:
                ## Get the columns that can be imputed;
                Columnas_Nulos = columnas_con_nulos(fases_completas)
                ## Get the columns only has one Null value,
                Columnas_Para_imputar = Verificar_columnas_para_imputar(Datafilter,Columnas_Nulos)
                if len(Columnas_Para_imputar) != 0 :
                    ##Get the column represent the sum:
                    target_row = getTargetRow(Datafilter,fases_completas);
                    ## Impute because only has one lost data.
                    Datafilter = replaceCurrentsGeneral(Datafilter,Columnas_Para_imputar,target_row)
                    ## Calculated the replaced data by nodal analysis.
                    

        elif len(fases_completas) == 2 :
            ##Calculate the percent error between the currents of the same phase and in case there is some full phase.
            flag = calcular_error_general(Datafilter,fases_completas)
            if flag:
                pass
            else:
                ## Get the columns that can be imputed.
                Columnas_Nulos = columnas_con_nulos(fases_completas)
                ## Get the columns that only has one lost data.
                Columnas_Para_imputar = Verificar_columnas_para_imputar(Datafilter,Columnas_Nulos)
                if len(Columnas_Para_imputar) != 0 :
                    ##Get the column represents the sum:
                    target_row = getTargetRow(Datafilter,fases_completas);
                    ## Impute, because only has one lost data
                    Datafilter = replaceCurrentsGeneral(Datafilter,Columnas_Para_imputar,target_row)
                    ## Dtermine the data replaced by nodal analysis

        elif len(fases_completas) == 3 :
            ##Calculate the error percent between the currents of the same phase and in case there is some full phase.
            flag = calcular_error_general(Datafilter,fases_completas)
            if flag:
                pass
        Datafilter = CheckPower(Datafilter) ## Chech if is possible to impute powers by historic calculates.
        return Datafilter





def Case_n_circuit(Datafilter):
    """ 
    DESCRIPTION: Function to do the nodal and power analysis in the case of n circuits.
    --------------------------------------------------------
    --------------------------------------------------------
    PARAMETERS:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables.
    --------------------------------------------------------
    --------------------------------------------------------
    RETURN:
    - Datafilter (pd.DataFrame): Dataframe with the interest variables and some imputed variables.
    """
    
    ##Analyse how many pairs in your respective phase there are
    fases_completas = analizar_fases_completas(Datafilter)
    if len(fases_completas) == 0:
        ## Check if for every circuit was possible to get the sign of the power
        BoolPower = VerifyPowerWithoutRegister(Datafilter)
        if (BoolPower < 2):
            Columnas_Nulos = ['IA','IB','IC']
            ## Get the columns only has one Null data, only in this case is possible to impute
            Columnas_Para_imputar = Verificar_columnas_para_imputar(Datafilter,Columnas_Nulos)
            if (len(Columnas_Para_imputar)!=0):
                Datafilter = NodalAnalysisWithoutFullPhases(Datafilter,Columnas_Para_imputar)
        Datafilter = CheckPower(Datafilter) ## Check if is possible to impute powers by historic calculates.
        return Datafilter

    else:
        ##Check if is possible to do nodal analysis, having the flow of the currents according to the sign of the last full register of the powers
        [Datafilter,flag,group_1,group_2] = checkNodalAnalysis(Datafilter,fases_completas)
        if (flag):
            pass
        else:
            ##Get the columns didn´t included in the imputation
            fases_restantes = columnas_con_nulos(fases_completas)
            ## Get the columnsonly has one Null data.
            Columnas_Para_imputar = Verificar_columnas_para_imputar(Datafilter,fases_restantes)
            if(Columnas_Para_imputar != 0):
                ##Nodal Analysis
                Datafilter = nodalCurrents(Datafilter,group_1,group_2,Columnas_Para_imputar)# Sent the dataframe, the groups and the phases
    Datafilter = CheckPower(Datafilter) ## Check if is possible to impute powers by historic analysis.
    return Datafilter




def iterateByUniqueDate(dataframe,arrayUniqueDate):
    """ 
    DESCRIPTION: Function to do the nodal analysis per every date.
    --------------------------------------------------------------
    --------------------------------------------------------------
    PARAMETERS:
    - dataframe (pd.DataFrame): Dataframe with the interest variables.
    - arrayUniqueDate (List): List with the unique dates.
    --------------------------------------------------------------
    --------------------------------------------------------------
    RETURN:
    - dataframe_post_algoritmo (pd.DataFrame): Dataframe gotten with the nodal analysis.
    """

    # Create a empty dataframe with the specified columns.
    columns = ['CIRCUITO','TIEMPO_AJUSTADO','IA', 'IB', 'IC', 'VA', 'VB', 'VC', 'P', 'Q']
    dataframe_post_algoritmo = pd.DataFrame(columns=columns)
    pool = Pool(1)
    results = pool.map(lambda args: process_date(args[0], args[1]), zip(arrayUniqueDate, [dataframe]*len(arrayUniqueDate)))
    dataframe_post_algoritmo = pd.concat([dataframe_post_algoritmo] + results, axis=0)
    try:
        variables_correct=["IA","IB","IC","VA","VB","VC","P","Q"]
        for i in variables_correct:
            dataframe_post_algoritmo[i]=dataframe_post_algoritmo[i].astype(float)
    except Exception as e:
        pass
    
    return dataframe_post_algoritmo


def InitProgram():
    #### CARGAMOS LOS MODELOS VAE QUE PODEMOS USAR
    VAE_MODELS  = VaeModel() 
    
    #Read the json file with the credentials:
    with open("Project/Model_reglas_impuestas/Archivo_de_Credenciales.json", 'r') as archivo:
        diccionario_cargado = json.load(archivo)

    # Credentials:
    server = diccionario_cargado["server"]
    username = diccionario_cargado["username"]
    password = diccionario_cargado["password"]
    driver = diccionario_cargado["driver"]
    conexion_str = f"DRIVER={driver};SERVER={server};UID={username};PWD={password}"

    #Current day:
    fecha_actual = datetime.datetime.now()
    fecha_una_semana_atras = fecha_actual - datetime.timedelta(days=7) #A week back
    # Dates with the format of the database
    fecha_actual_str = fecha_actual.strftime('%Y-%m-%d %H:%M:%S')
    fecha_una_semana_atras_str = fecha_una_semana_atras.strftime('%Y-%m-%d %H:%M:%S')

    names_sistemas=["MEDIDAS_ANALOGAS_HORIZONTALES","MEDIDAS_ANALOGAS_SURVALENT_H"]
    counter=0
    for sistema in names_sistemas:
        
        #Indicator_system:
        if sistema=="MEDIDAS_ANALOGAS_HORIZONTALES":
            indicator_sistema="ABB"
        else:
            indicator_sistema="SURVALENT"
        
        # All the circuits that there is in database (Cabecera y Red):
        try:
            #Connection:
            conexion = pyodbc.connect(conexion_str)
        
            # Create cursor to execute queries:
            cursor = conexion.cursor()
            consulta_sql = f"""
            SELECT DISTINCT LEFT(CIRCUITO, 5) AS primeros_cinco_caracteres
            FROM DM_OPERACION.dbo.{sistema}
            WHERE TIEMPO_AJUSTADO BETWEEN CONVERT(DATETIME, ?) AND CONVERT(DATETIME, ?);"""
            
            # Execute the query:
            cursor.execute(consulta_sql, (fecha_una_semana_atras_str, fecha_actual_str))

            # Get the query results:
            resultados =np.array(cursor.fetchall())
            nombres_columnas = np.array([columna[0] for columna in cursor.description]) #Name of the columns of the query
            # Create the dataframe with the query:
            DF_BARRAS_ABB = pd.DataFrame(resultados, columns=nombres_columnas)
            cursor.close() #Close the cursor
            conexion.close() #Clos the query              
                
        except pyodbc.Error as e:
            print("Error al conectar a la base de datos:", e)
            
        #Filter the circuits codes by your nomenclature:
        DF_BARRAS_ABB=DF_BARRAS_ABB['primeros_cinco_caracteres'].values
        DF_FILTRADO_ABB = [elemento for elemento in DF_BARRAS_ABB if cumple_condicion(elemento)]
            
        
        initialCounts(DF_FILTRADO_ABB); ## Start all the counters for the analysis
        for i in range(0,len(DF_FILTRADO_ABB)): #For every cabecera or circuito_red
            try:
                barra=[DF_FILTRADO_ABB[i]]
                # Conexion to database:
                conexion = pyodbc.connect(conexion_str)
                #Create cursor to execute queries:
                cursor = conexion.cursor()
                
            
                # Query:
                consulta_sql = f"""
                SELECT [CIRCUITO], [TIEMPO_AJUSTADO], [IA], [IB], [IC], [VA], [VB], [VC], [P], [Q]
                FROM (
                    SELECT
                        [CIRCUITO],
                        [TIEMPO_AJUSTADO],
                        [IA],
                        [IB],
                        [IC],
                        [VA],
                        [VB],
                        [VC],
                        [P],
                        [Q],
                        ROW_NUMBER() OVER (PARTITION BY [CIRCUITO] ORDER BY [TIEMPO_AJUSTADO] DESC) as row_num
                    FROM [DM_OPERACION].[dbo].[{sistema}]
                    WHERE CIRCUITO LIKE ? AND TIEMPO_AJUSTADO <= CONVERT(DATETIME, ?)
                ) AS Subconsulta
                WHERE row_num <= 672
                ORDER BY [CIRCUITO], [TIEMPO_AJUSTADO] DESC;
                """

                cursor.execute(consulta_sql, ('%' + DF_FILTRADO_ABB[i] + '%', fecha_actual_str))
                #Get the results of the query:
                resultados = np.array(cursor.fetchall())
                nombres_columnas = np.array([columna[0] for columna in cursor.description])
                df_MAGNITUDES_ABB = pd.DataFrame(resultados, columns=nombres_columnas)
                
                
                    
                #Get the unique Dates.
                
                df_MAGNITUDES_ABB = convertToDate(df_MAGNITUDES_ABB)
                #Replace the voltage and current values by your absolute
                df_MAGNITUDES_ABB = Absolute(df_MAGNITUDES_ABB)
                df_MAGNITUDES_ABB = ReScale(df_MAGNITUDES_ABB)
                #Correct the power and voltaje values out of the normal range:
                uniqueDates = getUniqueTime(df_MAGNITUDES_ABB)
                
                
                
                # Umbarl by cabecera and iter per time instant:
                getUmbrals(df_MAGNITUDES_ABB)
                findLastRegister(df_MAGNITUDES_ABB)
                
                
                
                dataframe_post_algoritmo=iterateByUniqueDate(df_MAGNITUDES_ABB,uniqueDates)
                
                # Create the confiabily columns:
                columnas_a_evaluar = ['IA', 'IB', 'IC', 'VA', 'VB', 'VC', 'P', 'Q']
                for columna in columnas_a_evaluar:
                    nombre_columna_confiabilidad = 'CONFIABILIDAD_' + columna
                    dataframe_post_algoritmo[nombre_columna_confiabilidad] = np.where(dataframe_post_algoritmo[columna].notnull(), 1, np.nan)  # 1 if is not NULL, NULL if is NULL
                
                #WHAT SCADA System:
                dataframe_post_algoritmo["SCADA"]=indicator_sistema
                
                
                
                cursor.close() #Close the cursor
                conexion.close() #Close the query 
                
                VAE_MODELS.model_iteration(dataframe_post_algoritmo)


                    
            except Exception as e:
                print("Error al generar la consulta:", e)
        

    
        



