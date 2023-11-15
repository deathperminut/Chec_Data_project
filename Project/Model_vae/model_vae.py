
###############################
##########LIBRERIAS############
############data###############
###############################

import pandas as pd
import numpy as np
from scipy import stats

###############################
########DATABASE###############
###############################

import json
import pyodbc

###############################
########MODEL##################
###############################

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf


###############################
########CLASSES################
###############################

class reparametrize(tf.keras.layers.Layer):
    """
    function to 
    -----------------------------------------
    tf.keras.layers.Layer => layout related with 
    the latent space of the autoencoder this
    help to generate de custom loss function
    """
    def call(self, inputs):
        mean, log_var = inputs
        eta = tf.random.normal(tf.shape(log_var))
        sigma = tf.math.exp(log_var / 2)
        return  mean + sigma * eta


class VaeModel():
      """
        VaeModel
        ------------------------------

        function to control all the models for the 
        imputation process of a specific dataset. 

      """
      def __init__(self):
         

         self.generate_models()### generate all the models 10,2,3,GOOD MODELS
         self.init_server() ### function for specify the credentials of the server.

      def init_server(self):
          """
          function to specify all the credentials to connect to the database
          ------------------------------------------------------------------

          all the information of this is in the json file "Archivo_de_Credenciales"

          """
          #Read the json file with the credentials for database:
          with open("Project/Model_vae/Archivo_de_Credenciales_escritura.json", 'r') as archivo:
            credenciales = json.load(archivo)
          # Set the connection string to connect to the database
          self.server = "10.46.6.56\CHECSQLDES"
          self.username = credenciales['username']
          self.password = credenciales['password']
          self.driver = '{ODBC Driver 17 for SQL Server}'
          self.database = 'DM_OPERACION'
          self.conexion_str = f"DRIVER={self.driver};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"


      def VAE_LATENT_DIM_(self,Chans, Samples, dropoutRate = 0.5, l1 = 0, l2 = 0,latent_dim = 160):
            """
            FUNCTION TO GET THE ARQUITECTURE OF THE MODEL VAE
            -----------------------------------
            Chans = Number of signals
            Samples = samples of  time serie
            dropoutRate = hyperparam for training
            l1, l2  = Regularization parameter
            latent_dim = dimensions for latent space
            """

            filters      = (1,40) ##ESTRUCTURA BASE DEL MODELO MULTITASK
            strid        = (1,15) ##ESTRUCTURA BASE DEL MODELO MULTITASK
            pool         = (1,75) ## ESTRUCTURA BASE DEL MODELO MULTITASK
            bias_spatial = True ## QUE ES ESTE PARAMETRO?
            ## ENCODER
            input_main   = Input((Chans, Samples, 1))
            block1       = Conv2D(latent_dim, filters, strides=(1,2),
                                        input_shape=(Chans, Samples, 1),kernel_regularizer=l1_l2(l1=l1,l2=l2),
                                        name='Conv2D_1',
                                        kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
            block1       = Conv2D(latent_dim, (Chans, 1), use_bias=bias_spatial, kernel_regularizer=l1_l2(l1=l1,l2=l2),
                                name='Conv2D_2',
                                kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
            block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
            Act1         = Activation('elu')(block1)
            block1       = AveragePooling2D(pool_size=pool, strides=strid)(Act1)
            block1       = Dropout(dropoutRate,name='bottleneck')(block1) ## ENCODER
            ### 
            mu           = Dense(latent_dim,name='mu')(block1) ##
            log_var      = Dense(latent_dim,name='log_var')(block1)
            codings      = reparametrize(name='Code')([mu, log_var]) ## CODIFICAMOS CON LA MEDIA Y VARIANZA DADA



            ##DECODER
            block2       = Conv2DTranspose(latent_dim, pool,strides=strid,activation='tanh', kernel_regularizer=l1_l2(l1=l1,l2=l2),
                                    kernel_constraint = max_norm(2., axis=(0,1,2)))(codings)
            block2       = Resizing(block2.shape[1], Act1.shape[2])(block2)
            block2       = Conv2DTranspose(latent_dim, (Chans, 1), use_bias=bias_spatial, kernel_regularizer=l1_l2(l1=l1,l2=l2),
                                    kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
            block2       = Conv2DTranspose(1, filters,strides=(1,2),
                                    input_shape=(Chans, Samples, 1),kernel_regularizer=l1_l2(l1=l1,l2=l2),
                                    kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)

            model = Model(inputs=input_main, outputs=[block2])

            return model


      def vae_loss(self,mu, log_var):
            """
            function to define the loss function of the VAE model,
            based on the kl divergence and mse
            """
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
            total_loss = kl_loss

            return total_loss

      def generate_models(self):
            """
            function to define each of the loaded models
            with saved weights
            -----------------------------------------

            parameters
            ------------------------
            None

            Return
            ........................

            None
            """
            #######################################
            ################ 10 ###################
            #######################################

            ##### WE DEFINE THE AMOUNT OF SIGNALS
            Channels = 6 
            #### TIME SERIES SAMPLES PER WEEK
            Samples = 672
            ####we build the model with the required latent space dimensions
            self.MODEL_10 = self.VAE_LATENT_DIM_(Chans=Channels,Samples = Samples,latent_dim = 260) 
            # WE OBTAIN THE INPUTS AND OUTPUTS OF THE MODEL
            inputs = self.MODEL_10.input
            outputs = self.MODEL_10.output
            # Get the output layers of the latent space (mu and log_var)
            mu_layer = self.MODEL_10.get_layer('mu')
            log_var_layer = self.MODEL_10.get_layer('log_var')

            # Obtener las salidas de las capas de mu y log_var para reparametrizar
            mu_output = mu_layer.output
            log_var_output = log_var_layer.output

            # Compile the model with the custom loss function
            self.MODEL_10.add_loss(self.vae_loss(mu_output,log_var_output))
            self.MODEL_10.compile(optimizer='adam',loss=['mse'])

            ###CARGAMOS LOS PESOS
            self.MODEL_10.load_weights('Project/Model_vae/Pesos/modelo_10_Class_v1 (1) (1).h5')


            #######################################
            ###############  2  ###################
            #######################################
            ####we build the model with the required latent space dimensions
            self.MODEL_2 = self.VAE_LATENT_DIM_(Chans=Channels,Samples = Samples,latent_dim = 340) ## OBTENEMOS EL MODELO
            # OBTENEMOS LAS ENTRADAS Y SALIDAS DEL MODELO
            inputs = self.MODEL_2.input ##ENTRADA
            outputs = self.MODEL_2.output ##SALIDAS [reconstrucción,clasificación]
            # Obtener las capas de salida del espacio latente (mu y log_var)
            mu_layer = self.MODEL_2.get_layer('mu')
            log_var_layer = self.MODEL_2.get_layer('log_var')

            # Obtener las salidas de las capas de mu y log_var
            mu_output = mu_layer.output
            log_var_output = log_var_layer.output

            # Compile the model with the custom loss function
            self.MODEL_2.add_loss(self.vae_loss(mu_output,log_var_output))
            self.MODEL_2.compile(optimizer='adam',loss=['mse'])

            ###CARGAMOS LOS PESOS
            self.MODEL_2.load_weights('Project/Model_vae/Pesos/model_2_340 (1) (1).h5')




            #######################################
            ###############  3  ###################
            #######################################


            self.MODEL_3 = self.VAE_LATENT_DIM_(Chans=Channels,Samples = Samples,latent_dim = 260) ## OBTENEMOS EL MODELO
            # OBTENEMOS LAS ENTRADAS Y SALIDAS DEL MODELO
            inputs = self.MODEL_3.input ##ENTRADA
            outputs = self.MODEL_3.output ##SALIDAS [reconstrucción,clasificación]
            # Obtener las capas de salida del espacio latente (mu y log_var)
            mu_layer = self.MODEL_3.get_layer('mu')
            log_var_layer = self.MODEL_3.get_layer('log_var')

            # Obtener las salidas de las capas de mu y log_var
            mu_output = mu_layer.output
            log_var_output = log_var_layer.output

            # Compile the model with the custom loss function
            self.MODEL_3.add_loss(self.vae_loss(mu_output,log_var_output))
            self.MODEL_3.compile(optimizer='adam',loss=['mse'])

            ###CARGAMOS LOS PESOS
            self.MODEL_3.load_weights('Project/Model_vae/Pesos/modelo_3_Class_v1_260 (1).h5')


            #######################################
            ################ GC ###################
            #######################################


            self.MODEL_GOOD = self.VAE_LATENT_DIM_(Chans=Channels,Samples = Samples,latent_dim = 560) ## OBTENEMOS EL MODELO
            # OBTENEMOS LAS ENTRADAS Y SALIDAS DEL MODELO
            inputs = self.MODEL_GOOD.input ##ENTRADA
            outputs = self.MODEL_GOOD.output ##SALIDAS [reconstrucción,clasificación]
            # Obtener las capas de salida del espacio latente (mu y log_var)
            mu_layer = self.MODEL_GOOD.get_layer('mu')
            log_var_layer = self.MODEL_GOOD.get_layer('log_var')

            # Obtener las salidas de las capas de mu y log_var
            mu_output = mu_layer.output
            log_var_output = log_var_layer.output

            # Compile the model with the custom loss function
            self.MODEL_GOOD.add_loss(self.vae_loss(mu_output,log_var_output))
            self.MODEL_GOOD.compile(optimizer='adam',loss=['mse'])

            ###CARGAMOS LOS PESOS
            self.MODEL_GOOD.load_weights('Project/Model_vae/Pesos/modelo_GOODClasses_560 (1).h5')


            ###################################
            ############## ALL ################
            ###################################


            self.MODEL_ALL = self.VAE_LATENT_DIM_(Chans=Channels,Samples = Samples,latent_dim = 560) ## OBTENEMOS EL MODELO
            # OBTENEMOS LAS ENTRADAS Y SALIDAS DEL MODELO
            inputs = self.MODEL_ALL.input ##ENTRADA
            outputs = self.MODEL_ALL.output ##SALIDAS [reconstrucción,clasificación]
            # Obtener las capas de salida del espacio latente (mu y log_var)
            mu_layer = self.MODEL_ALL.get_layer('mu')
            log_var_layer = self.MODEL_ALL.get_layer('log_var')

            # Obtener las salidas de las capas de mu y log_var
            mu_output = mu_layer.output
            log_var_output = log_var_layer.output

            # Compile the model with the custom loss function
            self.MODEL_ALL.add_loss(self.vae_loss(mu_output,log_var_output))
            self.MODEL_ALL.compile(optimizer='adam',loss=['mse'])

            ###CARGAMOS LOS PESOS
            self.MODEL_ALL.load_weights('Project/Model_vae/Pesos/modelo_GOODClasses_560 (1).h5')



      def model_iteration(self,df_circuits):

            """
            
            function to iterate through each time series of each circuit
            -------------------------------------------------------------
            parameters
            .................................
            df_circuits (pandas dataframe)=> dataframe with the information of time serie
    
            return
            ------------------------------
            None

            """
            #### obtenemos los circuitos unicos del dataframe
            circuits = df_circuits['CIRCUITO'].unique()

            ### iteramos por cada uno de esos circuitos

            for circuit in circuits:

                ###### obtenemos el dataframe referente a dicho circuito

                df_circuit = df_circuits.loc[df_circuits['CIRCUITO'] == circuit].sort_values(by='TIEMPO_AJUSTADO', ascending=True)

                ###### segmentamos el dataframe por cada 672 datos e iteramos
                self.iterar_por_grupos(df_circuit,672)


      def iterar_por_grupos(self,dataframe, tamano_grupo):
            
            """
            
            function to iterate through each time series of each circuit
            with a longitud of 672 samples
            -------------------------------------------------------------
            parameters
            .................................
            dataframe (pandas dataframe)=> dataframe with the information of one time serie of 672 samples
                                           that is the samples of a week.

            tamano_grupo (int) => longitude of dataframe standar 672 samples
                                           
            

            return
            ------------------------------

            None

            """
            # Obtener la cantidad total de filas en el DataFrame
            total_filas = len(dataframe)

            # Calcular la cantidad de grupos necesarios
            cantidad_grupos = total_filas % tamano_grupo

            if (cantidad_grupos == 0):
               cantidad_grupos = (total_filas // tamano_grupo)
            else:
               cantidad_grupos = (total_filas // tamano_grupo) + 1
            # Iterar sobre los grupos
            for i in range(cantidad_grupos):
                # Calcular el índice de inicio y fin para el grupo actual
                inicio = i * tamano_grupo
                fin = min((i + 1) * tamano_grupo, total_filas)

                # Obtener el grupo actual
                grupo = dataframe.iloc[inicio:fin]

                # Verificar si el tamaño del grupo es menor a 672
                if len(grupo) < tamano_grupo:
                    # Arreglamos la serie de tiempo para que cumpla con 672
                    grupo_completo = self.completeGroup(grupo,672)
                    grupo_imputado = self.processData(grupo_completo)
                    self.insertDataBase(grupo_imputado.iloc[:len(grupo)])

                else:
                    ### si armamos un grupo de 672 directamente pasamos a preprocesamiento del dataset
                    grupo_imputado = self.processData(grupo)

                    ### insertamos en base de datos
                    self.insertDataBase(grupo_imputado)

      def completeGroup(self,df,objetivo_filas):
          """
          function to complete the dimensions required to be introduced to the vae model
          ------------------------------------------------------------------------------------


          Parameters
          ----------
          df (pandas dataframe) :
              incomplete dataframe to modify
          objetivo_filas (int): 
              number of necessary rows needed in the dataframe

          Returns
          -------
          df_completo (pandas dataframe) :
             complete dataframe with the neccesary rows
          """
          # Calcular cuántas veces necesitas repetir el DataFrame
          repeticiones = -(-objetivo_filas // len(df))  # Equivalente a math.ceil(objetivo_filas / len(df))
          # Repetir el DataFrame
          df_repetido = pd.concat([df] * repeticiones, ignore_index=True)
          # Seleccionar las primeras objetivo_filas filas
          df_completo = df_repetido.head(objetivo_filas)
          return df_completo


      def insertDataBase(self,df):
            
            """
            function to insert the dataframe into the specific database
            ------------------------------------------------------------------------------------


            Parameters
            ----------
            df (pandas dataframe) :
                complete dataframe after model

            Returns
            -------
            None
            """
            try:
                 conexion = pyodbc.connect(self.conexion_str)

                 # Crear un DataFrame (simulando que tienes un DataFrame llamado DF_BARRAS_ABB)

                 # Crear un cursor para ejecutar consultas
                 cursor = conexion.cursor()

                 # Construir la consulta de inserción con todas las columnas
                 consulta_insert = "INSERT INTO dbo.SCADA_UNIFICADO (CIRCUITO, TIEMPO_AJUSTADO, IA,IB,IC,VA,VB,VC,P,Q,CONFIABILIDAD_IA,CONFIABILIDAD_IB,CONFIABILIDAD_IC,CONFIABILIDAD_VA,CONFIABILIDAD_VB,CONFIABILIDAD_VC,CONFIABILIDAD_P,CONFIABILIDAD_Q,SCADA) VALUES (?, ?, ?, ?, ?,?, ?, ?, ?, ?,?, ?, ?, ?, ?,?, ?, ?, ?)"

                 # Iterar sobre las filas del DataFrame e insertar cada fila en la base de datos
                 for index, row in df.iterrows():
                     cursor.execute(consulta_insert, tuple(row))

                 # Confirmar los cambios en la base de datos
                 conexion.commit()

                 print("Datos insertados exitosamente en la tabla.")
            except pyodbc.Error as e:
                 print("Error al conectar a la base de datos:", e)
            finally:
                 conexion.close()


      def processData(self,df):
                """
                process data from a selected dataframe of a specific circuit
                ----------------------------------------------------------------
                params
                ----------------------------------
                df => dataframe : that represent a seccion of time with 672 samples
               
                return
                ---------------------------------

                df => (pandas dataframe) : dataframe complete after the model VAE.

                """
                ##########################################
                ####### NORMALIZAMOS LOS DATOS ###########
                ##########################################
                name_circuit = df['CIRCUITO'].unique()[0]

                lista_10 = []
                lista_series_tiempo = []
                Data_10 = []

                df_matriz=df.values[:,1:8] ## eliminamos la columna del nombre del circuito que no nos interesa por el momento
                df_matriz_total = df.values
                # Tamaño de la serie de tiempo completa
                tamaño_completo = df_matriz.shape[0]

                # Tamaño deseado de las series segmentadas
                tamaño_segmento = 672 ## SEGMENTADO POR SEMANA
                channels = 7 ## CANTIDAD DE CANALES
                # Número de series de tiempo
                num_series = df_matriz.shape[1]

                # Calcular la cantidad de series segmentadas
                cantidad_series_segmentadas = tamaño_completo // tamaño_segmento
                timesList = []

                if (cantidad_series_segmentadas != 0):

                    # OBTENEMOS LAS SERIES DE TIEMPO
                    start = 0
                    end = 0
                    lista = []
                    datos = np.zeros((cantidad_series_segmentadas,channels,tamaño_segmento))
                    Label = []

                    for i in range(1 , cantidad_series_segmentadas+1):
                        start = end
                        end = start + 672

                        TIME = df_matriz[start:end,0]
                        timesList.append(TIME)
                        serie_ia = df_matriz[start:end,1]
                        serie_ib = df_matriz[start:end,2]
                        serie_ic = df_matriz[start:end,3]

                        serie_va = df_matriz[start:end,4]
                        serie_vb = df_matriz[start:end,5]
                        serie_vc = df_matriz[start:end,6]


                        datos[i-1,1,:] = serie_ia
                        datos[i-1,2,:] = serie_ib
                        datos[i-1,3,:] = serie_ic

                        datos[i-1,4,:] = serie_va
                        datos[i-1,5,:] = serie_vb
                        datos[i-1,6,:] = serie_vc

                    Label = [10] * cantidad_series_segmentadas
                    lista_10.append([datos,Label])
                    lista_series_tiempo.append([datos,Label])



                Data_10 = lista_10[0][0]
                Label_10 = lista_10[0][1]

                for i in range (1,len(lista_10)):
                  Data_10 = np.concatenate((Data_10, lista_10[i][0]), axis=0)

                ###YA TENEMOS LA SERIE DE TIEMPO DE LA FORMA QUE REQUIERE EL MODELO EN DATA_10
                ###AHORA APLICAMOS EL PREPROCESAMIENTO NECESARIO

                Data = np.copy(Data_10[:,1:,:])
                datos_normalizados = np.zeros_like(Data_10[:,1:,:], dtype=float)
                Lista_nulos = []
                Lista_reemplazo_z_score=[]
                Lista_limites = np.zeros_like(Data[:,:,:2], dtype=float)

                for i in range(Data.shape[0]):
                  # Iterar a través de cada serie de tiempo en el conjunto de datos
                  for j in range(Data.shape[1]):
                      # Obtener la serie de tiempo actual
                      """
                      MIRAR QUE LOS NULOS NO AFECTEN Y CONSEGUIR LOS INDICES DE LOS VALORES NULOS
                      """
                      serie_tiempo = Data[i, j, :]
                      index_null = np.where(np.isnan(serie_tiempo))
                      Lista_nulos.append(index_null)### guardamos la lista de los indices de valores nulos por serie de tiempo
                      mediana_arreglo = np.nanmedian(serie_tiempo) ## SIN TENER ENCUENTA LOS NULOS
                      ### reemplazamos los nulos por la media momentaneamente para retirar
                      serie_tiempo[index_null] = mediana_arreglo
                      # Detectar valores atípicos en la serie de tiempo usando el método Z-score
                      z_scores = np.abs(stats.zscore(serie_tiempo))
                      # Definir un umbral para considerar valores atípicos (por ejemplo, 2 desviaciones estándar)
                      umbral = 2.0
                      index_reemplazada = z_scores > umbral
                      index_reemplazada = np.where( index_reemplazada == True)[0]
                      valores_reemplazada = serie_tiempo[index_reemplazada]
                      Lista_reemplazo_z_score.append([index_reemplazada,valores_reemplazada])
                      # Reemplazar valores atípicos por la mediana de la serie de tiempo
                      serie_tiempo[z_scores > umbral] = mediana_arreglo
                      #### NORMALIZAMOS LA SERIE DE TIEMPO
                      min_ = np.min(serie_tiempo)
                      max_ = np.max(serie_tiempo)
                      if(min_ == max_):
                        Lista_limites[i, j, :]=np.array([min_,max_])
                        serie_tiempo = serie_tiempo/min_
                        serie_tiempo[index_null] = min_ ### ubicamos los nulos como negativos
                        # Almacenar la serie de tiempo normalizada en la matriz de datos normalizados

                      else:
                        Lista_limites[i, j, :]=np.array([min_,max_])
                        serie_tiempo = (serie_tiempo - np.min(serie_tiempo)) / (np.max(serie_tiempo) - np.min(serie_tiempo))
                        serie_tiempo[index_null] = -0.2 ### ubicamos los nulos como negativos
                        # Almacenar la serie de tiempo normalizada en la matriz de datos normalizados

                      datos_normalizados[i, j, :] = serie_tiempo

                #### UNA VEZ NORMALIZADA LA SERIE DE TIEMPO LO QUE HACEMOS ES DECIDIR A QUE MODELO HACER LA INFERENCIA
                #### obtenemos el modelo dependiendo del tipo de circuito
                model,model_name = self.findModel(name_circuit)
                ### hacemos la inferencia con el modelo
                X_model = model.predict(datos_normalizados)
                X_model = X_model.reshape((1, 6, 672))
                #### Con los datos del modelo predicho sacamos reemplazamos cada serie de tiempo por sus nulos
                r2_serie = []
                for i in range(0,len(Lista_nulos)):
                    ### reemplazamos los valores nulos por los imputados por el modelo
                    if (Lista_limites[0,i,0] != Lista_limites[0,i,1]):
                        datos_normalizados[0,i,Lista_nulos[i]] = X_model[0,i,Lista_nulos[i]]
                        r2_serie.append(self.calc_r2(len(Lista_nulos[i]),model_name,i))
                    else:
                        r2_serie.append(1)
                ### CALCULAMOS EL R2 DE LO OBTENIDO POR EL MODELO Y LA SERIE DE TIEMPO IMPUTADA CON LOS DATOS DEL MODELO

                for i in range(0,len(Lista_nulos)):
                    if (Lista_limites[0,i,0] != Lista_limites[0,i,1]):
                        datos_normalizados[0,i,:]=(datos_normalizados[0,i,:] * (Lista_limites[0,i,1] - Lista_limites[0,i,0]) ) + Lista_limites[0,i,0]
                        datos_normalizados[0,i,Lista_reemplazo_z_score[i][0]]=Lista_reemplazo_z_score[i][1]
                    else:
                        datos_normalizados[0,i,:]=datos_normalizados[0,i,:] * Lista_limites[0,i,1]
                        datos_normalizados[0,i,Lista_reemplazo_z_score[i][0]]=Lista_reemplazo_z_score[i][1]

                #### Restructuramos para devolver un dataframe con todos los datos imputados
                ##CORRIENTES
                df_matriz_total[:,2] = datos_normalizados[0,0] ## ia
                df_matriz_total[:,3] = datos_normalizados[0,1] ## ib
                df_matriz_total[:,4] = datos_normalizados[0,2] ## ic

                ##VOLTAJES
                df_matriz_total[:,5] = datos_normalizados[0,3] ## va
                df_matriz_total[:,6] = datos_normalizados[0,4] ## vb
                df_matriz_total[:,7] = datos_normalizados[0,5] ## vc


                ### REEMPLAZAMOS LOS VALORES DE CONFIABILIDAD DEL DATO DE LOS VOLTAJES Y CORRIENTES IMPUTADOS
                ##CORRIENTES
                df_matriz_total[Lista_nulos[0],10] = [r2_serie[0]]*len(Lista_nulos[0]) ## ia
                df_matriz_total[Lista_nulos[1],11] = [r2_serie[1]]*len(Lista_nulos[1]) ## ib
                df_matriz_total[Lista_nulos[2],12] = [r2_serie[2]]*len(Lista_nulos[2]) ## ic

                ##VOLTAJES
                df_matriz_total[Lista_nulos[3],13] = [r2_serie[3]]*len(Lista_nulos[3]) ## va
                df_matriz_total[Lista_nulos[4],14] = [r2_serie[4]]*len(Lista_nulos[4]) ## vb
                df_matriz_total[Lista_nulos[5],15] = [r2_serie[5]]*len(Lista_nulos[5]) ## vc
                
                df.loc[:, :] = df_matriz_total
                ##CALCULAMOS LAS POTENCIAS DE CADA FILA.
                sign_data =  self.findLastRegister(df) ## obtenemos el signo de las potencias

                if (sign_data != None) :
                    df['P'].fillna( (sign_data[0]*(df['VA'] * df['IA'] * 0.9 + df['VB'] * df['IB'] * 0.9 + df['VC'] * df['IC'] * 0.9))/1000,inplace = True)
                    df['Q'].fillna( (sign_data[1]*(df['VA'] * df['IA'] * 0.435 + df['VB'] * df['IB'] * 0.435 + df['VC'] * df['IC'] * 0.435))/1000,inplace = True)
                else:
                    df['P'].fillna( (df['VA'] * df['IA'] * 0.9 + df['VB'] * df['IB'] * 0.9 + df['VC'] * df['IC'] * 0.9)/1000,inplace = True)
                    df['Q'].fillna( (df['VA'] * df['IA'] * 0.435 + df['VB'] * df['IB'] * 0.435 + df['VC'] * df['IC'] * 0.435)/1000,inplace = True)
                
                df['CONFIABILIDAD_P'] = df.apply(lambda row: row[['CONFIABILIDAD_IA', 'CONFIABILIDAD_IB', 'CONFIABILIDAD_IC', 'CONFIABILIDAD_VA', 'CONFIABILIDAD_VB', 'CONFIABILIDAD_VC']].mean() if pd.isna(row['CONFIABILIDAD_P']) else row['CONFIABILIDAD_P'], axis=1)
                df['CONFIABILIDAD_Q'] = df.apply(lambda row: row[['CONFIABILIDAD_IA', 'CONFIABILIDAD_IB', 'CONFIABILIDAD_IC', 'CONFIABILIDAD_VA', 'CONFIABILIDAD_VB', 'CONFIABILIDAD_VC']].mean() if pd.isna(row['CONFIABILIDAD_Q']) else row['CONFIABILIDAD_Q'], axis=1)

                return df


      def calc_r2(self,cantidad_perdidos,modelo,variable):
            """
            function that calculated the fiability of the model of each time serie
            ----------------------------------------------------------------
            params
            ----------------------------------
            cantidad_perdidos => int : quantity of loss values.
            modelo =>  string : type of model.
            variable => int  : represent the time serie that we are focus.
            
            return
            ---------------------------------

            r2 => (float) : r2 for model.

            """
            dictionario = {
                '10':{
                    '0':{
                        'datos':[0.97,0.90,0.8,0.72,0.56,0.47],
                        'limite':[50,100,200,300,400]
                        },
                    '1':{
                        'datos':[0.97,0.90,0.8,0.72,0.56,0.47],
                        'limite':[50,100,200,300,400]
                        },
                    '2':{
                        'datos':[0.97,0.90,0.8,0.67,0.61,0.52],
                        'limite':[50,100,200,300,400]
                        },
                    '3':{
                        'datos':[0.97,0.90,0.8,0.67,0.61,0.52],
                        'limite':[50,100,200,300,400]
                        },
                    '4':{
                        'datos':[0.98,0.91,0.85,0.75,0.6,0.53],
                        'limite':[50,100,200,300,400]
                        },
                    '5':{
                        'datos':[0.97,0.90,0.87,0.77,0.68,0.51],
                        'limite':[50,100,200,300,400]
                        },
                    },
                '3':{
                    '0':{
                        'datos':[0.98,0.93,0.87,0.63,0.45,0.38],
                        'limite':[50,100,200,300,400]
                        },
                    '1':{
                        'datos':[0.99,0.93,0.88,0.67,0.44,0.37],
                        'limite':[50,100,200,300,400]
                        },
                    '2':{
                        'datos':[0.98,0.93,0.87,0.68,0.46,0.41],
                        'limite':[50,100,200,300,400]
                        },
                    '3':{
                        'datos':[0.96,0.90,0.8,0.57,0.41,0.13],
                        'limite':[50,100,200,300,400]
                        },
                    '4':{
                        'datos':[0.98,0.91,0.8,0.57,0.31,0.18],
                        'limite':[50,100,200,300,400]
                        },
                    '5':{
                        'datos':[0.97,0.88,0.82,0.58,0.33,0.21],
                        'limite':[50,100,200,300,400]
                        },
                },
                '2':{
                    '0':{
                        'datos':[0.97,0.89,0.83,0.67,0.46,0.28],
                        'limite':[50,100,200,300,400]
                        },
                    '1':{
                        'datos':[0.97,0.9,0.88,0.68,0.5,0.33],
                        'limite':[50,100,200,300,400]
                        },
                    '2':{
                        'datos':[0.97,0.9,0.8,0.67,0.52,0.34],
                        'limite':[50,100,200,300,400]
                        },
                    '3':{
                        'datos':[0.97,0.9,0.83,0.67,0.49,0.32],
                        'limite':[50,100,200,300,400]
                        },
                    '4':{
                        'datos':[0.98,0.89,0.78,0.67,0.45,0.31],
                        'limite':[50,100,200,300,400]
                        },
                    '5':{
                        'datos':[0.98,0.91,0.84,0.63,0.47,0.27],
                        'limite':[50,100,200,300,400]
                        },
                },
                'GOOD':{
                    '0':{
                        'datos':[0.98,0.88,0.83,0.64,0.37,0.25],
                        'limite':[50,100,200,300,400]
                        },
                    '1':{
                        'datos':[0.98,0.93,0.83,0.68,0.45,0.33],
                        'limite':[50,100,200,300,400]
                        },
                    '2':{
                        'datos':[0.98,0.92,0.87,0.67,0.51,0.42],
                        'limite':[50,100,200,300,400]
                        },
                    '3':{
                        'datos':[0.98,0.9,0.83,0.58,0.3,0.15],
                        'limite':[50,100,200,300,400]
                        },
                    '4':{
                        'datos':[0.98,0.9,0.87,0.67,0.32,0.25],
                        'limite':[50,100,200,300,400]
                        },
                    '5':{
                        'datos':[0.98,0.9,0.87,0.60,0.32,0.24],
                        'limite':[50,100,200,300,400]
                        },
                }
            }

            dict_model = dictionario[modelo][str(variable)]

            if (cantidad_perdidos <= dict_model['limite'][0]):
                return dict_model['datos'][0]
            elif (cantidad_perdidos <= dict_model['limite'][1]):
                return dict_model['datos'][1]
            elif (cantidad_perdidos <= dict_model['limite'][2]):
                return dict_model['datos'][2]
            elif (cantidad_perdidos <= dict_model['limite'][3]):
                return dict_model['datos'][3]
            elif (cantidad_perdidos <= dict_model['limite'][4]):
                return dict_model['datos'][4]
            else:
                return dict_model['datos'][5]
      

      def findLastRegister(self,registros_asociados):
            """
            function that find the last register of a specific circuit in the dataframe
            to find the sign of P and Q
            ----------------------------------------------------------------
            params
            ----------------------------------
            registros_asociados => (pandas dataframe)
            
            return
            ---------------------------------

            list => (list) with 2 values that represent the sign of P and Q

            """
            ###NO SE TENDRIA CERTEZA DEL FLUJO CON P=0 o Q=0
            registros_filtrados = registros_asociados[(registros_asociados['P'].notna()) & (registros_asociados['P'] != 0) & (registros_asociados['Q'].notna()) & (registros_asociados['Q'] != 0)]
            registros_ordenados = registros_filtrados.sort_values(by=['TIEMPO_AJUSTADO'], ascending=False)

            if (registros_ordenados.shape[0]==0):
               return None
            else:
                return [self.getSign(registros_ordenados.iloc[0].loc['P']),self.getSign(registros_ordenados.iloc[0].loc['Q'])]



      def getSign(self,Number):
          ## función para obtener el signo de un número
          if ( float(Number) < 0 ):
              return -1
          else:
              return 1


      def getFiability(self):
          """
          Model to get fiability
          -----------------------------------
          using the r coefficient related with the quantity
          of loss values in the time serie.
          """

          
          pass


      def findModel(self,name_model):
          """
          function to find the specific model that is useful for the circuit
          that we are evaluating
          """
          Lista_cabezera = ['10','23','30','46','40']
          Lista_red = ['2','3']

          dictionary = {

              '10' : self.MODEL_10,
              '2' : self.MODEL_2,
              '3' : self.MODEL_3,
              '23' : self.MODEL_GOOD,
              '30' : self.MODEL_GOOD,
              '46' : self.MODEL_GOOD,
              '40' : self.MODEL_GOOD

          }

          dictionary_2 = {

              '10' : '10',
              '2' : '2',
              '3' : '3',
              '23' : 'GOOD',
              '30' : 'GOOD',
              '46' : 'GOOD',
              '40' : 'GOOD'

          }

          if (name_model[3:5] in Lista_cabezera):
            return dictionary[name_model[3:5]],dictionary_2[name_model[3:5]]

          elif (name_model[0] in Lista_red):
            return dictionary[name_model[0]],dictionary_2[name_model[0]]
          else:
            return self.MODEL_ALL,'GOOD'



if __name__ == "__main__":

    VAE_MODELS  = VaeModel()
    #VAE_MODELS.model_iteration(dataframe)