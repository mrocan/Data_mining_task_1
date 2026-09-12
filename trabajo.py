
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import itertools
from sklearn.model_selection import train_test_split

# Establecemos nuestro escritorio de trabajo
os.chdir(r'C:\Users\maria\OneDrive\Documentos 1\Escritorio\MASTER\MODULO 6 MINERIA DE DATS\profe 1\Tarea')

# Cargo las funciones que voy a utilizar
from FuncionesMineria import (analizar_variables_categoricas, cuentaDistintos, frec_variables_num, 
                           atipicosAmissing, patron_perdidos, ImputacionCuant, ImputacionCuali, graficoVcramer,
                           mosaico_targetbinaria, boxplot_targetbinaria, hist_targetbinaria, Transf_Auto, 
                           lm, Rsq, validacion_cruzada_lm, modelEffectSizes, crear_data_modelo, 
                           Vcramer, hist_target_categorica, lm_stepwise, lm_backward, lm_forward,
                           glm_stepwise, glm_backward, glm_forward)

datos = pd.read_excel('DatosEleccionesEspaña.xlsx')

datos.dtypes

variables = list(datos.columns)   

# Indico las categóricas que aparecen como numéricas y las transformo en categoricas
numericasAcategoricas = ['CodigoProvincia', 'AbstencionAlta']
for var in numericasAcategoricas:
    datos[var] = datos[var].astype(str)
    
datos = datos.drop(columns=['Name', 'Izquierda', 'Derecha', 'Otros_Pct', 'AbstentionPtge', 'Dcha_Pct'])
nuevas_variables = list(datos.columns)

# Seleccionar las columnas numéricas del DataFrame


# Comprobamos que todas las variables tienen el formato que queremos  
datos.dtypes

# Frecuencias de los valores en las variables categóricas
analizar_variables_categoricas(datos)

# Cuenta el número de valores distintos de cada una de las variables numéricas de un DataFrame
#cuentaDistintos(datos)

# Descriptivos variables numéricas mediante función describe() de Python
descriptivos_num = datos.describe().T

# Añadimos más descriptivos a los anteriores
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)


# Muestra valores perdidos
datos[nuevas_variables].isna().sum()


# Comenzamos la depuración de los datos
# A veces los 'nan' vienen como como una cadena de caracteres, los modificamos a perdidos.
for x in categoricas:
    datos[x] = datos[x].replace('nan', np.nan) 

#volvemos a imprimir los nan a ver si sale alguno y vemos que ya no hay valores nan
datos[nuevas_variables].isna().sum

# Missings no declarados variables cualitativas (NSNC, ?)
datos['Densidad'] = datos['Densidad'].replace('?', np.nan)

# Missings no declarados variables cuantitativas (-1, 99999)
datos['Explotaciones'] = datos['Explotaciones'].replace(99999, np.nan)

# Valores fuera de 
porcentajes_0_100 = ['Age_19_65_pct', 'Age_over65_pct','ForeignersPtge', 'SameComAutonPtge']
for var in porcentajes_0_100:
    datos[var] = [x if 0 <= x <= 100 else np.nan for x in datos[var]]


# Junto categorías poco representadas de las variables categóricas
datos['ActividadPpal'] = datos['ActividadPpal'].replace({'Otro': 'Otro', \
                    'ComercTTEHosteleria': 'ComercTTEHosteleria', \
                    'Servicios': 'Servicios_Constr_Indus', 'Construccion': 'Servicios_Constr_Indus',
                    'Industria':'Servicios_Constr_Indus'})

# Indico la variableObj, el ID y las Input (los atipicos y los missings se gestionan
# solo de las variables input)
varObjCont = datos['Izda_Pct']
varObjBin = datos['AbstencionAlta']
datos_input = datos.drop(['Izda_Pct', 'AbstencionAlta'], axis = 1)

# Genera una lista con los nombres de las variables del cojunto de datos input.
variables_input = list(datos_input.columns)  

# Selecionamos las variables numéricas
numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Selecionamos las variables categóricas
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]

proporcion_atipicos = {x: atipicosAmissing(datos_input[x])[1] / len(datos_input) for x in numericas_input}
#vamos a analizar que variables tienen entorno a un 10%o mas de atipicos para categorizarlas y asi 
#no perder informacion pasando esos valores atipicos a valores missings 
#estas variables son Population, totalempresas, pob2010, totalcensus e inmuebles. 
#tb estan servicios, contruccion, industria, comercttehosteleria, pero estos son dependientes de totalempresas
labels = ['Muy baja', 'Baja', 'Media', 'Alta', 'Muy alta'] 
nuevas_categoricas = ['Population', 'totalEmpresas', 'Pob2010', 'TotalCensus', 'inmuebles']
for var in nuevas_categoricas:
    datos_input[var] = ImputacionCuant(datos_input[var], 'mediana')
    
    quintile_ranges = np.percentile(datos_input[var], [0, 20, 40, 60, 80, 100])
    
    datos_input[var] = pd.qcut(
        datos_input[var], 
        q=5,
        labels= labels 
    )
    datos_input[var] = datos_input[var].astype(str)
    # Mostrar los rangos de cada quintil
    print(f"Rangos de los quintiles de la variable {var}:")
    for i in range(len(quintile_ranges) - 1):
        print(f"{labels[i]}: {quintile_ranges[i]} - {quintile_ranges[i + 1]}")
    print("\n")
        
categoricas_input.extend(nuevas_categoricas)
numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Modifico los atipicos como missings
for x in numericas_input:
    datos_input[x] = atipicosAmissing(datos_input[x])[0]

# MISSINGS
# Visualiza un mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.
#patron_perdidos(datos_input)

# Muestra total de valores perdidos por cada variable
datos_input[variables_input].isna().sum()

# Muestra proporción de valores perdidos por cada variable (guardo la información)
prop_missingsVars = datos_input.isna().sum()/len(datos_input)

# Creamos la variable prop_missings que recoge el número de valores perdidos por cada observación
datos_input['prop_missings'] = datos_input.isna().mean(axis = 1)

# Realizamos un estudio descriptivo básico a la nueva variable
datos_input['prop_missings'].describe() 

# Calculamos el número de valores distintos que tiene la nueva variable
len(datos_input['prop_missings'].unique())

# Elimino las observaciones con mas de la mitad de datos missings (no hay ninguna)
eliminar = datos_input['prop_missings'] > 0.5
datos_input = datos_input[~eliminar]
varObjBin = varObjBin[~eliminar]
varObjCont = varObjCont[~eliminar]

# Transformo la nueva variable en categórica (ya que tiene pocos valores diferentes)
datos_input["prop_missings"] = datos_input["prop_missings"].astype(str)

# Agrego 'prop_missings' a la lista de nombres de variables input
variables_input.append('prop_missings')
categoricas_input.append('prop_missings')


# Elimino las variables con mas de la mitad de datos missings (no hay ninguna)
eliminar = [prop_missingsVars.index[x] for x in range(len(prop_missingsVars)) if prop_missingsVars[x] > 0.5]
datos_input = datos_input.drop(eliminar, axis = 1)


## IMPUTACIONES
# Imputo todas las cuantitativas, seleccionar el tipo de imputacion: media, mediana o aleatorio
for x in numericas_input:
    datos_input[x] = ImputacionCuant(datos_input[x], 'mediana')

# Imputo todas las cualitativas, seleccionar el tipo de imputacion: moda o aleatorio
for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'aleatorio')

# Reviso que no queden datos missings
datos_input.isna().sum()
 
analizar_variables_categoricas(datos_input)

# Junto categorías poco representadas de las variables categóricas
datos_input['prop_missings'] = datos_input['prop_missings'].replace({'0.0': '0-3%', 
                    '0.030303030303030304': '3-6%', 
                    '0.06060606060606061': '6-9%', 
                    '0.09090909090909091': '> 9%',
                    '0.12121212121212122':'> 9%',
                    '0.24242424242424243': '> 9%',
                    '0.15151515151515152': '> 9%',
                    '0.21212121212121213': '> 9%',
                    '0.18181818181818182': '> 9%'})

# Una vez finalizado este proceso, se puede considerar que los datos estan depurados. Los guardamos
datosVinoDep = pd.concat([varObjBin, varObjCont, datos_input], axis = 1)
with open('datosEleccionesDep.pickle', 'wb') as archivo:
    pickle.dump(datosVinoDep, archivo)
 
    
################################
################################
################################
################################
########REGRESION LINEAL########
################################
################################
################################
################################

#with open('datosEleccionesDep.pickle', 'rb') as f:
    #datos_input = pickle.load(f)
 
variables = list(datos_input.columns)  

# Obtengo la importancia de las variables
graficoVcramer(datos_input, varObjBin)
graficoVcramer(datos_input, varObjCont)

# Crear un DataFrame para almacenar los resultados del coeficiente V de Cramer
VCramer = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])


# Correlación entre todas las variables numéricas frente a la objetivo continua.
# Obtener las columnas numéricas del DataFrame 'datos_input'
numericas = datos_input.select_dtypes(include=['int', 'float']).columns
# Calcular la matriz de correlación de Pearson entre la variable objetivo continua ('varObjCont') y las variables numéricas
matriz_corr = pd.concat([varObjCont, datos_input[numericas]], axis = 1).corr(method = 'pearson')
# Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
# Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
plt.figure(figsize=(20, 16))
# Establecer el tamaño de fuente en el gráfico
sns.set(font_scale=1.2)
# Crear un mapa de calor (heatmap) de la matriz de correlación
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, mask=mask)
# Establecer el título del gráfico
plt.title("Matriz de correlación de valores ausentes")
# Mostrar el gráfico de la matriz de correlación
plt.show()

#################################################################################
## Comenzamos con la regresion lineal
#################################################################################


# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(datos_input, np.ravel(varObjCont), test_size = 0.2, random_state = 123456)

# Construyo un modelo preliminar con todas las variables (originales)
# Indico la tipología de las variables (numéricas o categóricas)
var_num1 = numericas_input.tolist()
var_categ1 = categoricas_input

# Creo el modelo
modelo1 = lm(y_train, x_train, var_num1, var_categ1)
# Visualizamos los resultado del modelo
modelo1['Modelo'].summary()

# Calculamos la medida de ajuste R^2 para los datos de entrenamiento
Rsq(modelo1['Modelo'], y_train, modelo1['X'])

# Preparamos los datos test para usar en el modelo
x_test_modelo1 = crear_data_modelo(x_test, var_num1, var_categ1)
# Calculamos la medida de ajuste R^2 para los datos test
Rsq(modelo1['Modelo'], y_test, x_test_modelo1)
len(modelo1['Modelo'].params)


# Vamos a probar un modelo con menos variables. Recuerdo el grafico de Cramer
graficoVcramer(datos_input, varObjCont) # Pruebo con las mas importantes
var_cuant_importantes = ['UnemployLess25_Ptge', 'AgricultureUnemploymentPtge','SameComAutonDiffProvPtge',
             'IndustryUnemploymentPtge', 'Age_0-4_Ptge']
for var1 in var_cuant_importantes:
    for var2 in var_cuant_importantes:
        if var1 != var2:
            corr = matriz_corr.loc[var1][var2]
            #print(f"La correlacion entre {var1} y {var2} es: {corr}")
            if corr>0.35 or corr<-0.35:
                print(f"correlacion alta entre {var1} y {var2}")
                #no se printea nada, por tanto entre las variables var_cuant_importantes no hay correlaciones altas


interacciones = var_cuant_importantes
interacciones_unicas = list(itertools.combinations(interacciones, 2)) 
  

##modelos clasicos
modeloStepAIC = lm_stepwise(y_train, x_train, var_num1, var_categ1, [], 'AIC')
modeloStepBIC = lm_stepwise(y_train, x_train, var_num1, var_categ1, [], 'BIC')
modeloBackAIC = lm_backward(y_train, x_train, var_num1, var_categ1, [], 'AIC')
modeloBackBIC = lm_backward(y_train, x_train, var_num1, var_categ1, [], 'BIC')
modeloForwAIC = lm_forward(y_train, x_train, var_num1, var_categ1, [], 'AIC')
modeloForwBIC = lm_forward(y_train, x_train, var_num1, var_categ1, [], 'BIC')


modeloBackBIC_int = lm_backward(y_train, x_train, var_num1, var_categ1, interacciones_unicas, 'BIC'),
modeloForwBIC_int=  lm_forward(y_train, x_train, var_num1, var_categ1, interacciones_unicas, 'BIC')


# Nombres y métricas correspondientes
nombres_metricas = [ ("Stepwise", "AIC"), ("Stepwise", "BIC"), ("Backward", "AIC"), ("Backward", "BIC"),
                    ("Forward", "AIC"), ("Forward", "BIC"), ("Backward + Inter", "BIC"), ("Forward + Inter", "BIC")]
modelos_clasicos = [ modeloStepAIC, modeloStepBIC, modeloBackAIC, modeloBackBIC,
                    modeloForwAIC, modeloForwBIC, modeloBackBIC_int, modeloForwBIC_int]
# Crear dataframe de resultados
columnas_tabla = ['Metodo', 'Metrica', 'R^2 train', 'R^2 test', 'num parametros']
resultados = pd.DataFrame(columns=columnas_tabla)

for i, modelo in enumerate(modelos_clasicos):
    
    nombre, metr = nombres_metricas[i]
    if isinstance(modelo, tuple):
        modelo = modelo[0]
    # Resumen del modelo
    #modelo['Modelo'].summary()
    
    # R-squared para el conjunto de entrenamiento
    Rsq_train = Rsq(modelo['Modelo'], y_train, modelo['X'])
    
    # Preparo datos de test considerando si hay interacciones o no
    if 'inter' in modelo['Variables']:  # Con interacciones
        x_test_modelo = crear_data_modelo(
            x_test,
            modelo['Variables']['cont'],
            modelo['Variables']['categ'],
            modelo['Variables']['inter']
        )
    else:  # Sin interacciones
        x_test_modelo = crear_data_modelo(
            x_test,
            modelo['Variables']['cont'],
            modelo['Variables']['categ']
        )
    
    # R-squared para el conjunto de test
    Rsq_test = Rsq(modelo['Modelo'], y_test, x_test_modelo)
    
    # Número de parámetros del modelo
    params = len(modelo['Modelo'].params)
    
    # Añadir resultados al DataFrame
    resultados.loc[len(resultados)] = [nombre, metr, Rsq_train, Rsq_test, params]

    

# Hago validacion cruzada repetida para ver que modelo es mejor
results2 = pd.DataFrame({
    'Rsquared': [],
    'Resample': [],
    'Modelo': []
})
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo_stepAIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepAIC['Variables']['cont']
        , modeloStepAIC['Variables']['categ']
    )
    modelo_stepBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC['Variables']['cont']
        , modeloStepBIC['Variables']['categ']
    )

    modelo_backAIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloBackAIC['Variables']['cont']
        , modeloBackAIC['Variables']['categ']
    )
    modelo_backBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloBackBIC['Variables']['cont']
        , modeloBackBIC['Variables']['categ']
    )
    modelo_forAIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloForwAIC['Variables']['cont']
        , modeloForwAIC['Variables']['categ']
    )
    modelo_forBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloForwBIC['Variables']['cont']
        , modeloForwBIC['Variables']['categ']
    )
    modelo_backBIC_int = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloBackBIC_int[0]['Variables']['cont']
        , modeloBackBIC_int[0]['Variables']['categ']
    )
    modelo_forwBIC_int = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloForwBIC_int['Variables']['cont']
        , modeloForwBIC_int['Variables']['categ']
    )
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición

    results_rep = pd.DataFrame({
        'Rsquared': modelo_stepAIC + modelo_stepBIC + modelo_backAIC + modelo_backBIC + modelo_forAIC + modelo_forBIC
        + modelo_backBIC_int + modelo_forwBIC_int
        , 'Resample': ['Rep' + str((rep + 1))]*5*8 # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 + [7]*5 + [8]*5# Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results2 = pd.concat([results2, results_rep], axis = 0)
    
# Boxplot de la validación cruzada
plt.figure(figsize=(10, 6))  
plt.grid(True)  
# Agrupa los valores de R-squared por modelo
grupo_metrica = results2.groupby('Modelo')['Rsquared']

boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys()) 
plt.xlabel('Modelo') 
plt.ylabel('Rsquared')  
plt.show() 
    

## Seleccion aleatoria (se coge la submuestra de los datos de entrenamiento)
variables_seleccionadas = {'Formula': [], 'Variables': []}
for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3, random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_stepwise(y_train2.astype(int), x_train2, var_num1, var_categ1, [], 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))

# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las tres fórmulas más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]
var_3 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][2])]


## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results3 = pd.DataFrame({'Rsquared': [], 'Resample': [], 'Modelo': []})
for rep in range(20):
    modelo1 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloBackBIC['Variables']['cont']
        , modeloBackBIC['Variables']['categ']
    )
    modelo2 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_1['cont']
        , var_1['categ']
    )
    modelo3 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_2['cont']
        , var_2['categ']
    )
    modelo4 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_3['cont']
        , var_3['categ']
    )
    results_rep2 = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 + modelo4
        , 'Resample': ['Rep' + str((rep + 1))]*5*4
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5
    })
    results3 = pd.concat([results3, results_rep2], axis = 0)
     
# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6)) 
plt.grid(True) 
grupo_metrica = results3.groupby('Modelo')['Rsquared']
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  
plt.xlabel('Modelo') 
plt.ylabel('Rsquared') 
plt.show()  

# Calcular la media de las métricas R-squared por modelo
media_r2_v2 = results3.groupby('Modelo')['Rsquared'].mean()
# Contar el número de parámetros en cada modelo
num_params_v2 = [len(modeloBackBIC['Modelo'].params), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+')), 
                 len(frec_ordenada['Formula'][2].split('+'))]

# Una vez decidido el mejor modelo, hay que evaluarlo 
ModeloGanador = modeloBackBIC

# Vemos los coeficientes del modelo ganador
ModeloGanador['Modelo'].summary()

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test
Rsq(ModeloGanador['Modelo'], y_train, ModeloGanador['X'])

x_test_modeloganador = crear_data_modelo(x_test, ModeloGanador['Variables']['cont'], 
                                                ModeloGanador['Variables']['categ'], 
                                                ModeloGanador['Variables']['inter'])
Rsq(ModeloGanador['Modelo'], y_test, x_test_modeloganador)
    




################################
################################
################################
################################
######REGRESION LOGISTICA#######
################################
################################
################################
################################

from FuncionesMineria import (graficoVcramer, impVariablesLog, pseudoR2, glm, summary_glm, 
                           validacion_cruzada_glm, sensEspCorte, crear_data_modelo, curva_roc)


# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(datos_input, varObjBin, test_size = 0.2, random_state = 1234567)
y_train, y_test = y_train.astype(int), y_test.astype(int)

###modelos clasicos
modeloStepAIC_Log = glm_stepwise(y_train, x_train, var_num1, var_categ1, [], 'AIC')
modeloStepBIC_Log = glm_stepwise(y_train, x_train, var_num1, var_categ1, [], 'BIC')
modeloBackAIC_Log = glm_backward(y_train, x_train, var_num1, var_categ1, [], 'AIC')
modeloBackBIC_Log = glm_backward(y_train, x_train, var_num1, var_categ1, [], 'BIC')
modeloForwAIC_Log = glm_forward(y_train, x_train, var_num1, var_categ1, [], 'AIC')
modeloForwBIC_Log = glm_forward(y_train, x_train, var_num1, var_categ1, [], 'BIC')
modeloBackBIC_int_Log = glm_backward(y_train, x_train, var_num1, var_categ1, interacciones_unicas, 'BIC'),
modeloForwBIC_int_Log =  glm_forward(y_train, x_train, var_num1, var_categ1, interacciones_unicas, 'BIC')

columnas_tabla = ['Metodo', 'Metrica','R^2 train', 'R^2 test', 'num_param', 'AUC']
resultados_Log = pd.DataFrame(columns=columnas_tabla)

modelos_clasicos_Log = [modeloStepAIC_Log, modeloStepBIC_Log, modeloBackAIC_Log, modeloBackBIC_Log,
                    modeloForwAIC_Log, modeloForwBIC_Log, modeloBackBIC_int_Log, modeloForwBIC_int_Log]

for i, modelo in enumerate(modelos_clasicos_Log):
    nombre, metr = nombres_metricas[i]

    if isinstance(modelo,tuple):
        modelo = modelo[0]
    Rsq_train = pseudoR2(modelo['Modelo'], modelo['X'],y_train)
    
    # Preparo datos de test considerando si hay interacciones o no
    if 'inter' in modelo['Variables']:  # Con interacciones
        x_test_modelo = crear_data_modelo(
            x_test,
            modelo['Variables']['cont'],
            modelo['Variables']['categ'],
            modelo['Variables']['inter']
        )
    else:  # Sin interacciones
        x_test_modelo = crear_data_modelo(
            x_test,
            modelo['Variables']['cont'],
            modelo['Variables']['categ']
        )   

    Rsq_test = pseudoR2(modelo['Modelo'],x_test_modelo,  y_test)    
    params = len(modelo['Modelo'].coef_[0])
    AUC = curva_roc(x_test_modelo, y_test, modelo)
    # Añadir resultados al DataFrame
    resultados_Log.loc[len(resultados_Log)] = [nombre, metr, Rsq_train, Rsq_test, params, AUC]
    



# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results_log = pd.DataFrame({
    'AUC': []
    , 'Resample': []
    , 'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo_stepAIC_Log = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloStepAIC_Log['Variables']['cont']
        , modeloStepAIC_Log['Variables']['categ']
    )
    modelo_stepBIC_Log = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloStepBIC_Log['Variables']['cont']
        , modeloStepBIC_Log['Variables']['categ']
    )
    
    modelo_backAIC_Log = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloBackAIC_Log['Variables']['cont']
        , modeloBackAIC_Log['Variables']['categ']
    )
    modelo_backBIC_Log = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloBackBIC_Log['Variables']['cont']
        , modeloBackBIC_Log['Variables']['categ']
    )
    modelo_forAIC_Log = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloForwAIC_Log['Variables']['cont']
        , modeloForwAIC_Log['Variables']['categ']
    )
    modelo_forBIC_Log = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloForwBIC_Log['Variables']['cont']
        , modeloForwBIC_Log['Variables']['categ']
    )
    modelo_backBIC_int_Log = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloBackBIC_int_Log[0]['Variables']['cont']
        , modeloBackBIC_int_Log[0]['Variables']['categ']
    )
    modelo_forwBIC_int_Log = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloForwBIC_int_Log['Variables']['cont']
        , modeloForwBIC_int_Log['Variables']['categ']
    )
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep_log = pd.DataFrame({
        'AUC': modelo_stepAIC_Log + modelo_stepBIC_Log + modelo_backAIC_Log + modelo_backBIC_Log + modelo_forAIC_Log + modelo_forBIC_Log + modelo_backBIC_int_Log + modelo_forwBIC_int_Log
        , 'Resample': ['Rep' + str((rep + 1))]*5*8  # Etiqueta de repetición (5 repeticiones 5 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 + [7]*5 + [8]*5 # Etiqueta de modelo (5 modelos 5 repeticiones)
    })
    results_log = pd.concat([results_log, results_rep_log], axis = 0)

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de AUC por modelo
grupo_metrica = results_log.groupby('Modelo')['AUC']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('AUC')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

 
    
# Calcular la media del AUC por modelo
media_auc_v2_log = results_log.groupby('Modelo')['AUC'].mean()
# Calcular la desviación estándar del AUC por modelo
results_log.groupby('Modelo')['AUC'].std()    

## Seleccion aleatoria (se coge la submuestra de los datos de entrenamiento)

# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas_log = {
    'Formula': [],
    'Variables': []
}


# Realizar 20 iteraciones de selección aleatoria. (en clase no se puede correr con 20)
for x in range(20):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3, random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = glm_stepwise(y_train2.astype(int), x_train2, var_num1, var_categ1, [], 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas_log['Variables'].append(modelo['Variables'])
    variables_seleccionadas_log['Formula'].append(sorted(modelo['X'].columns))
# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas_log['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las tres fórmulas más frecuentes y las variables correspondientes.
var_1_log = variables_seleccionadas_log['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2_log = variables_seleccionadas_log['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]
var_3_log = variables_seleccionadas_log['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][2])]


## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results3_log_2 = pd.DataFrame({
    'AUC': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloBackBIC_Log['Variables']['cont']
        , modeloBackBIC_Log['Variables']['categ']
    )

    modelo3 = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , var_2_log['cont']
        , var_2_log['categ']
    )
    modelo4 = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , var_3_log['cont']
        , var_3_log['categ']
    )
    results_rep2_log = pd.DataFrame({
        'AUC': modelo1 + modelo3 + modelo4
        , 'Resample': ['Rep' + str((rep + 1))]*5*3
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 
    })
    results3_log_2 = pd.concat([results3_log_2, results_rep2_log], axis = 0)
     

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results3_log_2.groupby('Modelo')['AUC']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('AUC')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Calcular la media de las métricas R-squared por modelo
media_auc_v2_log2 = results3_log_2.groupby('Modelo')['AUC'].mean()
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2_v2_log = results3_log_2.groupby('Modelo')['AUC'].std()
# Contar el número de parámetros en cada modelo
num_params_v2_log = [len(modeloBackBIC_Log['Modelo'].coef_[0]), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+')), 
                 len(frec_ordenada['Formula'][2].split('+'))]

modeloGanador_log = modeloBackBIC_Log

# Probamos dos puntos de corte
sensEspCorte(modeloGanador_log['Modelo'], x_test, y_test, 0.4, modeloGanador_log['Variables']['cont'], modeloGanador_log['Variables']['categ']) # punto de corte = 0.4
sensEspCorte(modeloGanador_log['Modelo'], x_test, y_test, 0.6, modeloGanador_log['Variables']['cont'], modeloGanador_log['Variables']['categ'])

# Generamos una rejilla de puntos de corte
posiblesCortes = np.arange(0, 1.01, 0.01).tolist()  # Generamos puntos de corte de 0 a 1 con intervalo de 0.01
rejilla = pd.DataFrame({
    'PtoCorte': [],
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'PosPredValue': [],
    'NegPredValue': []
})  # Creamos un DataFrame para almacenar las métricas para cada punto de corte

for pto_corte in posiblesCortes:  # Iteramos sobre los puntos de corte
    rejilla = pd.concat(
        [rejilla, sensEspCorte(modeloGanador_log['Modelo'], x_test, y_test, pto_corte, modeloGanador_log['Variables']['cont'] , modeloGanador_log['Variables']['categ'])],
        axis=0
    )  # Calculamos las métricas para el punto de corte actual y lo agregamos al DataFrame

rejilla['Youden'] = rejilla['Sensitivity'] + rejilla['Specificity'] - 1  # Calculamos el índice de Youden
rejilla.index = list(range(len(rejilla)))  # Reindexamos el DataFrame para que los índices sean consecutivos


plt.plot(rejilla['PtoCorte'], rejilla['Youden'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Youden')
plt.title('Youden')
plt.show()

plt.plot(rejilla['PtoCorte'], rejilla['Accuracy'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()

rejilla['PtoCorte'][rejilla['Youden'].idxmax()]
rejilla['PtoCorte'][rejilla['Accuracy'].idxmax()]

sensEspCorte(modeloGanador_log['Modelo'], x_test, y_test, 0.28, modeloGanador_log['Variables']['cont'], modeloGanador_log['Variables']['categ'])
sensEspCorte(modeloGanador_log['Modelo'], x_test, y_test, 0.47, modeloGanador_log['Variables']['cont'], modeloGanador_log['Variables']['categ'])

# Vemos los coeficientes del modelo ganador
summary_glm(modeloGanador_log['Modelo'], y_train, modeloGanador_log['X'])

