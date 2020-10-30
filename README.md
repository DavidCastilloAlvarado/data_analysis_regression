## **Analisis de Datos**
# **Regresión de valores de la curva de Iteración**
En ingeniería civil se realizan simulaciones para obtener el comportamiento sísmico de
estructuras, aquel comportamiento se logra con un diagrama de iteración.
Dicho diagrama necesita elementos de entrada como:

    longitud x,longitud y,Numero de varillas x,Numero de varillas y,f'c,B,Fy
![](/imagenes/diagrama.PNG)


## **Plan de trabajo** 
1. Inspección superficial de valores de la base de datos
2. EDA - Análisis exploratorio de datos<br>
    2.1. Histogramas<br>
    2.2. Corrección de errores<br>
    2.3. Matrices de correlación
3. Ingeniería de datos<br>
    3.1. Aumento de características <br>
    3.2. Análisis de las nuevas matrices de correlación
4. Preprocesamiento
5. Modelamiento de los regresores<br>
    5.1. Cross validation / interpretación<br>
    5.2. Selección del mejor modelo
6. Conclusiones

# 1. Inspección de datos
Se observa que es una base de datos **pequeña** con ningun valor nulo que corregir, sin embargo podría necesitar corrección en los valores, tal como veremos más adelante.

![](/imagenes/inspección.PNG)
![](/imagenes/inspección2.PNG)

#### Conteo de valores únicos por caracteristica
* Se observan columnas con valores únicos [n_bar_x,n_bar_y, B, Fy, X_cp, Y], los cuales serán eliminados posteriormente debido a su insignificancia estadistica.
* Se observa una columna con solo dos valores, la cual se interpretará como binaria para optimizar el análisis.
![](/imagenes/inspección3.PNG)

# 2. EDA 
## 2.1. Histogramas
* Visualizar el comportamiento de las caracteristicas, distribución
* Corr matrix en relación con la variable a objetivo

    * Se observa que algunas características tienen distribución uniforme Lonf_x, Long_y
    * Las variables de salida dependientes poseen un aceptable distribución normalizada. Y_cp y Y_fb
    * La Variable de salida dependiente X_fb posee un valor outlier que dista mucho de la distribución se procederá a eliminar o corregir dependiendo de sus variables  independientes.
    * La variable dependiente X, muestra un comportamiento próxima a una distribución uniforme.
![](/imagenes/histograma1.png)

## 2.2. Corrección de errores
#### Eliminando las columnas invariantes

df_study_n = df_study.drop(columns = ['n_bar_x', 'n_bar_y'  ,'B','Fy', 'X_cp', 'Y'])<br>
df_study_n.head()

#### Corrigiendo valor outlier
df_study_n.loc[df_study_n['X_fb']>1400]<br>
df_study_n.loc[df_study_n['long_x']==0.25].loc[df_study_n['long_y']==0.25]<br>
df_study_n.loc[1, 'X_fb'] = df_study.loc[1].X_fb/100.0<br>
df_study_n.loc[1, 'X_fb']<br>
feature_columns = ['long_x', 'long_y', 'fc']<br>
target_columns  = ['Y_cp', 'X_fb', 'Y_fb', 'X']

#### Mostrando histogramas con las correcciones
![](/imagenes/histograma2.png)

#### También graficaremos la representación de la característica "fc" dentro de la base de datos.
![](/imagenes/histograma3.png)
![](/imagenes/histograma4.png)

## 2.3. Matrices de correlación
Analizaremos la relación entre las características y cada salida objetivo.
Podemos observar que existe cierta tendencia de las características con las variables objetivos.<br>
La relación lineal más fuerte existe al analizar la variable objetivo X, la cual tiene una relación lineal con correlación Pearson de 1 con long_y.<br>
La segunda correlación fuerte existe entre X_fb y long_y, con una correlación Pearson de 0.89<br>
Si tomamos en cuenta que las variables de entrada corresponden a características físicas de dimensiones geométricas, no nos debería sorprender tanto estos resultados, más adelante veremos como podemos sacar ventaja de estos comportamientos.


for output in target_columns:<br>
. . . columns_cm = feature_columns.copy()<br>
. . . columns_cm.append(output)<br>
. . . sns_g = sns.pairplot(df_study_n[columns_cm], kind="reg", hue="fc", . . . markers=["o", "s"], palette="Set1", corner=True)<br>
. . . sns_g.fig.suptitle(output)<br>
. . . plt.show()<br>
. . . fig, ax = plt.subplots(figsize=(8,8))  <br>
. . . corr_df = df_study_n[columns_cm].corr(method='pearson') <br>
. . . matrix = np.triu(corr_df)<br>
. . . hmap=sns.heatmap(corr_df,annot=True, ax=ax, mask=matrix, vmin=-.8,vmax=.8,center=0)<br>
![](/imagenes/corr1_ycp.png)
![](/imagenes/corr1_xfb.png)
![](/imagenes/corr1_yfb.png)
![](/imagenes/corr1_x.png)

# 3. Ingeniería de Caracteristicas
* Se aumentarán las caracteristicas, generanfo una interaccion entre x y Y, 
    * Se probo con una relacion cuadrática y cúbica
    * Los mejores resultados se obtubieron al combinar estas dos
            XY y XY2 y X2Y

def add_new_features(data):<br>
... data = data.copy()<br>
... def improve_feat(x):<br>
... ...     xy   = x['long_x']*x['long_y']<br>
... ...     xy2  = x['long_x']*x['long_y']**2<br>
... ...     x2y  = (x['long_x']**2)*x['long_y']<br>
... ...     return pd.Series([ xy, xy2,x2y   ],index = [ 'xy','xy2','x2y' ])<br>
... return data.join(data.apply(improve_feat, axis=1))<br>

![](/imagenes/aumentación.PNG)

## 3.1. Correlación con las nuevas caracteristicas
Con las nuevas características las correlaciones se han disparado, ahora tenemos casi perfectas correlaciones lineales entre las características y las salidas objetivos.<br>
Por lo tanto podemos especular y decir de ante mano que el mejor modelo regresor para esta situación será un modelo lineal.<br>
Todas las relaciones de correlación tienen a 'fc' como característica binaria.<br>
Se observan fuertes correlaciones entre:
* Y_cp 
    * xy
    * xy2
* X_fb
    * long_y
    * xy2
    * xy
* Y_fb
    * xy 
    * xy2
* X
    * long_y
    * xy2

![](/imagenes/corr2_ycp.png)
![](/imagenes/corr2_xfb.png)
![](/imagenes/corr2_yfb.png)
![](/imagenes/corr2_x.png)

# 4. Preprocesamiento de datos
* Generando marcadores de estratificación para el split de los datos, con la finalidad
de obtener buena representación de los datos.

![](/imagenes/stt.PNG)

* Binarizamos 'fc'

![](/imagenes/stt2.PNG)

* Establecemos que el modelo final corresponde a 4 modelos regresores<br>

        feature_model = ['long_x', 'long_y', 'fc', 'xy', 'xy2', 'x2y']
        targets = [ ['Y_cp','Y_cp_stt'],
                    ['X_fb', 'X_fb_stt'],
                    ['Y_fb', 'Y_fb_stt'],
                    ['X','X_stt'] ]

# 5. Modelamiento
Se seleccionarán 10 candidatos regresores para la tarea, entre ellos modelos lineales y neigborhood aproach, arboles de decisión, random forrest y modelos ensemble.

    from sklearn.model_selection import train_test_split, GridSearchCV , RandomizedSearchCV,StratifiedKFold
    from sklearn.tree import DecisionTreeRegressor   as DTR
    from sklearn.linear_model import LinearRegression as lr
    from sklearn.linear_model import RidgeCV as rcv
    from sklearn.neighbors import KNeighborsRegressor as KNR
    from xgboost.sklearn import XGBRegressor
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import AdaBoostRegressor       as ABR
    from sklearn.ensemble import ExtraTreesRegressor     as ETR
    from sklearn.ensemble import RandomForestRegressor   as RFR
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

Se realizaron las pruebas usando Grid search, los resultados arrojaron que los modelos lineales obtuvieron los mejores resultados muy por delante de los modelos de mayor complejidad. 
![](/imagenes/lr1.PNG)
![](/imagenes/lr2.PNG)
![](/imagenes/lr3.PNG)
![](/imagenes/lr4.PNG)

# Conclusión
De todos los modelos probados, los modelos lineales, en especial la regresión lineal es la que mejor resultados arroja
asegurando una predicción sin overfiting.<br>
   * hyperparametros = {'copy_X': True, 'normalize': True}<br>
   
Los Modelos lineales son los recomendados para estimar estos valores ya que como se observo en el anális previo en las matrices de correlación, ciertas caracteristicas ya tienen una buena correlación de por sí con cada valor objetivo.

    hyperparametros = {'copy_X': True, 'normalize': True}
    lr_model = lr(**hyperparametros)
    X = df_study_model[feature_model]
    Y = df_study_model[['Y_cp', 'X_fb', 'Y_fb', 'X']]
    train_score = lr_model.fit(X,Y).score(X,Y) #Mean accuracy
    print("train_score Mean accuracy : ",train_score )

Graficamos los resultados.

    graaficas = [['Y_cp','Y_cp_pr'],
                ['X_fb','X_fb_pr'],
                ['Y_fb','Y_fb_pr'],
                ['X','X_pr']
                ]

    for graafica in graaficas:
        df_final[graafica].plot(figsize=(15,5), title =graafica[0])

![](/imagenes/out1.png)

![](/imagenes/out2.png)

![](/imagenes/out3.png)

![](/imagenes/out4.png)