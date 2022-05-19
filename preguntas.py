"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el código presentado a continuación.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv(
        "gm_2008_region.csv",
        sep=",",
        header="infer",
    )

    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    y = df['life'].copy()
    X = df['fertility'].copy()

    # Imprima las dimensiones de `y`
    print(y.shape)

    # Imprima las dimensiones de `X`
    print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
    y_reshaped =  y.to_numpy().reshape(len(y),-1)

    # Trasforme `X` a un array de numpy usando reshape
    X_reshaped = X.to_numpy().reshape(len(X),-1)

    # Imprima las nuevas dimensiones de `y`
    print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
    print(X_reshaped.shape)


def pregunta_02():
    """
    En este punto se realiza la impresión de algunas estadísticas básicas
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv(
        "gm_2008_region.csv",
        sep=",",
        header="infer",
    )

    # Imprima las dimensiones del DataFrame
    print(df.shape)
    # print(df)

    # Imprima la correlación entre las columnas `life` y `fertility` con 4 decimales.
    print( '{:.4f}'.format(df['life'].corr( df['fertility'])) )

    # Imprima la media de la columna `life` con 4 decimales.
    print( '{:.4f}'.format(df['life'].mean()) )

    # Imprima el tipo de dato de la columna `fertility`.
    print( type(df["fertility"]) )

    # Imprima la correlación entre las columnas `GDP` y `life` con 4 decimales.
    print( '{:.3f}'.format(df['GDP'].corr( df['life'])) )


def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv(
        "gm_2008_region.csv",
        sep=",",
        header="infer",
    )

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = pd.Series( df['fertility'].values )
    """
    print("Feature: X_fertility " )
    print( X_fertility )
    print(" Dimension --> ", X_fertility.ndim )
    print(" Shape     --> ", X_fertility.shape )
    print(" Size      --> ", X_fertility.size )
    print(" Size 1stD --> ", len(X_fertility) )
    """
    X_fertility = X_fertility.to_numpy().reshape(len(X_fertility),1)
    # print("\nFeature: X_fertility RESHAPE... " )
    # print( X_fertility[:10,0] )
    
    # Asigne a la variable los valores de la columna `life`
    y_life = pd.Series( df['life'].values )
    """
    print("Target: y_life ")
    print( y_life )
    print(" Dimensions --> ", y_life.ndim )
    print(" Shape      --> ", y_life.shape )
    print(" Size       --> ", y_life.size )
    print(" Size 1stD  --> ", len(y_life) )
    """
    y_life = y_life.to_numpy().reshape(len(y_life),1)
    # print("\nTarget: y_life RESHAPE... ")
    # print( y_life[:10,0] )
    
    # Importe LinearRegression
    from sklearn.linear_model import LinearRegression

    # Cree una instancia del modelo de regresión lineal
    reg = LinearRegression()

    # Cree El espacio de predicción. Esto es, use linspace para crear
    # un vector con valores entre el máximo y el mínimo de X_fertility
    prediction_space = np.linspace(
        X_fertility.min(),
        X_fertility.max(),
        len(X_fertility)
    ).reshape(len(X_fertility),1)


    # Entrene el modelo usando X_fertility y y_life
    reg = reg.fit(X_fertility, y_life, sample_weight=None)
    # print ("linear model intercept (b) : {:.3f}".format( reg.intercept_) )

    # Compute las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)

    # Imprima el R^2 del modelo con 4 decimales
    print("{:6.4f}".format(reg.score( X_fertility, y_life, sample_weight=None )) )


def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el código presentado a continuación.
    """

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv(
        "gm_2008_region.csv",
        sep=",",
        header="infer",
    )
    # print ("Print Column from Dataframe... ")
    # print (df['fertility'])

    # Asigne a la variable los valores de la columna `fertility`
    # X_fertility = df['fertility']
    # X_fertility = pd.Series( df['fertility'].values )
    X_fertility = df.iloc[:,1]
    """
    print("\nFeature: X_fertility " )
    print( X_fertility.head() )
    print(" Dimensions --> ", X_fertility.ndim )
    print(" Shape      --> ", X_fertility.shape )
    print(" Size       --> ", X_fertility.size )
    print(" Length 1stD--> ", len(X_fertility) )
    """
    X_fertility = X_fertility.to_numpy().reshape(len(X_fertility),1)
    """
    print("\nFeature: X_fertility RESHAPE... " )
    print( X_fertility[:10,0] )
    print(" Dimensions --> ", X_fertility.ndim )
    print(" Shape      --> ", X_fertility.shape )
    print(" Size       --> ", X_fertility.size )
    print(" Length 1stD--> ", len(X_fertility) )
    """

    # Asigne a la variable los valores de la columna `life`
    y_life = pd.Series( df['life'].values ) # df['life']
    """
    print("\nTarget: y_life ")
    print( y_life.head())
    print(" Dimensions --> ", y_life.ndim )
    print(" Shape      --> ", y_life.shape )
    print(" Size       --> ", y_life.size )
    print(" Length 1stD--> ", len(y_life) )
    """
    y_life = y_life.to_numpy().reshape(len(y_life),1)
    """
    print("\nTarget: y_life RESHAPE... ")
    print( y_life[:10,0] )
    print(" Dimensions --> ", y_life.ndim )
    print(" Shape      --> ", y_life.shape )
    print(" Size       --> ", y_life.size )
    print(" Length 1stD--> ", len(y_life) )
    """

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    X_train, X_test, y_train, y_test = train_test_split(
        X_fertility,
        y_life,
        test_size=0.2,
        random_state=53,
    )
    """
    print("\nSET TRAIN feature fertility > X_train" )
    print(" Type       --> ", type(X_train) )
    print(X_train[:10,0] )
    print(" Dimensions --> ", X_train.ndim )
    print(" Shape      --> ", X_train.shape )
    print(" Size       --> ", X_train.size )
    print(" Length 1stD--> ", len(X_train) )

    print("\nSET TRAIN var target life > y_train" )
    print(" Type       --> ", type(y_train) )
    print( y_train[:10,0] )
    print(" Dimensions --> ", y_train.ndim )
    print(" Shape      --> ", y_train.shape )
    print(" Size       --> ", y_train.size )
    print(" Length 1stD--> ", len(y_train) )
    """

    # Cree una instancia del modelo de regresión lineal
    linearRegression = LinearRegression()

    # Entrene el clasificador usando X_train y y_train
    linearRegression.fit( X_train , y_train, sample_weight=None )

    # Pronostique y_test usando X_test
    y_pred = linearRegression.predict(X_test)

    # Compute and print R^2 and RMSE
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt( mean_squared_error(y_test, y_pred) )
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
