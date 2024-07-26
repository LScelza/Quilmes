rutas = './rutas.txt'

with open(rutas, 'r') as archivo:
    for linea in archivo.readlines():
        exec(linea.strip(), globals())

# !pip install -r requirements

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau
from random import randint
from datetime import datetime
import time
import sys
import warnings
warnings.filterwarnings('ignore')

def preprocesamiento(data:pd.DataFrame, usuario:int) -> tuple | np.ndarray:
    data = data[data['id_usuario']==usuario]
    recomendaciones = int(data['recomendaciones'][data['id_usuario']==usuario].unique())
    productos = []
    for producto in df_final.filter(regex='^producto_').columns:
        productos.append({producto:df_final[producto].sum()})
    productos.sort(key=lambda x: x[list(x.keys())[0]], reverse=True)
    productos_prediccion = []
    for i in range(10):
        productos_prediccion.append(list(productos[i].keys())[0])
    productos_codificados = []
    for i in range(len(df_final.columns[-530:])):
        productos_codificados.append({df_final.columns[-530:][i]:i})
    productos_frecuencias = []
    for producto in productos_codificados:
        for productop in productos_prediccion:
            if productop in producto.keys():
                productos_frecuencias.append(producto[productop])
    features = ['perfil_digital', 'mes', 'dia', 'dia_semana', 'nse', 'segmento_unico']
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    features_total = features + list(data.filter(regex='^canal_').columns)
    target = data.filter(regex='^producto_').columns
    X = data[features_total]
    y = data[target]
    if len(data) == 1:
        return y.values
    else:
        X_train = X.iloc[:-1,:]
        y_train = y.iloc[:-1,:]
        X_test = X.iloc[-1:,:]
        y_test = y.iloc[-1:,:]
        y_train = y_train.values
        y_test = y_test.values
        X_test = X_test.astype('float32')
        X_train = X_train.astype('float32')
        y_test = y_test.astype('float32')
        y_train = y_train.astype('float')
        return X_train, y_train, X_test, y_test, recomendaciones, productos_frecuencias

def modelo(data:pd.DataFrame, umbral:float=0.4) -> pd.DataFrame:
    usuario = int(input('Ingrese el id del usuario: '))
    while usuario not in data['id_usuario'].unique():
        usuario = int(input('Error en id de usuario. Ingrese nuevamente el id del usuario: '))
    pre = preprocesamiento(data, usuario)
    if isinstance(pre, tuple):
        X_train, y_train, X_test, y_test = pre[:4]
        recomendaciones = pre[4]
        productos_frecuencia= pre[5]
        model = Sequential([Dense(512, input_dim=X_train.shape[1], activation='swish', kernel_regularizer=l2(0.01)),
                            Dense(512, activation='swish'),
                            BatchNormalization(),
                            Dense(256, activation='swish', kernel_regularizer=l2(0.01)),
                            Dense(256, activation='swish'),
                            Dense(128, activation='swish'),
                            Dense(128, activation='swish'),
                            BatchNormalization(),
                            Dense(64, activation='swish'),
                            Dense(64, activation='swish'),
                            Dense(32, activation='swish'),
                            Dense(32, activation='swish'),
                            Dense(y_train.shape[1], activation='sigmoid')])
        model.compile(optimizer=Nadam(learning_rate=0.03), loss='binary_crossentropy')
        model.fit(X_train, y_train, epochs=30, batch_size=3, validation_data=(X_test, y_test), verbose=0, callbacks=ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=1e-8))
        y_pred = model.predict(X_test, verbose=0)
        prediccion = np.where(y_pred >= umbral)[1]
        if len(prediccion) > recomendaciones:
            prediccion = np.argsort(y_pred[0])[-recomendaciones:][::-1]
        productos_recomendacion = []
        for producto in productos_frecuencia:
            if producto not in prediccion:
                productos_recomendacion.append(producto)
        for i in range(len(productos_recomendacion)):
            if len(prediccion) < recomendaciones:
                prediccion = np.append(prediccion, productos_recomendacion[i])
            else:
                break
        prediccion.sort()
        columnas_productos = data.filter(regex='^producto_').columns
        prediccion = [int(columnas_productos[i].split('_')[1]) for i in prediccion]
        y_test = np.where(y_test == 1)[1]
        y_test = [int(columnas_productos[i].split('_')[1]) for i in y_test]
        return usuario, prediccion, y_test
    else:
        prediccion = pre
        columnas_productos = data.filter(regex='^producto_').columns
        prediccion = [int(columnas_productos[i].split('_')[1]) for i in prediccion]
        y_test = np.where(y_test == 1)[1]
        y_test = [int(columnas_productos[i].split('_')[1]) for i in y_test]
        return usuario, prediccion, y_test



df_final = pd.read_csv(dataset_entrenamiento_final)
try:
    prediccion = modelo(df_final)
    print(f'Usuario: {prediccion[0]}\nProductos: {prediccion[1]}\nTest: {prediccion[2]}')
except Exception as e:
    print('No se pudo realizar la operación. Verifique que los datos estén en el formato correctos.')
    print(e)
