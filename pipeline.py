try:
    rutas = './rutas.txt'

    with open(rutas, 'r') as archivo:
        for linea in archivo.readlines():
            exec(linea.strip(), globals())

    # !pip install -r requirements

    import numpy as np
    import pandas as pd
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')

    df_atributos = pd.read_csv(atributos)
    df_transacciones = pd.read_csv(transacciones)
    df = pd.merge(df_atributos, df_transacciones, left_on='POC', right_on='ACCOUNT_ID', how='outer')
    df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)
    for columna in df.columns:
        df[columna].fillna('S/D', inplace=True)
    df.drop_duplicates(inplace=True)
    df['POC'][df['POC']=='S/D'] = df['ACCOUNT_ID']
    df['ACCOUNT_ID'][df['ACCOUNT_ID']=='S/D'] = df['POC']
    df.drop(columns=['POC'], inplace=True)
    for registro in df['ACCOUNT_ID'][df['SkuDistintosToTales']=='S/D'].unique():
        calculado = df['SKU_ID'][df['ACCOUNT_ID']==registro].nunique()
        df['SkuDistintosToTales'][df['ACCOUNT_ID']==registro] = calculado
    for registro in df['ACCOUNT_ID'][df['SkuDistintosPromediosXOrden']=='S/D'].unique():
        productos = 0
        ordenes = 0
        lista = df['ORDER_ID'][df['ACCOUNT_ID']==registro].unique()
        for orden in lista:
            productos += df['SKU_ID'][df['ORDER_ID']==orden].nunique()
            ordenes += 1
        df['SkuDistintosPromediosXOrden'][df['ACCOUNT_ID']==registro] = round(productos/ordenes, 2)
    nombres = {'BussinessSegment':'perfil_digital', 'totalVolumen':'volumen_total', 'SkuDistintosPromediosXOrden':'productos_distintos_promedio',
            'SkuDistintosToTales':'productos_distintos_total', 'segmentoUnico':'segmento_unico', 'ACCOUNT_ID':'id_usuario', 'SKU_ID':'id_producto',
            'INVOICE_DATE':'fecha', 'ORDER_ID':'id_orden', 'ITEMS_PHYS_CASES':'bultos_productos'}
    df.rename(columns=nombres, inplace=True)
    df = df[['id_usuario', 'id_producto', 'id_orden', 'fecha', 'perfil_digital', 'volumen_total', 'productos_distintos_promedio', 'productos_distintos_total', 'segmento_unico', 'concentracion', 'nse', 'canal', 'bultos_productos']]
    df.drop(df[df['id_producto']=='S/D'].index, inplace=True)
    df['id_usuario'] = df['id_usuario'].astype(int)
    df['id_producto'] = df['id_producto'].astype(int)
    df['fecha'] = df['fecha'].astype(str)
    # df['volumen_total'] = df['volumen_total'].astype(float)
    df['productos_distintos_promedio'] = df['productos_distintos_promedio'].astype(float)
    df['productos_distintos_total'] = df['productos_distintos_total'].astype(int)
    df['bultos_productos'] = df['bultos_productos'].astype(int)
    df['anio'] = df['fecha'].str[:4].astype(int)
    df['mes'] = df['fecha'].str[4:6].astype(int)
    df['dia'] = df['fecha'].str[6:8].astype(int)
    df.drop(columns=['fecha'], inplace=True)
    df['fecha'] = df.apply(lambda row: datetime(2022, row['mes'], row['dia']), axis=1)
    # 0 = Lunes, 6 = Domingo
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df.drop(columns=['fecha'], inplace=True)
    pedidos_por_dia = df.groupby('dia_semana')['id_orden'].nunique()
    df['volumen_total'].replace('S/D', -1, inplace=True)
    df['volumen_total'] = df['volumen_total'].astype(float)
    df = df[['id_orden', 'anio', 'mes', 'dia', 'dia_semana', 'id_usuario',  'volumen_total', 'productos_distintos_promedio',
        'productos_distintos_total', 'bultos_productos', 'perfil_digital', 'segmento_unico', 'concentracion', 'nse',
        'canal', 'id_producto']]
    perfil_digital_t = {'S/D': 0, 'MinimalUsage': 1, 'MediumUsage': 2, 'HighUsage': 3, 'PowerUsage': 4}
    concentracion_t = {'S/D': 0, 'Bajo': 1, 'Medio': 2, 'Alto': 3}
    nse_t = {'S/D': 0, 'Bajo': 1, 'Medio': 2, 'Alto': 3}
    segmento_unico_t= {'S/D': 0, '1.Inactivos': 1, '2.Masivos': 2, '3.Potenciales': 3, '4.Activos': 4, '5.Select': 5}
    df['perfil_digital_t'] = df['perfil_digital'].map(perfil_digital_t)
    df['concentracion_t'] = df['concentracion'].map(concentracion_t)
    df['nse_t'] = df['nse'].map(nse_t)
    df['segmento_unico_t'] = df['segmento_unico'].map(segmento_unico_t)
    df.drop(columns=['perfil_digital', 'concentracion', 'nse', 'segmento_unico'], inplace=True)
    df.rename(columns={'perfil_digital_t': 'perfil_digital', 'concentracion_t': 'concentracion', 'nse_t': 'nse', 'segmento_unico_t': 'segmento_unico'}, inplace=True)
    canal_encoded = pd.get_dummies(df['canal'], prefix='canal')
    df.drop(columns=['canal'], inplace=True)
    df = pd.concat([df, canal_encoded], axis=1)
    df['canal_S/D'][df['canal_OTROS REF']==1] = True
    df.drop(columns=['canal_OTROS REF'], inplace=True)
    df = df[['id_orden', 'id_usuario', 'productos_distintos_promedio', 'perfil_digital','mes', 'dia', 'dia_semana',  'nse', 'segmento_unico', 'bultos_productos', 'canal_Autoservicio',
        'canal_BEBIDA', 'canal_Bar/Restaurant', 'canal_COMIDA',
        'canal_ENTRETENIMIENTO', 'canal_Instituciones', 'canal_KA Minoristas',
        'canal_Kioscos/Maxikioscos', 'canal_Mayorista', 'canal_S/D',
        'canal_Tradicional', 'id_producto']]
    df1 = df.copy()
    pivot_table_entrenamiento = df.pivot_table(
        index=['id_orden', 'id_usuario', 'perfil_digital', 'mes', 'dia', 'dia_semana', 'nse', 'segmento_unico', 'canal_Autoservicio',
        'canal_BEBIDA', 'canal_Bar/Restaurant', 'canal_COMIDA',
        'canal_ENTRETENIMIENTO', 'canal_Instituciones', 'canal_KA Minoristas',
        'canal_Kioscos/Maxikioscos', 'canal_Mayorista', 'canal_S/D',
        'canal_Tradicional'],
        columns='id_producto',
        values='bultos_productos',
        aggfunc='sum',
        fill_value=0)
    pivot_table_entrenamiento.columns = [f'producto_{int(col)}' for col in pivot_table_entrenamiento.columns]
    df = pivot_table_entrenamiento.reset_index()
    for columna in df.columns:
        if 'producto' in columna:
            df[columna][df[columna]>0] = 1
    productos_distintos_dict = df1.drop_duplicates(subset=['id_usuario']).set_index('id_usuario')['productos_distintos_promedio'].to_dict()
    df['promedio'] = df['id_usuario'].map(productos_distintos_dict)
    df['recomendaciones'] = df['promedio'].apply(lambda x: int(np.ceil(x))).astype(int)
    df.drop(columns=['promedio', 'id_orden'], inplace=True)
    columnas = list(df.columns)
    columnas.insert(1, columnas.pop(columnas.index('recomendaciones')))
    df = df[columnas]
    df['recomendaciones'][df['recomendaciones'] > 6] = 6
    df.to_csv(dataset_entrenamiento_final, index=False)
except Exception as e:
    print(e)