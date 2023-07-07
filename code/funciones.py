from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np

def split_dataset(serie, tr_size=0.8, vl_size=0.1, ts_size=0.1 ):
    # Definir número de datos en cada subserie
    N = serie.shape[0]
    Ntrain = int(tr_size*N)  # Número de datos de entrenamiento
    Nval = int(vl_size*N)    # Número de datos de validación
    Ntst = N - Ntrain - Nval # Número de datos de prueba

    # Realizar partición
    train = serie[0:Ntrain]
    val = serie[Ntrain:Ntrain+Nval]
    test = serie[Ntrain+Nval:]

    return train, val, test

def split_dataset_byN(dataset, n, tr_size=0.8, vl_size=0.1, ts_size=0.1 ):
    # dividir el dataset por nro de samples
    n_sample = dataset.shape[0] / n

    # Definir número de datos en cada subserie
    N = dataset.shape[0]
    Ntrain = int(n_sample * tr_size) * n # Número de datos de entrenamiento
    Nval = int(n_sample * vl_size) * n   # Número de datos de validación
    Ntst = N - Ntrain - Nval # Número de datos de prueba

    # Realizar partición
    train = dataset[0:Ntrain]
    val = dataset[Ntrain:Ntrain+Nval]
    test = dataset[Ntrain+Nval:]

    return train, val, test

def crear_dataset_supervisado(array, input_length, output_length):
    '''Permite crear un dataset con las entradas (X) y salidas (Y)
    requeridas por la Red LSTM.

    Parámetros:
    - array: arreglo numpy de tamaño N x features (N: cantidad de datos, f: cantidad de features)
    - input_length: instantes de tiempo consecutivos de la(s) serie(s) de tiempo usados para alimentar el modelo
    - output_length: instantes de tiempo a pronosticar (salida del modelo)
    '''

    # Inicialización
    X, Y = [], []    # Listados que contendrán los datos de entrada y salida del modelo
    shape = array.shape
    if len(shape)==1: # Si tenemos sólo una serie (univariado)
        fils, cols = array.shape[0], 1
        array = array.reshape(fils,cols)
    else: # Multivariado
        fils, cols = array.shape

    # Generar los arreglos
    for i in range(fils-input_length-output_length):
        X.append(array[i:i+input_length,0:cols])
        Y.append(array[i+input_length:i+input_length+output_length,-1].reshape(output_length,1))
    
    # Convertir listas a arreglos de NumPy
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

def escalar_dataset(data_input, col_ref, scaler, col_cat=[]):
  # Número de instantes de tiempo de entrada y de covariables
  NFEATS = data_input['x_tr'].shape[2]

  # Generar listado con "scalers" (1 por cada covariable de entrada)
  scalers = [scaler for i in range(NFEATS)]

  # Arreglos que contendrán los datasets escalados
  x_tr_s = np.zeros(data_input['x_tr'].shape)
  x_vl_s = np.zeros(data_input['x_vl'].shape)
  x_ts_s = np.zeros(data_input['x_ts'].shape)
  y_tr_s = np.zeros(data_input['y_tr'].shape)
  y_vl_s = np.zeros(data_input['y_vl'].shape)
  y_ts_s = np.zeros(data_input['y_ts'].shape)

  # Escalamiento: se usarán los min/max del set de entrenamiento para
  # escalar la totalidad de los datasets
  x_tr, y_tr = data_input['x_tr'], data_input['y_tr']
  x_vl, y_vl = data_input['x_vl'], data_input['y_vl']
  x_ts, y_ts = data_input['x_ts'], data_input['y_ts']

  # Escalamiento Xs
  for i in range(NFEATS):
      if i not in col_cat:
        x_tr_s[:,:,i] = scalers[i].fit_transform(x_tr[:,:,i])
        x_vl_s[:,:,i] = scalers[i].transform(x_vl[:,:,i])
        x_ts_s[:,:,i] = scalers[i].transform(x_ts[:,:,i])
      else:
        x_tr_s[:,:,i] = x_tr[:,:,i]
        x_vl_s[:,:,i] = x_vl[:,:,i]
        x_ts_s[:,:,i] = x_ts[:,:,i]
  
  # Escalamiento Ys (teniendo en cuenta "col_ind")
  y_tr_s[:,:,0] = scalers[col_ref].fit_transform(y_tr[:,:,0])
  y_vl_s[:,:,0] = scalers[col_ref].transform(y_vl[:,:,0])
  y_ts_s[:,:,0] = scalers[col_ref].transform(y_ts[:,:,0])

  # Conformar diccionario de salida
  data_scaled = {
      'x_tr_s': x_tr_s, 'y_tr_s': y_tr_s,
      'x_vl_s': x_vl_s, 'y_vl_s': y_vl_s,
      'x_ts_s': x_ts_s, 'y_ts_s': y_ts_s,
  }

  return data_scaled, scalers[col_ref]

def cargar_dataset(dataset, div_split, inout_shape, name_scaler, conv):
    # 1. dividir en train, val, test
    tr, vl, ts = split_dataset_byN(dataset, div_split)

    # 2. Generar salidas y entradar (dataset supervisado) para el entrenamiento en cada dataset
    x_tr, y_tr = crear_dataset_supervisado(tr.values, inout_shape[0], inout_shape[1])
    x_vl, y_vl = crear_dataset_supervisado(vl.values, inout_shape[0], inout_shape[1])
    x_ts, y_ts = crear_dataset_supervisado(ts.values, inout_shape[0], inout_shape[1])

    # 3 .escalar
    # Crear diccionariinout_shape[0]
    data_in = {
        'x_tr': x_tr, 'y_tr': y_tr,
        'x_vl': x_vl, 'y_vl': y_vl,
        'x_ts': x_ts, 'y_ts': y_ts,
    }

    col_y = dataset.columns.get_loc('travel_time')
    scaler = {
        'Maxmin': MinMaxScaler(feature_range=(-1,1)),
        'Standar': StandardScaler(),
        'Robust': RobustScaler()
    }
    col_cat = [0, 1, 2]
    data_s, scaler = escalar_dataset(data_in, col_y, scaler[name_scaler], col_cat)

    if conv:
        for key, set in data_s.items():
            data_s[key] = set.reshape(set.shape[0], set.shape[1], set.shape[2], 1, 1)
            #print(key, data_s[key].shape)

    data_s['tr'], data_s['vl'], data_s['ts'] = tr, vl, ts
    data_s['x_tr'], data_s['y_tr'] = x_tr, y_tr
    data_s['x_vl'], data_s['y_vl'] = x_vl, y_vl
    data_s['x_ts'], data_s['y_ts'] = x_ts, y_ts
    
    return data_s, scaler

def find_input(df, stop, time, input_size):
    # encontramos la muestra con la ultima fila coicidente
    found = df[(df['id_linkref']==stop) & (df['horas']==time[0]) & (df['minute']==time[1])].index
    # retornamos la secuencia anterior a los buscado
    return df.iloc[found - input_size].index

def predecir(x, model, scaler, conv=False):
    # Calcular predicción escalada en el rango de -1 a 1
    y_pred_s = model.predict(x,verbose=0)

    # Llevar la predicción a la escala original
    if conv:
        y_pred = scaler.inverse_transform(y_pred_s.reshape((-1, 1)))
    else:
        y_pred = scaler.inverse_transform(y_pred_s)

    return y_pred.flatten()


def predict_travel_time(n_links, shape_input, id_sample, data_in_s, data_in, modelo, scaler, conv, show_pred=False):
    n = id_sample
    input_of_pred = data_in_s[n].reshape(shape_input) # 14 - 53 => 15:30 - 16:00

    real = data_in[n + 40][:n_links, -1]

    y_of_set = []
    y_of_pred = [] 
    for i in range(n_links):
        # En base a muestras del dataset test
        input_of_set = data_in_s[n + i].reshape(shape_input)
        y_of_set.append(predecir(input_of_set, modelo, scaler, conv=conv)[0])

        # Concadenando predicciones previas
        y_s = modelo.predict(input_of_pred, verbose=0)
        y_of_pred.append(scaler.inverse_transform(y_s.reshape(-1, 1))[0][0])
        new_row = data_in_s[n+i+1][-1].copy()
        if conv:
            new_row[-1][0] = y_s.flatten()[0]
        else:
            new_row[-1] = y_s
        new_row =  new_row.reshape((1, ) + shape_input[2:])
        input_of_pred = np.vstack((input_of_pred[0][1:], new_row)).reshape(shape_input).astype(np.float64)

        # show predicts
        if show_pred:
            id_stop = input_of_set[0][-1][0][0, 0] if conv else input_of_set[0][-1][0]
            print(f"stop: {id_stop+1:0.0f} => {real[i]}\t{y_of_set[-1]:0.1f}\t{y_of_pred[-1]:0.1f}")
    
    return real, y_of_set, y_of_pred

