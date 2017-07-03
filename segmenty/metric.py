import keras.backend as K


def tpr(y_true, y_pred):
    t = 1 - K.cast(K.less(y_true, 0.5), dtype='float32')
    p = 1 - K.cast(K.less(y_pred, 0.5), dtype='float32')
    tpr = (K.sum(t * p, axis=[-3, -2]) /
           K.maximum(K.sum(t, axis=[-3, -2]), 1.0))
    return tpr


def tnr(y_true, y_pred):
    t = K.cast(K.less(y_true, 0.5), dtype='float32')
    n = K.cast(K.less(y_pred, 0.5), dtype='float32')
    tnr = K.sum(t * n, axis=[-3, -2]) / K.maximum(K.sum(t, axis=[-3, -2]), 1.0)
    return tnr


