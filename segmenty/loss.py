from keras import backend as K


def balanced_loss(y_true, y_pred):
    sq = K.square(y_pred - y_true)
    gt = K.cast(K.greater(y_true, 0.5), dtype='float')
    lt = K.cast(K.less(y_true, 0.5), dtype='float')
    pos = (K.sum(gt * sq, axis=[-3, -2]) /
           K.maximum(K.sum(gt, axis=[-3, -2]), 1.0))
    neg = (K.sum(lt * sq, axis=[-3, -2]) /
           K.maximum(K.sum(lt, axis=[-3, -2]), 1.0))
    return (pos + neg) * 0.5


