from keras.layers import (BatchNormalization, Concatenate, Conv2D,
                          Conv2DTranspose, Dense, MaxPooling2D)


def downsamples(x, nfilts):
    def downsample(x, nfilt):
        x = BatchNormalization()(x)
        x = Conv2D(nfilt, (5, 5), padding='same', activation='relu')(x)
        x = Conv2D(nfilt, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x
    sources = []
    for nfilt in nfilts:
        sources.append(x)
        x = downsample(x, nfilt)
    return x, sources


def upsamples(x, sources, nfilts):
    def upsample(x, source, nfilt):
        x = Conv2DTranspose(nfilt, (3, 3), padding='same', strides=(2, 2))(x)
        x = Concatenate()([x, source])
        x = Conv2D(nfilt, (5, 5), padding='same', activation='relu')(x)
        x = Conv2D(nfilt, (3, 3), padding='same', activation='relu')(x)
        return x
    for source, nfilt in zip(reversed(sources), nfilts):
        x = upsample(x, source, nfilt)
    return x


def mix(x, nfilt):
    x = Dense(nfilt, activation='relu')(x)
    x = Dense(nfilt, activation='relu')(x)
    return x

