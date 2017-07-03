#!/usr/bin/env python

from keras.models import Model
from keras.layers import Conv2D, Dropout, Input
from segmenty.create import load_or_create_model, persist_model
from segmenty.hourglass import downsamples, mix, upsamples
from segmenty.image_to_image import ImageToImage
from segmenty.metric import tpr, tnr
from segmenty.loss import balanced_loss


flags = {
    'image_size': (256, 256),
    'batch_size': 25
}


def model(flags):
    
    inputs = Input(shape=(flags['image_size'] + (3,)))
    x = inputs

    x, sources = downsamples(x, [40, 40, 80, 100, 100, 100, 80, 80])
    x = mix(x, 100)
    x = Dropout(0.1)(x)
    x = upsamples(x, sources, [100] * 8)

    x = Conv2D(30, (1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(30, (1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(len(flags['example']['y']), (1, 1), border_mode='valid',
               activation='sigmoid')(x)

    mod = Model(input=inputs, outputs=x)

    return mod


v_i2i = ImageToImage('validation.json', **flags)
i2i = ImageToImage('training.json', **flags)

flags['example'] = v_i2i.example


mod = load_or_create_model(
    constructor=model,
    flags=flags,
    filename='root.thing',
    custom_objects=[balanced_loss, tpr, tnr])

mod.compile(optimizer='adam', loss=balanced_loss, metrics=[tpr, tnr, 'mse'])

with persist_model(mod, 'save.thing', '/tmp/model.thing') as callbacks:
    mod.fit_generator(i2i.generator(),
                      steps_per_epoch=i2i.steps,
                      epochs=100000,
                      validation_data=v_i2i.generator(),
                      validation_steps=v_i2i.steps,
                      callbacks=callbacks)
