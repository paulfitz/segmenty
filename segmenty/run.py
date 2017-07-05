from segmenty.create import load_or_create_model, persist_model
from segmenty.image_to_image import ImageToImage
from segmenty.metric import tpr, tnr
from segmenty.loss import balanced_loss


class Run(object):

    def __init__(self):
        pass

    def set_flags(self, flags):
        self.v_i2i = ImageToImage('validation.json', **flags)
        self.i2i = ImageToImage('training.json', **flags)
        self.flags = flags
        self.flags['example'] = self.v_i2i.example

    def use_model(self, constructor, filename='root.thing'):
        self.model = load_or_create_model(
            constructor=constructor,
            flags=self.flags,
            filename=filename,
            custom_objects=[balanced_loss, tpr, tnr])
        return self.model

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss=balanced_loss,
                           metrics=[tpr, tnr, 'mse'])

    def train(self, filename='save.thing', checkpoint='/tmp/model.thing'):
        with persist_model(self.model,
                           filename,
                           checkpoint) as callbacks:
            self.model.fit_generator(self.i2i.generator(),
                                     steps_per_epoch=self.i2i.steps,
                                     epochs=100000,
                                     validation_data=self.v_i2i.generator(),
                                     validation_steps=self.v_i2i.steps,
                                     callbacks=callbacks)
