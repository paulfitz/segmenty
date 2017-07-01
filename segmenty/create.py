from keras.callbacks import CSVLogger, LambdaCallback, ModelCheckpoint
from keras.models import load_model
import os
from segmenty.process import process_images


def load_or_create_model(constructor, flags, filename, custom_objects):
    custom_objects = dict((o.__name__, o) for o in custom_objects)
    if os.path.exists(filename):
        mod = load_model(filename, custom_objects=custom_objects)
    else:
        mod = constructor(flags)
    mod.summary()
    return mod


class persist_model:
    def __init__(self, obj, filename, checkpoint):
        self.obj = obj
        self.filename = filename
        self.checkpoint = checkpoint

    def __enter__(self):
        chk = ModelCheckpoint(self.checkpoint, verbose=0, save_best_only=False,
                              save_weights_only=False, mode='auto')
        csv_logger = CSVLogger('training.log')
        snaps = LambdaCallback(on_epoch_end=lambda epoch, logs: self.snap(epoch))
        return [chk, csv_logger, snaps]

    def __exit__(self, type, value, traceback):
        self.obj.save(self.filename)
        print("model saved: {}".format(self.filename))

    def snap(self, epoch):
        import glob
        lst = glob.glob('wild/*.jpg')
        if len(lst) == 0:
            return
        process_images(self.obj, lst, 'snapshots', '%06d' % epoch)
