
Computer vision is in some ways the inverse problem to computer graphics.
Neural nets are pretty good at solving inverse problems, given enough
examples.  Computer graphics can generate a lot of examples.  If the
examples are realistic enough (a big if), then to solve the computer vision
problem all you need to do is write computer graphics code and throw
a neural network at it.

Steps:

 * Install opencv, opencv python wrappers, and imagemagick
 * `pip install pixplz[parallel] svgwrite tensorflow-gpu keras`
 * `./fetch_backgrounds.sh`
 * `./make_patterns.sh`
 * `./make_samples.py validation 200`
 * `./make_samples.py training 3000`
 * `./train_segmenter.py`
 * While that is running, also run `./freshen_samples.sh` in a separate console.
   This will replace the training examples periodically to combat overfitting.
 * Run `./test_segmenter.py /tmp/model.thing validation scratch` from time to
   time and look at visual quality of results in `scratch`.
 * Kill training when results are acceptable.
