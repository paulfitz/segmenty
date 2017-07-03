
Computer vision is in some ways the inverse problem to computer graphics.
Neural nets are pretty good at solving inverse problems, given enough
examples.  Computer graphics can generate a lot of examples.  If the
examples are realistic enough (a big if), then to solve the computer vision
problem all you need to do is write computer graphics code and throw
a neural network at it.

This is an experiment in recognizing printed patterns in real images
based on training on generated images.

Synthetic test data
-------------------

Generated images on left.  Masks that we want the network to output
on right.  There are masks for the full pattern, for vertical and
horizontal grid lines, and for the center of the pattern (if visible).
Occasional images without the target pattern are also included.

![example inputs](https://user-images.githubusercontent.com/118367/27806154-c5312030-5fed-11e7-9d12-807d831415de.png)

Progress on real data
---------------------

Here are some real photos of the target pattern.  The network never
gets to see photos like this during training.  This animation shows
network output at every 10th epoch across a training run.

![iterations](https://user-images.githubusercontent.com/118367/27806155-c5316bee-5fed-11e7-928c-bbe38e2f1174.gif)

Steps
-----

 * Install opencv, opencv python wrappers, and imagemagick
 * `pip install pixplz[parallel] svgwrite tensorflow-gpu keras Pillow`
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
