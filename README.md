
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
are shown on the right.  There are masks for the full pattern, for vertical and
horizontal grid lines, and for the center of the pattern (if visible).
Occasional images without the target pattern are also included.

![example inputs](https://user-images.githubusercontent.com/118367/27806154-c5312030-5fed-11e7-9d12-807d831415de.png)

Here are how the images are generated:

 * Grab a large number of random images from anywhere to use as backgrounds.
 * Shade the pattern with random gradients.
   This is a crude simulation of lighting effects.
 * Overlay the pattern on a background, with a random affine transformation.
   This is a crude simulation of perspective.
 * Draw random blobs here and there across the image.
   This is a crude simulation of occlusion.
 * Overlay another random image on the result so far, with a random level
   of transparency.  This is a crude simulation of reflections, shadows,
   more lighting effects.
 * Distort the image randomly.
   This is a crude simulation of camera projection effects.
 * Apply a random directional blur to the image.
   This is a crude simulation of motion blur.
 * Sometimes, just leave the pattern out of all this, and leave the mask
   empty, if you want to let the network know the pattern may not
   always be present.
 * Randomize the overall lighting.

The precise details are hopefully not too important,
the key is to leave nothing reliable except the pattern you want learned.
To classic computer vision eyes, this all looks crazy.  There's occlusion!
Sometimes the pattern is only partially in view!  Sometimes its edges are
all smeared out!  Relax about that.  It is not our problem anymore.

The mechanics of training doesn't actually depend much on the actual
pattern to detect and the masks to return (although the quality of the
result may).  For this run the pattern and masks were:

![masks](https://user-images.githubusercontent.com/118367/27806980-ba1b1656-5ff2-11e7-8af5-21e16cf0a9e7.png)


Progress on real data
---------------------

Here are some real photos of the target pattern.  The network never
gets to see photos like this during training.  This animation shows
network output at every 10th epoch across a training run.

![iterations](https://user-images.githubusercontent.com/118367/27806155-c5316bee-5fed-11e7-928c-bbe38e2f1174.gif)

How to train
------------

 * Install opencv, opencv python wrappers, and imagemagick
 * `pip install pixplz[parallel] svgwrite tensorflow-gpu keras Pillow`
 * `./fetch_backgrounds.sh`
 * `./make_patterns.sh`
 * `./make_samples.py validation 200`
 * `./make_samples.py training 3000`
 * `./train_segmenter.py`
 * While that is running, also run `./freshen_samples.sh` in a separate console.
   This will replace the training examples periodically to combat overfitting.
   If you don't want to do this, use a much, much bigger number than 3000
   for the number of training images.
 * Run `./test_segmenter.py /tmp/model.thing validation scratch` from time to
   time and look at visual quality of results in `scratch`.
 * Kill training when results are acceptable.
