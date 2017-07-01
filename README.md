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
