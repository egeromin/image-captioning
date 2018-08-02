"""
Global config for the image captioning pipeline.
"""

num_shapenet_classes = 1000
max_vocab_size = 10000
image_input_size = (299,299)
image_resize_size = (346,346)
batch_size = 32

# compute the step size = sequence length on the fly, depending how long the
# longest caption is.

# alternatively discard all captions above a certain length? Try the latter
# approach too. 
