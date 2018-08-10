import tensorflow as tf
import keras
from keras.models import load_model, Model
import os
import numpy as np

from collections import OrderedDict

import ipdb
import argparse
import json

from convert_to_tfrecord import load_image
import config
from data import make_data_generator
from convert_to_tfrecord import load_conversions

from model import ExpandDims, image_captioning_model



def load_images(images_dir, limit=None):

    file_list = sorted(os.listdir(images_dir))
    if limit is not None:
        file_list = file_list[:limit]

    def crop_center(img):
        img_size = config.image_input_size[0]
        beg = (config.image_resize_size[0] - 
               img_size) // 2
        end = beg + img_size

        return img[beg:end,beg:end,:] # center crop
        
    images = []
    for path_image in file_list:
        path_image = os.path.join(images_dir, path_image)
        image = load_image(None, path=path_image)
        image = crop_center(image)
        images.append(image)

    images = np.array(images)
    # print(images.shape)
    assert(images.shape[1:] == config.image_input_size + (3,))
    return images


def make_prediction(model, input_images, word_from_id, seq_length=10):
    imagenet_layer = Model(inputs=model.input,
                           outputs=model.layers[1].get_output_at(-1))

    captions = np.zeros((input_images.shape[0], seq_length), dtype=np.int32)

    for i in range(seq_length-1):
        predicted_captions = model.predict([input_images, captions])  # output is 1-hot encoded
        assert(len(predicted_captions.shape) == 3)
        predicted_captions = predicted_captions[:,:-1,:]  # drop the last token
        assert(predicted_captions.shape[0:2] == captions.shape)

        captions[:,i+1] = \
                predicted_captions[:,i+1,:].argmax(axis=-1)

    imagenet_output = imagenet_layer.predict([input_images, captions])
    ipdb.set_trace() # check if the output of the imagenet model is always
    # constant

    # word ids to sentences
    prepared_captions = []
    for i in range(captions.shape[0]):
        caption_words = []
        for j in range(captions.shape[1]):
            word_id = captions[i,j]
            caption_words.append(word_from_id[word_id])

        prepared_captions.append(" ".join(caption_words))

    return prepared_captions


def main():
    parser = argparse.ArgumentParser(description="predict using an image "
                                     "captioning model")
    parser.add_argument("--model_path", help="Path to the trained model", 
                        default="./checkpoints/final.hdf5")
    parser.add_argument("--image_dir", default="./data/mscoco/train2014",
                        help="Path to the images to caption")
    parser.add_argument("--num_images", type=int, default=config.batch_size)
    parser.add_argument("--results", help="Path to results file", 
                        default="./captions.json")

    args = parser.parse_args()

    word_from_id, id_from_word, seq_length = load_conversions()
    vocabulary_size = len(word_from_id)
    model = image_captioning_model(seq_length, 1000,
                                   vocabulary_size)
    model.load_weights(args.model_path)

    # model = load_model(args.model_path, custom_objects={'ExpandDims':
    #                                                     ExpandDims})
    # cannot use load_model because didn't define a config for
    # the custom ExpandDims layer
    images = load_images(args.image_dir, limit=args.num_images)
    captions = make_prediction(model, images, word_from_id, seq_length)
    
    file_list = sorted(os.listdir(args.image_dir))
    files_and_captions = OrderedDict(zip(file_list, captions))

    with open(args.results, "w") as fh:
        json.dump(files_and_captions, fh, indent=2)



if __name__ == "__main__":
    main()


