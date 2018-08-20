import tensorflow as tf
import keras
from keras.models import load_model, Model
from keras import backend as K
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


def get_model_components(model):
    """
    Take the keras model and produce a dictionary with:
        
        - the imagenet layer as keras model
        - the embedding as tensor
        - the lstm as keras `LSTMCell`
        - the final dense layer

    These components can then be used separately.
    """

    imagenet_layer = Model(inputs=model.get_layer('input_1').output,
                           outputs=model.get_layer('inception_v3').
                           get_output_at(-1))

    embedding = model.get_layer('embedding_1').embeddings
    lstm_cell = model.get_layer('lstm_1').cell
    dense = model.get_layer('time_distributed_1').layer

    return imagenet_layer, embedding, lstm_cell, dense


def compute_initial_state(model, input_images):
    """Compute the initial state, given the input images"""

    imagenet_layer, _, lstm_cell, _ = get_model_components(model)
    image_classes = imagenet_layer.predict(input_images)
    image_classes_tensor = tf.convert_to_tensor(image_classes)
    initial_state = [
        tf.zeros((input_images.shape[0], config.lstm_hidden_size),
                 dtype=tf.float32),
        tf.zeros((input_images.shape[0], config.lstm_hidden_size),
                 dtype=tf.float32)
    ]
    _, state = lstm_cell.call(image_classes_tensor, initial_state)
    return K.get_session().run(state)


def compute_next_word_distribution(model, word_value,
                                   current_state_value,
                                   hidden_state_value):
    """Compute the distribution of the next word in the sequence,
    given the current word and the current state"""

    _, embedding, lstm_cell, dense = get_model_components(model)

    word = tf.placeholder(dtype=tf.int32,
                          shape=word_value.shape)

    current_state = tf.placeholder(dtype=tf.float32,
                                   shape=current_state_value.shape)
    hidden_state =  tf.placeholder(dtype=tf.float32,
                                   shape=hidden_state_value.shape)

    state = [current_state, hidden_state]

    word_embedding = tf.gather(embedding, word)
    output, next_state = lstm_cell.call(word_embedding, state)
    word_distribution = tf.nn.softmax(dense(output))
    
    return K.get_session().run((word_distribution, next_state),
                               feed_dict={
                                   word: word_value,
                                   current_state: current_state_value,
                                   hidden_state: hidden_state_value
                               })


def initialise_candidate_sentences(beam_window, 
                                   word_dist,
                                   current_state_value,
                                   hidden_state_value):
    top_k_next_words = \
        np.argpartition(word_dist, beam_window)[:,-beam_window:]
    probs = np.take_along_axis(word_dist,
                               top_k_next_words,
                               axis=1)
    word_value = top_k_next_words.reshape(-1)
    probs = probs.reshape(-1)

    # expand state
    current_state_value = np.repeat(
        current_state_value, beam_window, axis=0)
    hidden_state_value = np.repeat(
        hidden_state_value, beam_window, axis=0)

    return word_value, current_state_value, hidden_state_value, probs


def select_new_candidates(beam_window,
                          word_dist,
                          current_state_value,
                          hidden_state_value,
                          probs):
    top_k_next_words = \
        np.argpartition(word_dist, beam_window)[:,-beam_window:]
    new_current_probs = np.take_along_axis(word_dist,
                                           top_k_next_words,
                                           axis=1)
    cumulative_probs = probs.reshape((-1, 1)) * new_current_probs
    num_images = word_dist.shape[0] // beam_window
    selected_sentences = np.zeros(probs.shape, dtype=np.int32)
    new_probs = np.zeros(probs.shape)
    new_word_value = np.zeros(probs.shape, dtype=np.int32)

    for i in range(num_images):
        prob_square = cumulative_probs[
            i*beam_window:(i+1)*beam_window,:].reshape(-1)
        top_prob_indices = np.argpartition(prob_square,
                                           beam_window)[-beam_window:]
        top_probs = prob_square[top_prob_indices]
        selected_sentences[i*beam_window:(i+1)*beam_window] = \
            top_prob_indices // beam_window
        new_probs[i*beam_window:(i+1)*beam_window] = top_probs

        words_in_square = \
            top_k_next_words[i*beam_window:(i+1)*beam_window].reshape(-1)
        new_words = words_in_square[top_prob_indices]
        new_word_value[i*beam_window:(i+1)*beam_window] = new_words

    return new_word_value, new_probs, selected_sentences


def make_prediction(model, input_images, word_from_id, seq_length=10,
                    beam_window=20):
    """
    Generate sentences using beam search, i.e. at each time step
    retain the K sentences with highest probability and use those to 
    generate the sentences for the next time step
    """
    current_state_value, hidden_state_value = compute_initial_state(model,
                                                                    input_images)
    vocabulary_size = max(word_from_id.keys()) + 1
    word_value = np.ones((input_images.shape[0],),
                         dtype=np.int32) * vocabulary_size
    captions = np.zeros((input_images.shape[0] * beam_window, seq_length), 
                        dtype=np.int32)

    for i in range(seq_length):
        word_dist, (current_state_value, hidden_state_value) = \
                compute_next_word_distribution(model,
                                               word_value,
                                               current_state_value,
                                               hidden_state_value)
        if i == 0:
            # first word initialises our captions to contain beam_window
            # candidate sentences per image
            word_value, current_state_value, \
            hidden_state_value, probs = \
                initialise_candidate_sentences(beam_window,
                                               word_dist,
                                               current_state_value,
                                               hidden_state_value)
            captions[:,i] = word_value

        else:
            word_value, new_probs, selected_sentences = \
                select_new_candidates(beam_window,
                                      word_dist,
                                      current_state_value,
                                      hidden_state_value,
                                      probs)
            current_state_value = current_state_value[selected_sentences,:]
            hidden_state_value = hidden_state_value[selected_sentences,:]
            captions = captions[selected_sentences,:]
            captions[:,i] = word_value

    # word ids to sentences
    prepared_captions = []
    for i in range(captions.shape[0]):
        if i % beam_window == 0:
            prepared_captions.append([])

        caption_words = []
        for j in range(captions.shape[1]):
            word_id = captions[i,j]
            caption_words.append(word_from_id[word_id])

        prepared_captions[-1].append(" ".join(caption_words))

    return prepared_captions


def main():
    parser = argparse.ArgumentParser(description="predict using an image "
                                     "captioning model")
    parser.add_argument("--model_path", help="Path to the trained model", 
                        default="./checkpoints/final.hdf5")
    parser.add_argument("--image_dir", default="./data/mscoco/test-images",
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


