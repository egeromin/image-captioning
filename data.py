"""
Define data generators for use with a keras model

- add shuffling, cropping, and filling in
"""
import sys
from keras import backend as K

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import ipdb

from convert_to_tfrecord import parse_tfrecord, \
    load_conversions
import config


def tuple_to_list_middleware(inputs_captions):
    """
    Keras expects multiple inputs to be in a list, 
    whereas tensorflow dataset can only return tuples

    So apply this `middleware'
    """
    (a, b), c = inputs_captions
    return [a, b], c


class DataGenerator:
    """Take a tensorflow dataset object and returns a 
    generator that can be used with keras' fit_generator
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def generate(self):
        iterator = self.dataset.make_one_shot_iterator()
        # iterator = tfe.Iterator(self.dataset)

        images_and_captions = iterator.get_next()
        while True:
            yield K.get_session().run(
                tuple_to_list_middleware(images_and_captions))
            # yield next(iterator)


def make_data_generator(stage='train', num_chunks=1):

    word_from_id, id_from_word, seq_length = load_conversions()
    vocabulary_size = len(word_from_id)
    pattern_tfrecord  = "./data/mscoco/{stage}.tfrecord.*".format(stage=stage)
    dataset = parse_tfrecord(pattern_tfrecord=pattern_tfrecord, 
                             gzipped=True,
                             num_chunks=num_chunks,
                             end_token=id_from_word['.'],
                             seq_length=seq_length,
                             vocabulary_size=vocabulary_size)
    dataset = dataset.repeat()  # repeat dataset indefinitely
    dataset = dataset.batch(config.batch_size)
    # ipdb.set_trace()
    return DataGenerator(dataset)


if __name__ == "__main__":
    # tf.enable_eager_execution()
    # pattern_tfrecord  = "./data/mscoco/{stage}.tfrecord.*".format(stage='train')
    # print(pattern_tfrecord)
    # dataset = tf.data.TFRecordDataset.list_files(
    #     file_pattern=pattern_tfrecord
    # )

    # iterator = tfe.Iterator(dataset)
    # print(list(iterator))

    # sys.exit(1)

    generator = make_data_generator(stage='val')
    list(generator.generate())

