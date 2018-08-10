"""
Convert the MSCOCO dataset to tfrecord format

Generates an image for each of the captions. 

For cropping, takes the approach in im2tell:

- resize to 346 * 346, then do a random crop.

The random cropping is done during train time. The resizing should be as well,
because then we can experiment with different sizes during training. However,
to save disk space, we resize in advance when making the tfrecord file. 
"""


from PIL import Image
import numpy as np
import json
import re
from collections import defaultdict
import itertools
import random
import unittest
import argparse
import tempfile
import sys
import os
import subprocess
import itertools

import ipdb

import numpy as np
from scipy.misc import imsave
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from image_captioning import config
# from image_captioning import config
# from image_captioning import config
# from image_captioning.data import make_data_generator




CAPTIONS_FILE = "./data/mscoco/annotations/captions_{stage}2014.json"
IMAGES_DIR = "./data/mscoco/{stage}2014"
WORD_ID_CONVERSION = "./data/mscoco/word_id.json"

TFRECORD_FILE = "./data/mscoco/{stage}.tfrecord.{chunk}"


def get_full_image_path(image_id, stage='train'):
    padded_image_id = str(image_id).zfill(12)

    return "{dir}/COCO_{stage}2014_{padded_id}.jpg".format(
        dir=IMAGES_DIR.format(stage=stage), padded_id=padded_image_id,
        stage=stage)


def sanitize_caption(caption):
    return re.sub('-', ' -', re.sub('[.,"]', '', caption)).lower().split()


def load_image(image_id, stage='train', path=None):
    """Returns the image corresponding to the image ID"""
    def _resize_image(image):
        return image.resize(config.image_resize_size)

    if path is None:
        path = get_full_image_path(image_id, stage)

    image = Image.open(path)
    resized_image = _resize_image(image)
    np_image = np.array(resized_image)
    if len(np_image.shape) == 2:  # sometimes we get a black-and-white image
        np_image = np.broadcast_to(np_image, (3,) + np_image.shape )
        np_image = np.moveaxis(np_image, 0, -1)

    return np_image


# change strategy
# first load all of the captions so that I can compute the tokens
# and then stream the images


def load_captions(stage='train'):
    """
    Load the text captions for a given stage, where 
    stage is 'train', 'test' or 'val'
    """
    if stage not in ('train', 'test', 'val'):
        raise RuntimeError("stage must be 'train', 'test' or 'val'")
    loaded_data = [] 
    word_count_dictionary = defaultdict(int)
    with open(CAPTIONS_FILE.format(stage=stage)) as fh:
        print("Loading captions")
        all_annotations = json.load(fh)['annotations']

    word_count_dictionary = defaultdict(int)

    def count_words(caption):
        for word in caption:
            word_count_dictionary[word] += 1

    seq_length = 0
    for caption in all_annotations:
        item = {k: caption[k] for k in ('image_id', 'id', 'caption')}
        item['sanitized_caption'] = sanitize_caption(item['caption'])
        loaded_data.append(item)

        if stage == "train":
            count_words(item['sanitized_caption'])
            seq_length = max(seq_length, len(item['sanitized_caption']) + 1)

    return loaded_data, word_count_dictionary, seq_length
    

def make_id_word_conversions(training_data, word_count_dictionary):
    """
    Takes the full training data and computes

        - id_from_word to id (dictionary)
        - word_from_id (list)
    """
    most_frequent_words = sorted(word_count_dictionary.items(), key=lambda x: -x[1])
    most_frequent_words = [x[0] for x in
                           most_frequent_words][:config.max_vocab_size - 2]
    most_frequent_words.extend(['.', '<unk>'])

    # Here, we add a default to return the index of
    # "<unk>", which is the last element of most_frequent_words
    id_from_word = defaultdict( lambda: len(most_frequent_words) - 1, dict(zip(most_frequent_words,
                                         range(config.max_vocab_size))))
    word_from_id = {v:k for k,v in id_from_word.items()}
    return id_from_word, word_from_id


def tokens_from_data(input_data, id_from_word):
    """
    Given input data and id_from_word, make the captions for each data
    item by populating the `tokenized_caption` attribute.
    """

    def get_token(word):
        return id_from_word.get(word, id_from_word['<unk>'])

    for item in input_data:
        item['tokenized_caption'] = [get_token(word) for word in
                                     item['sanitized_caption']]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def _int_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def write_single_record(tfrecord, data, seq_length, stage):
    """
    Write a single item of data to the file. 
    """
    caption = data['tokenized_caption']
    assert(len(caption) == len(data['sanitized_caption']))
    np_caption = np.array(caption).astype(np.int32)
    image = load_image(data['image_id'], stage)

    feature = {
        'image': _bytes_feature(image.tobytes()),
        'caption': _bytes_feature(np_caption.tobytes())
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    tfrecord.write(example.SerializeToString())



def load_conversions():
    with open(WORD_ID_CONVERSION, 'r') as fh:
        conversions = json.load(fh)
        word_from_id = conversions['word_from_id']
        id_from_word = conversions['id_from_word']
        seq_length   = conversions['seq_length']
    
    # integers must be converted explicitly 
    word_from_id = {int(k): v for k, v in word_from_id.items()}
    id_from_word = {k: int(v) for k, v in id_from_word.items()}
    return word_from_id, id_from_word, seq_length



def to_tfrecord(stage='train', shuffle=True, limit=None, path_tfrecord=None,
                chunk=1, num_chunks=1):
    """
    Convert a specific stage to tfrecord. If the stage is train, it also
    computes the id_from_word and word_from_id dictionaries; otherwise,
    it attempts to read these from file.
    """
    loaded_data, word_dictionary, seq_length = load_captions(stage=stage)
    if stage == 'train':
        id_from_word, word_from_id = make_id_word_conversions(loaded_data,
                                                          word_dictionary)
        with open(WORD_ID_CONVERSION, 'w') as fh:
            json.dump({
                'id_from_word': id_from_word,
                'word_from_id': word_from_id,
                'seq_length': seq_length
            }, fh)
    else:
        word_from_id, id_from_word, seq_length = load_conversions()

    tokens_from_data(loaded_data, id_from_word)

    if stage=='train' and shuffle: # only shuffle for training
        random.seed(123)  # use random seed for consistent shuffling
        random.shuffle(loaded_data)

    if limit is None:
        limit = len(loaded_data)

    loaded_data = loaded_data[:limit]
    
    sz_chunk = limit / num_chunks
    beginning = int((chunk-1) * sz_chunk)
    end = int(chunk * sz_chunk)
    print(beginning, end)
    loaded_data = loaded_data[beginning:end]

    if path_tfrecord is None:
        path_tfrecord = TFRECORD_FILE.format(stage=stage, chunk=chunk)
    with tf.python_io.TFRecordWriter(path_tfrecord) as tfrecord:
        for i, data in enumerate(loaded_data):

            write_single_record(tfrecord, data, seq_length, stage)

            if (i+1) % 1000 == 0:
                print("Processed {} of {}".format(i+1, int(sz_chunk)))
    
    
def main():
    parser = argparse.ArgumentParser(description="convert mscoco data to "
                                     "tfrecord format")
    parser.add_argument("--test", help="run test case?",
                        action='store_true')
    parser.add_argument("--limit", help="Limit processing to top N images",
                        type=int)
    parser.add_argument("--chunk", help="Chunk index", 
                        type=int, default=1)
    parser.add_argument("--num_chunks", help="Total number of chunks",
                        type=int, default=1)
    parser.add_argument("--stage", help="Stage to process. One of 'train', "
                        "'val' or 'test'", default='train')
    args = parser.parse_args()

    print(args.test)

    if args.test:
        tf.enable_eager_execution()
        sys.argv = list(set(sys.argv) - {'--test', '-t'}) 
        # slight hack. `unittest.main` also reads command line arguments,
        # so need to sanitize them before invoking `unittest.main`
        unittest.main()
    else:
        to_tfrecord(args.stage, limit=args.limit, chunk=args.chunk,
                    num_chunks=args.num_chunks)


def parse_tfrecord(path_tfrecord=None, pattern_tfrecord=None, gzipped=False,
                   num_chunks=1, seq_length=50, end_token=1,
                   vocabulary_size=10000):
    # ipdb.set_trace()
    if path_tfrecord is None and pattern_tfrecord is None:
        raise ValueError("One of `path_tfrecord` or `pattern_tfrecord` "
                         " must be set")
    def parse_tfexample(example_proto):
        features = {
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string), 
            'caption': tf.FixedLenFeature(shape=[], dtype=tf.string)
        }
        parsed_features = tf.parse_single_example(
            serialized=example_proto, features=features)
        image = tf.reshape(tf.cast(
            tf.decode_raw(parsed_features['image'], out_type=tf.uint8),
            tf.float32), config.image_resize_size + (3,))

        image = tf.random_crop(image, size=config.image_input_size + (3,))
        # random crop

        caption_no_pad = tf.decode_raw(parsed_features['caption'], 
                                       out_type=tf.int32)
        padding = tf.ones(shape=[seq_length - tf.shape(caption_no_pad)[0]],
                          dtype=tf.int32) * end_token
        padding_output = tf.ones(shape=[1 + seq_length - tf.shape(caption_no_pad)[0]],
                          dtype=tf.int32) * end_token
        start_word = tf.ones(shape=[1], dtype=tf.int32) * vocabulary_size

        caption = tf.concat([start_word, caption_no_pad, padding], axis=0)
        caption_output = tf.concat([caption_no_pad, padding_output], axis=0)

        one_hot_caption = tf.one_hot(caption_output, depth=vocabulary_size,
                                     dtype=tf.float32)
        # caption padded with full stops to reach seq_length
        return (image, caption), one_hot_caption
    
    compression_type = ""
    if gzipped:
        compression_type = "GZIP"

    def read_fromfile(path_tf):
        return tf.data.TFRecordDataset(path_tf,
                                       compression_type=compression_type)

    if path_tfrecord is not None:
        dataset = read_fromfile(path_tf)
    else:
        dataset = tf.data.TFRecordDataset.list_files(
            file_pattern=pattern_tfrecord
        ).interleave(read_fromfile, cycle_length=num_chunks,
                     block_length=1)

    dataset = dataset.map(parse_tfexample)
    return dataset
    # return dataset.make_one_shot_iterator()
    # return tfe.Iterator(dataset)
    # return dataset
    

class TestConvert(unittest.TestCase):
    """
    Test the conversion to tfrecord format. 
    """

    def setUp(self):
        self.temp_tfrecord = tempfile.mktemp()
    
    def test_captions(self):
        loaded_data, word_dictionary, _ = load_captions(stage="train")
        id_from_word, word_from_id = make_id_word_conversions(loaded_data,
                                                          word_dictionary)
        for item in loaded_data[:10]:
            print([word_from_id[id_from_word[word]]
                   for word in item['sanitized_caption']])

        self.assertEqual(loaded_data[0]['sanitized_caption'],
                         ['a', 'very', 'clean', 'and', 'well', 'decorated',
                          'empty', 'bathroom'])

    def test_save_tfrecord(self):
        to_tfrecord(shuffle=False, limit=1, path_tfrecord=self.temp_tfrecord)
        word_from_id, id_from_word, seq_length = load_conversions()

        # now GZIP manually using the `gzip` command, as was done
        # in real life
        
        subprocess.call(['gzip', self.temp_tfrecord])

        path_gzipped = self.temp_tfrecord + ".gz"
        self.assertTrue(os.path.isfile(path_gzipped))
        self.assertFalse(os.path.isfile(self.temp_tfrecord))

        # check that the caption written is a very clean
        # and well decorated empty bathroom
        images_and_captions = parse_tfrecord(path_gzipped, gzipped=True,
                                             end_token=id_from_word['.'],
                                             seq_length=seq_length,
                                             vocabulary_size=len(word_from_id))
        images_and_captions = tfe.Iterator(images_and_captions)
        (image, caption), _ = next(images_and_captions)
        print(caption.shape)
        word_from_id, id_from_word, seq_length = load_conversions()
        print(caption.numpy())
        
        parsed_caption = [word_from_id[word] for word in caption.numpy()]
        target_caption = ['a', 'very', 'clean', 'and', 'well', 'decorated',
                          'empty', 'bathroom']
        
        target_caption.extend(itertools.repeat('.', seq_length -
                                               len(target_caption)))
        self.assertEqual(parsed_caption, target_caption)

        # temp_image = tempfile.mktemp(suffix=".png")
        # imsave(temp_image, image.numpy())
        # print(temp_image)

    def tearDown(self):
        if os.path.isfile(self.temp_tfrecord):
            os.remove(self.temp_tfrecord)
            # need to check for existence, because with mktemp,
            # the file is only created if it's used.

        # follow same approach to remove the gzipped file,
        # if necessary
        path_gzipped = self.temp_tfrecord + ".gz"
        if os.path.isfile(path_gzipped):
            os.remove(path_gzipped)



if __name__ == "__main__":
    main()

