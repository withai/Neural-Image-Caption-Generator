"""An interface to train, test and inference the image captioning problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

from caption_generator import *
from utils.data_util import generate_captions
from utils.config import Config


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    help="In what model would you like to use Train | Test | Eval?",
    choices=[
        "train",
        "test",
        "eval"],
    required=True)
parser.add_argument(
    "--resume",
    help="make model training resumable",
    action="store_true")
parser.add_argument(
    "--caption_path",
    type=str,
    help="The location of Flicker8k Captions \'.token\' file.")
parser.add_argument(
    "--feature_path",
    type=str,
    help="The location of Flickr8k image features: features.npy file.")
parser.add_argument(
    "--inception_path",
    type=str,
    help="The location of \'inception_v4.pb\' to encode images.",
    default="Conv-Network/inception_v4.pb")
parser.add_argument(
    "--saveencoder",
    help="Save Encoder graph in model/Encoder/",
    action="store_true")
parser.add_argument(
    "--savedecoder",
    help="Save Decoder graph in model/Decoder/",
    action="store_true")
parser.add_argument(
    "--image_path",
    type=str,
    help="Image path required during test mode for Generation of Captions.")
parser.add_argument(
    "--load_image",
    help="If mode is test then, displays and stores image with generated caption",
    action="store_true")
parser.add_argument(
    "--validation_data",
    type=str,
    help="If mode is eval then, Path to the Validation Data for evaluation")
FLAGS = parser.parse_args()
config = Config(vars(FLAGS))

if(config.mode == "train"):
    data = generate_captions(
        config, FLAGS.caption_path, FLAGS.feature_path)
    model = Caption_Generator(config, data=data)
    loss, inp_dict = model.build_train_graph()
    model.train(loss, inp_dict)

elif config.mode == "test":
    if os.path.exists(FLAGS.image_path):
        model = Caption_Generator(config)
        model.decode(FLAGS.image_path)
    else:
        print("Please provide a valid image path.\n Usage:\n python main.py --mode test --image_path VALID_PATH")

elif config.mode == "eval":
    config.mode = "test"
    config.batch_decode = True
    print(FLAGS.validation_data)
    if os.path.exists(FLAGS.validation_data):
        features = np.load(FLAGS.validation_data)
        #with open("Dataset/Validation_Captions.txt") as f:
        #    data = f.readlines()
        with open("Dataset/image_info_test2014.json",'r') as f:
            data=json.load(f)

        #filenames = [caps.split('\t')[0].split('#')[0] for caps in data]
        filenames  = sorted([d["file_name"].split('.')[0] for d in data['images']])
        #captions = [caps.replace('\n', '').split('\t')[1] for caps in data]
        #features, captions = validation_data[:, 0], validation_data[:, 1]
        features = np.array([feat.astype(float) for feat in features])
        model = Caption_Generator(config)
        generated_captions = model.batch_decoder(filenames, features)
