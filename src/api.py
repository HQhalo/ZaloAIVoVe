from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
sys.path.append('../tools')
import toolkits
import utils as ut
from flask import Flask, request, jsonify
import json
import os 
import io
import hashlib
import copy 
from werkzeug.utils import secure_filename
import tensorflow as tf
AUDIO_STORAGE = os.path.join("/content", "audio_storage")
if not os.path.isdir(AUDIO_STORAGE):
    os.makedirs(AUDIO_STORAGE)
global graph
graph = tf.get_default_graph() 

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--data_path', default='/media/weidi/2TB-2/datasets/voxceleb1/wav', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
parser.add_argument('--warmup_file', default='', type=str)
parser.add_argument("--debug", default= True, type=bool)
global args
args = parser.parse_args()
import model
params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }
network_eval  = None
global cacheTable
cacheTable= {} 
def load_model():
    toolkits.initialize_GPU(args)
    global network_eval 
    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

def calcHash(filename):
    with open(filename,"rb") as f:
        bytes = f.read() # read entire file as bytes
        return hashlib.sha256(bytes).hexdigest()
def warmup():
    specs = ut.load_data(args.warmup_file, win_length=params['win_length'], sr=params['sampling_rate'],
                                hop_length=params['hop_length'], n_fft=params['nfft'],
                                spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)

    with graph.as_default():
        v1 = network_eval.predict(specs)
# Flask
app = Flask(__name__)

@app.route("/api/predict", methods=['POST'])
def api_predict():
    """
    Required params:
        audio_1, audio_2
    """
    import timeit

    
    audio_file_1 = request.files['audio_1'] # Required
    audio_file_2 = request.files['audio_2'] # Required
    counttime = 0
    start = timeit.default_timer()
    if audio_file_1:
        filename_1 = os.path.join(AUDIO_STORAGE,secure_filename(audio_file_1.filename))
        audio_file_1.save(filename_1) # Save audio in audio_storage, path: audio_storage/filename_1

    if audio_file_2:
        filename_2 = os.path.join(AUDIO_STORAGE,secure_filename(audio_file_2.filename))
        audio_file_2.save(filename_2) # Save audio in audio_storage, path: audio_storage/filename_2
    stop = timeit.default_timer()
    counttime += stop - start
    print('Time save file: ', stop - start)
    v1 = None
    v2 = None
    # Import code here
    start = timeit.default_timer()
    ch1 = calcHash(filename_1) 
    ch2 = calcHash(filename_2) 
    stop = timeit.default_timer()
    counttime += stop - start
    print('Time hash: ', stop - start)
    start = timeit.default_timer()
    if( ch1 in cacheTable.keys()):
        v1 = cacheTable[ch1]
    else:
        specs = ut.load_data(filename_1, win_length=params['win_length'], sr=params['sampling_rate'],
                                hop_length=params['hop_length'], n_fft=params['nfft'],
                                spec_len=params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)

        with graph.as_default():
            v1 = network_eval.predict(specs)
        cacheTable[ch1] = v1
    
    if( ch2 in cacheTable.keys()):
        v2 = cacheTable[ch2]
    else:
        specs = ut.load_data(filename_2, win_length=params['win_length'], sr=params['sampling_rate'],
                                hop_length=params['hop_length'], n_fft=params['nfft'],
                                spec_len=params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
        with graph.as_default():
            v2 = network_eval.predict(specs)
        cacheTable[ch2] = v2

    label = (np.linalg.norm(v1 - v2) < 0.547742) * 1
    stop = timeit.default_timer()
    counttime += stop - start
    print('Time: ', stop - start)
    print("Time overal ", counttime)
    return jsonify(
        label = label
    )

if __name__ == '__main__':
    load_model()
    warmup()
    app.run(host='0.0.0.0', port='6677', debug=False)
