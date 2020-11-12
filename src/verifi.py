from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
sys.path.append('../tool')
import toolkits
import utils as ut
from progressbar import ProgressBar
pbar = ProgressBar()

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

global args
args = parser.parse_args()

def addPath(x):
  return "/content/VoveDataset/public-test/"+ x

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    # ==================================
    #       Get Train/Val.
    # ==================================
    print('==> calculating test({}) data lists...'.format(args.test_type))

    
    publicTest = pd.read_csv("/content/VoveDataset/public-test.csv")
    
    list1 = addPath(np.array(publicTest["audio_1"]))
    list2 = addPath(np.array(publicTest["audio_2"]))

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)
    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            result_path = "/content/VGG-Speaker-Recognition/result"
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

    print('==> start testing.')

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    total_length = len(unique_list)
    feats, scores, labels = [], [], []
    for c, ID in enumerate(pbar(unique_list)):
        specs = ut.load_data(ID, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    
        v = network_eval.predict(specs)
        feats += [v]
    
    feats = np.array(feats)
    np.save("/content/feats.npy",feats)

if __name__ == "__main__":
    main()
