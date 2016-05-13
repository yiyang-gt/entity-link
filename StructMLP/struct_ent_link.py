import sys, logging, cPickle

import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T

from data_proc import WordVecs
from eval import *
from neural_net_classes import *

logger = logging.getLogger("struct.entity.linking")

def train(datasets, n_ent=11, n_feat=40, n_hidden=30):

    ####################################
    #           MODEL
    ####################################

    # Random Seed
    rng = np.random.RandomState(3435)        

    index = T.lscalar()
    x = T.ftensor3('x')
    y = T.ivector('y') 
    maxl = T.ivector('maxl')
    prev = T.ivector('prev')

    classifier = StructMLP(rng, x, n_feat, n_hidden)
    params = classifier.params
    cost = classifier.get_cost(y, maxl, prev)
    grad_updates = vanilla_sgd(params, cost, 0.01)

    ####################################
    #           PREPARE DATA
    ####################################

    train_x, train_y, train_maxl, train_prev, _ = make_data(datasets[0], n_ent, n_feat)
    test1_x, test1_y, test1_maxl, test1_prev, test1_meta = make_data(datasets[1], n_ent, n_feat)
    test2_x, test2_y, test2_maxl, test2_prev, test2_meta = make_data(datasets[2], n_ent, n_feat)
    test3_x, test3_y, test3_maxl, test3_prev, test3_meta = make_data(datasets[3], n_ent, n_feat)
    
    # RESHAPE TRAIN DATA AS A SINGLE NUMPY ARRAY
    # Start and end indices
    lens = np.array([len(tr) for tr in train_x]).astype('int32')
    st   = np.cumsum(np.concatenate((np.zeros((1, )), lens[:-1]), 0)).astype('int32')
    ed   = (st + lens).astype('int32')
    train_set_x, train_set_y, train_set_maxl, train_set_prev = np.concatenate(train_x), np.concatenate(train_y), np.concatenate(train_maxl), np.concatenate(train_prev)
    
    train_set_x = theano.shared(np.asarray(train_set_x, dtype=theano.config.floatX), borrow=True) 
    train_set_y = theano.shared(np.asarray(train_set_y, dtype='int32'), borrow=True)
    train_set_maxl = theano.shared(np.asarray(train_set_maxl, dtype='int32'), borrow=True)
    train_set_prev = theano.shared(np.asarray(train_set_prev, dtype='int32'), borrow=True)
    st = theano.shared(st, borrow=True)
    ed = theano.shared(ed, borrow=True)

    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[st[index]:ed[index]],
            y: train_set_y[st[index]:ed[index]],
            maxl: train_set_maxl[st[index]:ed[index]],
            prev: train_set_prev[st[index]:ed[index]]}, 
          on_unused_input="ignore",
          profile=False)

    test_y_pred = classifier.predict(x, maxl, prev)
    test_model = theano.function([x, maxl, prev], test_y_pred, on_unused_input="ignore")

    ####################################
    #           EPOCH LOOP
    ####################################

    gold1 = readfile("../../data/label-test.tsv")
    gold2 = readfile("../../data/label-tacl.tsv")
    gold3 = readirfile("../../data/label-ir.tsv")
    n_iter = 1000
    for epoch in np.arange(n_iter):
        # Training Epoch                         
        p_train = 0 
        for index in np.arange(len(train_x)).astype('int32'): 
            up_f = train_model(index)
            p_train += up_f
            #if not (index % 1):
            #    sys.stdout.write("\rTraining %d/%d" % (index+1, len(train_x)))
            #    sys.stdout.flush()

        # Evaluation
        if (epoch+1) % 5 != 0: continue
        logger.info("Epoch %2d/%2d: Cost %2.2f" % (epoch+1, n_iter, p_train)),
        pred = {}
        for i, x, y, maxl, prev, meta in zip(np.arange(len(test1_x)), test1_x, test1_y, test1_maxl, test1_prev, test1_meta):
            tid, ment_meta = meta # meta: [[(mention, sidx, eidx), [wikient]]]
            test_y_pred = test_model(x, maxl, prev)
            for j, sense in enumerate(test_y_pred):
                if sense > 0:
                    if tid not in pred:
                        pred[tid] = []
                    triple = (ment_meta[j][1][sense], ment_meta[j][0][1], ment_meta[j][0][2])
                    pred[tid].append(triple)
        prf = get_results_ass(gold1, pred)
        print "Epoch: %d NEEL Precision: %.4f Recall: %.4f F1: %.4f" %(epoch+1, prf[0], prf[1], prf[2])
        pred = {}
        for i, x, y, maxl, prev, meta in zip(np.arange(len(test2_x)), test2_x, test2_y, test2_maxl, test2_prev, test2_meta):
            tid, ment_meta = meta # meta: [[(mention, sidx, eidx), [wikient]]]
            test_y_pred = test_model(x, maxl, prev)
            for j, sense in enumerate(test_y_pred):
                if sense > 0:
                    if tid not in pred:
                        pred[tid] = []
                    triple = (ment_meta[j][1][sense], ment_meta[j][0][1], ment_meta[j][0][2])
                    pred[tid].append(triple)
        prf = get_results_ass(gold2, pred)
        print "TACL Precision: %.4f Recall: %.4f F1: %.4f" %(prf[0], prf[1], prf[2])
        pred = {}
        for i, x, y, maxl, prev, meta in zip(np.arange(len(test3_x)), test3_x, test3_y, test3_maxl, test3_prev, test3_meta):
            tid, ment_meta = meta # meta: [[(mention, sidx, eidx), [wikient]]]
            if tid not in pred: pred[tid] = set()
            test_y_pred = test_model(x, maxl, prev)
            for j, sense in enumerate(test_y_pred):
                if sense > 0:
                    pred[tid].add(ment_meta[j][1][sense])
        prf = get_ir_prf(gold3, pred)
        print "IR Precision: %.4f Recall: %.4f F1: %.4f" %(prf[0], prf[1], prf[2])

def vanilla_sgd(params, cost, learning_rate):
    gparams = T.grad(cost, params)
    updates = [(param, param - learning_rate * gp) for param,gp in zip(params, gparams)]
    return updates

def make_data(dataset, n_ent=11, n_feat=40):
    data_x, data_y, data_maxl, data_prev, data_meta = [], [], [], [], []
    for tupair in dataset: # tweet: [[(mention, sidx, eidx), [(NIL, NIL, feats), (wikient, fbent, feats),...], sense]]
        loc_x, loc_y, loc_maxl, loc_prev, loc_meta = [], [], [], get_prev(dataset[tupair]), []
        for mention in dataset[tupair]: # mention
            inner_x = [mention[1][i][2][:n_feat] for i in xrange(min(n_ent, len(mention[1])))]
            loc_maxl.append(len(inner_x))
            while len(inner_x) < n_ent: inner_x.append(np.zeros((n_feat,), dtype=theano.config.floatX))
            inner_meta = [ent[0] for ent in mention[1]] # wiki id
            loc_x.append(inner_x)
            inner_y = mention[2]
            if inner_y >= n_ent:
                inner_y = 0
            loc_y.append(inner_y)
            loc_meta.append([mention[0], inner_meta])
        data_x.append(np.asarray(loc_x, dtype=theano.config.floatX))
        data_y.append(np.asarray(loc_y, dtype='int32'))
        data_maxl.append(np.asarray(loc_maxl, dtype='int32'))
        data_prev.append(np.asarray(loc_prev, dtype='int32'))
        data_meta.append((tupair[0], loc_meta))
    return data_x, data_y, data_maxl, data_prev, data_meta

def get_prev(mentions):
    res = []
    for i in xrange(len(mentions)):
        sidx, prev = mentions[i][0][1], -1
        for j in xrange(i-1,-1,-1):
            eidx = mentions[j][0][2]
            if eidx <= sidx:
                prev = j
                break
        res.append(prev)
    return res


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    datafile = sys.argv[1]
    with open(datafile, "rb") as f:
        datasets, vocab, ent_vocab, user_vocab, max_l = cPickle.load(f)

    train(datasets, n_ent=21, n_feat=37, n_hidden=40)

    logger.info('end logging')
