import sys, cPickle, logging
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter

from twokenize import tokenize, tokenizeRawTweetText, normalizeTextForSentiment, normalizeTextForNLSE

logger = logging.getLogger("data.proc")

def build_data(trnfname, tstfnames, tfname, wffname, label_idx=0, mid_idx=1, ent_idx=2, fstart_idx=5):
    """
    Loads and process data.
    trnfname, tstfname: train/test bing files
    tfname: tid to tweet
    wffname: wiki entity to freebase entity mappings
    """
    tid_tweet_map, tid_uid_map, wiki_fb_map = {}, {}, {}
    with open(tfname, "rb") as f:
        for line in f:
            tid, uid, tweet = line.strip().split("\t")
            tid_tweet_map[tid] = tweet
            tid_uid_map[tid] = uid
    with open(wffname, "rb") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            wiki_fb_map[parts[0]] = parts[2]

    label_dict = {"Good":1, "Bad":0}

    datasets, max_l = [], 0
    # data structures {(tid, uid), [[(mention, sidx, eidx), [(NIL, NIL, feats), (wikient, fbent, feats),...], sense], ...]}
    vocab, fb_vocab, user_vocab = set(), set(), set()
    for i, fname in enumerate([trnfname] + tstfnames):
        split = i + 1
        ment_cnt, ent_cnt = 0, 0
        dataset = OrderedDict()
        with open(fname, "rb") as f:
            f.readline() # skip header
            for line in f:  
                parts = line.strip().split("\t")
                label, mid, wikient = label_dict[parts[label_idx]], parts[mid_idx], parts[ent_idx]
                tid, ment_idx, sidx, eidx = mid.split("-")
                sidx, eidx = int(sidx), int(eidx)
                
                if (tid, uid) not in dataset: # new tweet
                    tweet = tid_tweet_map[tid]
                    uid = tid_uid_map[tid]
                    if uid != "unknown":
                        user_vocab.add(uid)
                    dataset[(tid, uid)] = []

                if wikient == "_NULL_": # new mention
                    #mention = tweet[sidx:eidx]
                    #mention = unicode(mention.strip(), errors='replace')
                    #mention = normalizeTextForNLSE(tokenizeRawTweetText(mention), True).lower()
                    mention = parts[ent_idx+1].lower()
                    max_l = max(max_l, len(mention.split()))
                    words = set(mention.split()) # tokens in mention
                    vocab.update(words)
                    dataset[(tid, uid)].append([(mention, sidx, eidx), [], 0]) # add mention information
                    ment_cnt += 1
                
                fbent = "NULL"
                if wikient in wiki_fb_map: 
                    fbent = wiki_fb_map[wikient]
                    fb_vocab.add(fbent)

                feats = map(float, parts[fstart_idx:])
                feats = np.asarray(feats, dtype="float32")
                feats[-1] = 1. - feats[-1] # NIL -> 1, others -> 0

                if label == 1: dataset[(tid, uid)][-1][-1] = len(dataset[(tid, uid)][-1][1])

                dataset[(tid, uid)][-1][1].append((wikient, fbent, feats))
                ent_cnt += 1
        datasets.append(dataset)
        logger.info("processed dataset %d with %d mention candidates and %d entity candidates " %(split, ment_cnt, ent_cnt))

    logger.info("vocab size: %d, fb_vocab size: %d, user_vocab size: %d, max mention length: %d" %(len(vocab), len(fb_vocab), len(user_vocab), max_l))
    return datasets, vocab, fb_vocab, user_vocab, max_l

class WordVecs(object):
    """
    precompute embeddings for word/feature/tweet etc.
    """
    def __init__(self, fname, vocab, binary=1):
        if binary == 1:
            word_vecs, self.k = self.load_bin_vec(fname, vocab)
        else:
            word_vecs, self.k = self.load_txt_vec(fname, vocab)
        self.add_unknown_words(word_vecs, vocab, k=self.k)
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))            
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs, layer1_size
    
    def load_txt_vec(self, fname, vocab):
        """
        Loads 50x1 word vecs from sentiment word embeddings (Tang et al., 2014)
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs, layer1_size

    def add_unknown_words(self, word_vecs, vocab, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs:
                #print word
                word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
                #word_vecs[word] = np.zeros(k)  
    


if __name__=="__main__":    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')
    
    trnfname, tstfnames, tfname, wefname, eefname, wffname, ofname = sys.argv[1], sys.argv[2:-5], sys.argv[-5], sys.argv[-4], sys.argv[-3], sys.argv[-2], sys.argv[-1]
    datasets, vocab, fb_vocab, user_vocab, max_l = build_data(trnfname, tstfnames, tfname, wffname)

    # data structures {(tid, uid), [[(mention, sidx, eidx), [(NIL, NIL, feats), (wikient, fbent, feats),...], sense], ...]}

    logger.info("loading and processing pretrained word vectors")
    wordvecs = WordVecs(wefname, vocab, binary=0)
    logger.info("loading and processing pretrained entity vectors")
    entvecs = WordVecs(eefname, fb_vocab, binary=1)
    entvecs.word_idx_map["NULL"] = 0

    cPickle.dump([datasets, wordvecs, entvecs, user_vocab, max_l], open(ofname, "wb"))
    #cPickle.dump([datasets, vocab, fb_vocab, user_vocab, max_l], open(ofname, "wb"))
    logger.info("dataset created!")
    logger.info("end logging")
    

