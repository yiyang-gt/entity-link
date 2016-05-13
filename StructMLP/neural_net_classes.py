import numpy as np
import theano
import theano.tensor as T

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.activation = activation

        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)
        # parameters of the model
        self.params = [self.W, self.b]

    def get_output(self, inp):
        lin_output = T.dot(inp, self.W) + self.b
        return self.activation(lin_output)

class StructMLP(object):

    def __init__(self, rng, input, n_in, n_hidden, activation=T.tanh):
        """
        input: (n_mention, n_entity, n_feature)
        """
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=activation)
        self.scoringLayer = LinearScore(inp=self.hiddenLayer.output, n=n_hidden) # (n_mention, n_entity)
        self.scores = self.scoringLayer.scores
        self.params = self.hiddenLayer.params + self.scoringLayer.params

    def predict(self, inp, ment_lens, prevs):
        hidden_output = self.hiddenLayer.get_output(inp)
        scoring_output = self.scoringLayer.predict(hidden_output)
        #local_argmax, _ = theano.scan(fn=lambda scrs, mlen: T.argmax(T.set_subtensor(scrs[:mlen][0], scrs[:mlen][0]-0.)), outputs_info=None, sequences=[scoring_output, ment_lens])
        n_ment, n_ent = scoring_output.shape
        sscrs_tab, paths_tab = T.zeros((n_ment, n_ent), dtype=theano.config.floatX), T.zeros((n_ment, n_ent, n_ment), dtype='int32')
        result, _ = theano.scan(fn=loss_aug_decoding, outputs_info=[sscrs_tab, paths_tab], non_sequences=[scoring_output,prevs,ment_lens], sequences=[T.cast(T.arange(n_ment), 'int32')])
        local_argmax = result[1][-1][-1][T.argmax(result[0][-1][-1])]
        return local_argmax

    def get_cost(self, y, ment_lens, prevs):
        local_gold, _ = theano.scan(fn=lambda scrs, loc_y: scrs[loc_y], outputs_info=None, sequences=[self.scores, y])
        loss_aug_scores, _ = theano.scan(fn=lambda scrs, loc_y: scrs + .1 * T.set_subtensor(T.ones_like(scrs)[loc_y], 0.), outputs_info=None, sequences=[self.scores, y])
        #local_max, _ = theano.scan(fn=lambda scrs, mlen: T.max(scrs[:mlen]), outputs_info=None, sequences=[loss_aug_scores, ment_lens])
        n_ment, n_ent = self.scores.shape
        sscrs_tab, paths_tab = T.zeros((n_ment, n_ent), dtype=theano.config.floatX), T.zeros((n_ment, n_ent, n_ment), dtype='int32')
        result, _ = theano.scan(fn=loss_aug_decoding, outputs_info=[sscrs_tab, paths_tab], non_sequences=[loss_aug_scores,prevs,ment_lens], sequences=[T.cast(T.arange(n_ment), 'int32')])
        local_argmax = result[1][-1][-1][T.argmax(result[0][-1][-1])]
        local_max, _ = theano.scan(fn=lambda scrs, loc_y: scrs[loc_y], outputs_info=None, sequences=[loss_aug_scores, local_argmax])
        return T.max([0, T.mean(local_max - local_gold)])

def loss_aug_decoding(step, prior_sscrs, prior_paths, scrs, prevs, mlens):
    """
    prior_sscrs: (n_mention, n_ent) storing best scores
    prior_paths: (n_mention, n_ent, n_mention+1) storing best paths
    scrs: (n_mention, n_ent)
    prevs: n_mention
    mlens: n_mention
    """
    s, p, ml = scrs[step], prevs[step], mlens[step]
    # NIL entity
    out_sscrs = T.inc_subtensor(prior_sscrs[step, 0], s[0]+T.max(prior_sscrs[step-1]))
    out_paths = T.set_subtensor(prior_paths[step, 0, :], prior_paths[step-1, T.argmax(prior_sscrs[step-1]), :])
    # other entities
    out_sscrs = T.inc_subtensor(out_sscrs[step, 1:ml], s[1:ml]+T.sum(scrs[p+1:step, 0])) # NIL scores of overlapped mentions
    out_sscrs = T.inc_subtensor(out_sscrs[step, 1:ml], T.max(prior_sscrs[p]))
    prev_max_path = prior_paths[p, T.argmax(prior_sscrs[p])]
    result, _ = theano.scan(fn=lambda inner_step, prior_result: T.set_subtensor(prior_result[step, inner_step], prev_max_path), outputs_info=out_paths, sequences=[T.cast(T.arange(ml-1)+1, 'int32')] )
    out_paths = result[-1]
    out_paths = T.set_subtensor(out_paths[step, :ml, step], T.cast(T.arange(ml),'int32'))
    return [out_sscrs, out_paths]


class LinearScore(object):
    
    def __init__(self, inp, n):
        self.w = theano.shared(value=np.zeros(n, dtype=theano.config.floatX), name='w')
        self.b = theano.shared(value=np.cast[theano.config.floatX](0.), name='b')
        self.scores = T.dot(inp, self.w) + self.b
        self.l2_norm = T.sum(self.w ** 2)
        self.params = [self.w, self.b]

    def predict(self, inp):
        scores = T.dot(inp, self.w) + self.b
        return scores

class BilinearScore(object):

    def __init__(self, linp, rinp, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W')
        self.b = theano.shared(value=np.cast[theano.config.floatX](0.), name='b')
        self.scores = T.batched_tensordot(T.dot(linp, self.W), rinp, [[1],[2]]) + self.b
        self.l2_norm = T.sum(self.W ** 2)
        self.params = [self.W, self.b]

    def predict(self, linp, rinp):
        scores = T.batched_tensordot(T.dot(linp, self.W), rinp, [[1],[2]]) + self.b
        return scores

