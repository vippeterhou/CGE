from theano import sparse
import theano.tensor as T
import lasagne as lg
import lasagne.layers as lgl
import lasagne.objectives as lgo
import layers
import theano
import numpy as np
import random
from collections import defaultdict as dd
import cPickle
from utils import compAcc, getMaxAcc, subPath


class CGETransModel:
    def __init__(self, args):
        self.dataset = args.dataset
        self.embedding_size = args.embedding_size
        self.super_pairs = args.super_pairs
        self.step_learning_rate = args.step_learning_rate
        self.sup_learning_rate = args.sup_learning_rate
        self.valida_rate = args.valida_rate
        self.sup_batch_size = args.sup_batch_size
        self.unsup_batch_size = args.unsup_batch_size
        self.path_length = args.path_length
        self.window_size = args.window_size
        num_pairs = (self.path_length+1-2*self.window_size) * \
                    (2*self.window_size+1) + \
                    ((self.window_size+1+2*self.window_size)*self.window_size)
        self.subpath_num = num_pairs*self.unsup_batch_size
        self.lstm_hidden_units = 128
        self.bilevel = 10
        self.max_test_acc = 0

        print('Initialize parameters.')

    def loadDataset(self):
        print('Loading {}'.format(self.dataset))
        NAMES = ['x', 'y', 'tx', 'ty', 'graph']
        OBJECTS = []
        for i in range(len(NAMES)):
            if self.dataset[0] == 'n':
                OBJECTS.append(cPickle.load(open("../nell_data/ind.{}.{}"
                               .format(self.dataset, NAMES[i]))))
            else:
                OBJECTS.append(cPickle.load(open("data/trans.{}.{}"
                               .format(self.dataset, NAMES[i]))))
        self.x, self.y, self.tx, self.ty, self.graph = tuple(OBJECTS)
        self.num_ver = len(self.graph)
        self.mid = int(self.tx.shape[0] * self.valida_rate)
        print("Load dataset {} successfully.".format(self.dataset))

    def buildModel(self):
        print('Building...')
        x_init = sparse.csr_matrix('x', dtype='float32')
        y_init = T.imatrix('y')
        g_init = T.imatrix('g')
        ind_init = T.ivector('ind')
        sub_path_init = T.imatrix('subPathsBatch')
        mask_init = T.fmatrix('subMask')

        # step train
        x_input = lgl.InputLayer(shape=(None, self.x.shape[1]),
                                 input_var=x_init)
        g_input = lgl.InputLayer(shape=(None, 2), input_var=g_init)
        ind_input = lgl.InputLayer(shape=(None, ), input_var=ind_init)
        pair_second = lgl.SliceLayer(g_input, indices=1, axis=1)
        pair_first = lgl.SliceLayer(g_input, indices=0, axis=1)
        pair_first_emd = lgl.EmbeddingLayer(pair_first,
                                            input_size=self.num_ver,
                                            output_size=self.embedding_size)
        emd_to_numver = layers.DenseLayer(pair_first_emd, self.num_ver,
                                          nonlinearity=lg.nonlinearities.softmax)
        index_emd = lgl.EmbeddingLayer(ind_input,
                                       input_size=self.num_ver,
                                       output_size=self.embedding_size,
                                       W=pair_first_emd.W)
        x_to_ydim = layers.SparseLayer(x_input, self.y.shape[1],
                                       nonlinearity=lg.nonlinearities.softmax)
        index_emd = layers.DenseLayer(index_emd, self.y.shape[1],
                                      nonlinearity=lg.nonlinearities.softmax)
        concat_two = lgl.ConcatLayer([x_to_ydim, index_emd], axis=1)
        concat_two = layers.DenseLayer(concat_two, self.y.shape[1],
                                       nonlinearity=lg.nonlinearities.softmax)
        concat_two_output = lgl.get_output(concat_two)
        step_loss = lgo.categorical_crossentropy(concat_two_output,
                                                 y_init).mean()
        hid_loss = lgl.get_output(x_to_ydim)
        step_loss += lgo.categorical_crossentropy(hid_loss, y_init).mean()
        emd_loss = lgl.get_output(index_emd)
        step_loss += lgo.categorical_crossentropy(emd_loss, y_init).mean()
        step_params = [index_emd.W, index_emd.b, x_to_ydim.W,
                       x_to_ydim.b, concat_two.W, concat_two.b]
        step_updates = lg.updates.sgd(step_loss, step_params,
                                      learning_rate=self.step_learning_rate)
        self.step_train = theano.function([x_init, y_init, ind_init], step_loss,
                                          updates=step_updates,
                                          on_unused_input='ignore')
        self.test_fn = theano.function([x_init, ind_init], concat_two_output,
                                       on_unused_input='ignore')

        # supervised train
        fc_output = lgl.get_output(emd_to_numver)
        pair_second_output = lgl.get_output(pair_second)
        sup_loss = lgo.categorical_crossentropy(fc_output,
                                                pair_second_output).sum()
        sup_params = lgl.get_all_params(emd_to_numver, trainable=True)
        sup_updates = lg.updates.sgd(sup_loss, sup_params,
                                     learning_rate=self.sup_learning_rate)
        self.sup_train = theano.function([g_init], sup_loss,
                                         updates=sup_updates,
                                         on_unused_input='ignore')

        cross_entropy = lgo.categorical_crossentropy(fc_output,
                                                     pair_second_output)
        cross_entropy = T.reshape(cross_entropy,
                                  (1, self.unsup_batch_size),
                                  ndim=None)

        mask_input = lgl.InputLayer(shape=(None, self.window_size+1),
                                    input_var=mask_init)
        subPath_in = lgl.InputLayer(shape=(None, self.window_size+1),
                                    input_var=sub_path_init)
        sub_path_emd = lgl.EmbeddingLayer(subPath_in, input_size=self.num_ver,
                                          output_size=self.embedding_size,
                                          W=pair_first_emd.W)

        lstm_layer = lgl.LSTMLayer(sub_path_emd, self.lstm_hidden_units,
                                   grad_clipping=3, mask_input=mask_input)

        # handle path weight
        max1 = T.mean(lgl.get_output(lstm_layer), axis=1)
        max2 = T.mean(max1, axis=1)
        max2_init = T.fcol('max2')
        max2_init = T.reshape(max2, ((self.subpath_num, 1)))
        max2_input = lgl.InputLayer(shape=(self.subpath_num, 1),
                                    input_var=max2_init)
        max2_input = lgl.BatchNormLayer(max2_input)
        path_weight = lgl.get_output(max2_input)
        path_weight = lg.nonlinearities.sigmoid(path_weight)
        path_weight = 1 + 0.3 * path_weight

        # unsupervised train
        reweight_loss = T.dot(cross_entropy, path_weight)[0][0]
        lstm_params_all = lgl.get_all_params(lstm_layer, trainable=True)
        lstm_params = list(set(lstm_params_all).difference(set(sup_params)))
        lstm_updates = lg.updates.sgd(reweight_loss, lstm_params,
                                      learning_rate=0.01)
        self.lstm_fn = theano.function(
                                [sub_path_init, g_init, mask_init],
                                reweight_loss,
                                updates=lstm_updates,
                                on_unused_input='ignore')
        alpha_updates = lg.updates.sgd(reweight_loss, sup_params,
                                       learning_rate=0.001)
        self.alpha_fn = theano.function(
                                [sub_path_init, g_init, mask_init],
                                reweight_loss,
                                updates=alpha_updates,
                                on_unused_input='ignore')

    def forSupervised(self):
        labels, label2inst, not_label = [], dd(list), dd(list)
        for i in range(self.x.shape[0]):
            flag = False
            for j in range(self.y.shape[1]):
                if self.y[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.y[i, j] == 0:
                    not_label[j].append(i)
        while True:
            g = []
            for _ in range(self.sup_batch_size):
                x1 = random.randint(0, self.x.shape[0] - 1)
                label = labels[x1]
                if len(label2inst) == 1:
                    continue
                x2 = random.choice(label2inst[label])
                g.append([x1, x2])
            yield np.array(g, dtype=np.int32)

    def forUnsupervised(self):
        while True:
            ind = np.random.permutation(self.num_ver)
            i = 0
            while i < ind.shape[0]:
                g = []
                sub_paths = []
                mask = []
                j = min(ind.shape[0], i + self.unsup_batch_size)
                for k in ind[i: j]:
                    if len(self.graph[k]) == 0:
                        continue
                    path = [k]
                    for _ in range(self.path_length):
                        path.append(random.choice(self.graph[path[-1]]))
                    for l in range(len(path)):
                        for m in range(l - self.window_size,
                                       l + self.window_size + 1):
                            if m < 0 or m >= len(path):
                                continue
                            g.append([path[l], path[m]])
                            s, m = subPath(l, m, path, self.window_size)
                            sub_paths.append(s)
                            mask.append(m)
                yield np.array(g, dtype=np.int32),\
                    np.array(sub_paths, dtype=np.int32),\
                    np.array(mask, dtype=np.float32)
                i = j

    def forStepTrain(self):
        while True:
            ind = np.array(np.random.permutation(self.x.shape[0]),
                           dtype=np.int32)
            i = 0
            while i < self.x.shape[0]:
                j = min(ind.shape[0], i + self.sup_batch_size)
                yield self.x[ind[i: j]], self.y[ind[i: j]], ind[i: j]
                i = j

    def test(self):
        x, y, ind = next(self.forStepTrain())
        self.step_train(x, y, ind)
        pred_vali = self.test_fn(self.tx[0:self.mid],
                                 np.arange(self.x.shape[0],
                                           self.x.shape[0] + self.mid,
                                           dtype=np.int32))
        acc_vali = compAcc(pred_vali, self.ty[0:self.mid])
        pred_test = self.test_fn(self.tx[self.mid:self.tx.shape[0]],
                                 np.arange(self.x.shape[0] + self.mid,
                                           self.x.shape[0] + self.tx.shape[0],
                                           dtype=np.int32))
        acc_test = compAcc(pred_test, self.ty[self.mid:self.tx.shape[0]])
        self.max_test_acc = getMaxAcc(self.max_test_acc, acc_test)
        print('Validate accuracy {:.3f}, Max test accuracy {:.3f}'
              .format(acc_vali, self.max_test_acc))

    def supervisedTrain(self, sup_iter):
        for i in range(sup_iter):
            pairs = next(self.forSupervised())
            loss = self.sup_train(pairs)
            print('Supervised training: iter {}, loss {:.4f}'.format(i, loss))

    def unsupervisedTrain(self, unsup_iter):  # naming bi-level is better?
        for i in range(unsup_iter):
            pairs, sub_path, mask = next(self.forUnsupervised())
            if i % self.bilevel == 0:
                loss = self.lstm_fn(sub_path, pairs, mask)
            else:
                loss = self.alpha_fn(sub_path, pairs, mask)
            print('Unsupervised training: iter {}, loss {:.4f}'.format(i, loss))
            self.test()
