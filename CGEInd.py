from theano import sparse
import theano.tensor as T
import lasagne
import lasagne as lg
import lasagne.layers as lgl
import lasagne.objectives as lgo
import layers
import theano
import numpy as np
import random
from collections import defaultdict as dd
import cPickle
from utils import compAcc, getMaxAcc, subPath, samplePathInd


class CGEIndModel:
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
        num_pairs = (self.path_length + 1 - 2 * self.window_size)\
            * (2 * self.window_size + 1)\
            + ((self.window_size + 1 + 2 * self.window_size) * self.window_size)
        self.subpath_num = num_pairs * self.unsup_batch_size
        self.lstm_hidden_units = 128
        self.bilevel = 10
        self.max_test_acc = 0

        print('Initialize parameters.')

    def loadDataset(self):
        print('Loading {}'.format(self.dataset))
        NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'graph']
        OBJECTS = []
        for i in range(len(NAMES)):
            if self.dataset[0] == 'n':
                OBJECTS.append(cPickle.load(open("../nell_data/ind.{}.{}"
                               .format(self.dataset, NAMES[i]))))
            else:
                OBJECTS.append(cPickle.load(open("data/ind.{}.{}"
                               .format(self.dataset, NAMES[i]))))
        self.x, self.y, self.tx, self.ty, self.allx, self.graph = tuple(OBJECTS)
        self.num_ver = self.allx.shape[0]
        self.mid = int(self.tx.shape[0] * self.valida_rate)
        print("Load dataset {} successfully.".format(self.dataset))

    def buildModel(self):
        print('Building...')
        x_init = sparse.csr_matrix('x', dtype='float32')
        y_init = T.imatrix('y')
        gx_init = sparse.csr_matrix('gx', dtype='float32')
        gy_init = T.ivector('gy')
        gz_init = T.vector('gz')
        mask_init = T.fmatrix('subMask')

        # step train
        x_input = lgl.InputLayer(shape=(None, self.x.shape[1]),
                                 input_var=x_init)
        x_to_label = layers.SparseLayer(x_input, self.y.shape[1],
                                        nonlinearity=lg.nonlinearities.softmax)
        x_to_emd = layers.SparseLayer(x_input, self.embedding_size)
        W = x_to_emd.W
        x_to_emd = layers.DenseLayer(x_to_emd, self.y.shape[1],
                                     nonlinearity=lg.nonlinearities.softmax)
        x_concat = lgl.ConcatLayer([x_to_label, x_to_emd], axis=1)
        x_concat = layers.DenseLayer(x_concat, self.y.shape[1],
                                     nonlinearity=lg.nonlinearities.softmax)
        pred = lgl.get_output(x_concat)
        step_loss = lgo.categorical_crossentropy(pred, y_init).mean()
        hid_loss = lgl.get_output(x_to_label)
        step_loss += lgo.categorical_crossentropy(hid_loss, y_init).mean()
        emd_loss = lgl.get_output(x_to_emd)
        step_loss += lgo.categorical_crossentropy(emd_loss, y_init).mean()
        step_params = lgl.get_all_params(x_concat)
        step_updates = lg.updates.sgd(step_loss, step_params,
                                      learning_rate=self.step_learning_rate)
        self.step_train = theano.function([x_init, y_init], step_loss,
                                          updates=step_updates)
        self.test_fn = theano.function([x_init], pred)

        # supervised train
        gx_input = lgl.InputLayer(shape=(None, self.x.shape[1]),
                                  input_var=gx_init)
        gx_to_emd = layers.SparseLayer(gx_input, self.embedding_size, W=W)
        gx_to_emd = lgl.DenseLayer(gx_to_emd, self.num_ver,
                                   nonlinearity=lg.nonlinearities.softmax)
        gx_pred = lgl.get_output(gx_to_emd)
        g_loss = lgo.categorical_crossentropy(gx_pred, gy_init).sum()
        sup_params = lgl.get_all_params(gx_to_emd)
        sup_updates = lg.updates.sgd(g_loss, sup_params,
                                     learning_rate=self.sup_learning_rate)
        self.sup_train = theano.function([gx_init, gy_init, gz_init], g_loss,
                                         updates=sup_updates,
                                         on_unused_input='ignore')

        # handle lstm input
        cross_entropy = lgo.categorical_crossentropy(gx_pred, gy_init)
        cross_entropy = T.reshape(cross_entropy, (1, self.subpath_num), ndim=None)
        mask_input = lgl.InputLayer(shape=(None, self.window_size + 1),
                                    input_var=mask_init)
        sub_path_batch1 = sparse.csr_matrix('x', dtype='float32')
        sub_path_input1 = lgl.InputLayer(shape=(None, self.x.shape[1]),
                                         input_var=sub_path_batch1)
        sub_path_batch2 = sparse.csr_matrix('x', dtype='float32')
        sub_path_input2 = lgl.InputLayer(shape=(None, self.x.shape[1]),
                                         input_var=sub_path_batch2)
        sub_path_batch3 = sparse.csr_matrix('x', dtype='float32')
        sub_path_input3 = lgl.InputLayer(shape=(None, self.x.shape[1]),
                                         input_var=sub_path_batch3)
        sub_path_batch4 = sparse.csr_matrix('x', dtype='float32')
        sub_path_input4 = lgl.InputLayer(shape=(None, self.x.shape[1]),
                                         input_var=sub_path_batch4)
        sub_path_emd1 = layers.SparseLayer(sub_path_input1, self.embedding_size,
                                           W=W)
        sub_path_emd1 = T.reshape(lgl.get_output(sub_path_emd1),
                                  (self.subpath_num, 1, self.embedding_size))
        sub_path_emd2 = layers.SparseLayer(sub_path_input2,
                                           self.embedding_size, W=W)
        sub_path_emd2 = T.reshape(lgl.get_output(sub_path_emd2),
                                  (self.subpath_num, 1, self.embedding_size))
        sub_path_emd3 = layers.SparseLayer(sub_path_input3, self.embedding_size,
                                           W=W)
        sub_path_emd3 = T.reshape(lgl.get_output(sub_path_emd3),
                                  (self.subpath_num, 1, self.embedding_size))
        sub_path_emd4 = layers.SparseLayer(sub_path_input4, self.embedding_size,
                                           W=W)
        sub_path_emd4 = T.reshape(lgl.get_output(sub_path_emd4),
                                  (self.subpath_num, 1, self.embedding_size))
        sub_path_concat = T.concatenate([sub_path_emd1, sub_path_emd2,
                                         sub_path_emd3, sub_path_emd4], axis=1)
        sub_path_concat_layer = lgl.InputLayer(shape=(None, self.window_size + 1,
                                                      self.embedding_size),
                                               input_var=sub_path_concat)

        # lstm layer
        lstm_layer = lgl.LSTMLayer(sub_path_concat_layer,
                                   self.lstm_hidden_units,
                                   grad_clipping=3,
                                   mask_input=mask_input)

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
        lstm_params = lgl.get_all_params(lstm_layer, trainable=True)
        lstm_updates = lg.updates.sgd(reweight_loss, lstm_params,
                                      learning_rate=0.01)
        self.lstm_fn = theano.function([gx_init, gy_init, gz_init,
                                        sub_path_batch1, sub_path_batch2,
                                        sub_path_batch3, sub_path_batch4,
                                        mask_init],
                                       reweight_loss,
                                       updates=lstm_updates,
                                       on_unused_input='ignore')
        alpha_updates = lg.updates.sgd(reweight_loss, sup_params,
                                       learning_rate=0.001)
        self.alpha_fn = theano.function([gx_init, gy_init, gz_init,
                                         sub_path_batch1, sub_path_batch2,
                                         sub_path_batch3, sub_path_batch4,
                                         mask_init],
                                        reweight_loss,
                                        updates=alpha_updates,
                                        on_unused_input='ignore')

        print('Build done.')

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
            g, gy = [], []
            for _ in range(self.super_pairs):
                x1 = random.randint(0, self.x.shape[0] - 1)
                label = labels[x1]
                if len(label2inst) == 1:
                    continue
                x2 = random.choice(label2inst[label])
                g.append([x1, x2])
                gy.append(1.0)
            g = np.array(g, dtype=np.int32)
            yield self.allx[g[:, 0]], g[:, 1], gy

    def forUnsupervised(self):
        while True:
            ind = np.random.permutation(self.num_ver)
            i = 0
            while i < ind.shape[0]:
                g, gy = [], []
                sub_paths = []
                mask = []
                j = min(ind.shape[0], i + self.unsup_batch_size)
                if j == ind.shape[0]:
                    i = j - self.unsup_batch_size
                for k in ind[i: j]:
                    if len(self.graph[k]) == 0:
                        continue
                    path = samplePathInd(k, self.path_length, self.graph,
                                         self.num_ver)
                    for l in range(len(path)):
                        if path[l] >= self.allx.shape[0]:
                            continue
                        for m in range(l - self.window_size,
                                       l + self.window_size + 1):
                            if m < 0 or m >= len(path):
                                continue
                            if path[m] >= self.allx.shape[0]:
                                continue
                            g.append([path[l], path[m]])
                            s, m = subPath(l, m, path, self.window_size)
                            sub_paths.append(s)
                            mask.append(m)
                            gy.append(1.0)
                g = np.array(g, dtype=np.int32)
                yield \
                    self.allx[g[:, 0]], g[:, 1], \
                    gy, np.array(sub_paths, dtype=np.int32), \
                    np.array(mask, dtype=np.float32)
                i = j

    def forStepTrain(self):
        while True:
            ind = np.array(np.random.permutation(self.x.shape[0]),
                           dtype=np.int32)
            i = 0
            while i < self.x.shape[0]:
                j = min(ind.shape[0], i + self.sup_batch_size)
                yield self.x[ind[i: j]], self.y[ind[i: j]]
                i = j

    def test(self):
        x, y = next(self.forStepTrain())
        self.step_train(x, y)
        pred_vali = self.test_fn(self.tx[0:self.mid])
        acc_vali = compAcc(pred_vali, self.ty[0:self.mid])
        pred_test = self.test_fn(self.tx[self.mid:self.tx.shape[0]])
        acc_test = compAcc(pred_test, self.ty[self.mid:self.tx.shape[0]])
        self.max_test_acc = getMaxAcc(self.max_test_acc, acc_test)
        print('Validate accuracy {:.3f}, Max test accuracy {:.3f}'
              .format(acc_vali, self.max_test_acc))

    def supervisedTrain(self, sup_iter):
        for i in range(sup_iter):
            gx, gy, gz = next(self.forSupervised())
            loss = self.sup_train(gx, gy, gz)
            print('Supervised training: iter {}, loss {:.4f}'.format(i, loss))

    def unsupervisedTrain(self, unsup_iter):  # naming bi-level is better?
        for i in range(unsup_iter):
            gx, gy, gz, subPath, mask = next(self.forUnsupervised())
            if i % self.bilevel == 0:
                loss = self.lstm_fn(gx, gy, gz,
                                    self.allx[subPath[:, 0]],
                                    self.allx[subPath[:, 1]],
                                    self.allx[subPath[:, 2]],
                                    self.allx[subPath[:, 3]], mask)
            else:
                loss = self.alpha_fn(gx, gy, gz,
                                     self.allx[subPath[:, 0]],
                                     self.allx[subPath[:, 1]],
                                     self.allx[subPath[:, 2]],
                                     self.allx[subPath[:, 3]], mask)
            print('Unsupervised training: iter {}, loss {:.4f}'.format(i, loss))
            self.test()
