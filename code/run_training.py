import sys

import numpy as np

from chainer import Chain, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

from sklearn.metrics import roc_auc_score, precision_score, recall_score


class CompoundProteinInteraction(Chain):
    def __init__(self):
        super(CompoundProteinInteraction, self).__init__(
              embed_vertex=L.EmbedID(n_vertex, dim, ignore_label=-1),
              embed_edge=L.EmbedID(n_edge, dim, ignore_label=-1),
              W_vertex=L.Linear(dim, dim),
              W_edge=L.Linear(dim, dim),
              embed_ngram=L.EmbedID(n_word, dim),
              Convo1=L.Convolution2D(1, dim, (window, dim)),
              Convo2=L.Convolution2D(1, dim, (window, dim)),
              Convo3=L.Convolution2D(1, dim, (window, dim)),
              W_attention=L.Linear(dim, dim),
              W_z=L.Linear(2*dim, output_dim)
              )

    def gnn(self, vertex, edge, adjacency, vertex_):

        x_vertex = self.embed_vertex(vertex)
        x_edge = self.embed_edge(edge)
        V, degree = edge.shape

        for _ in range(layer_gnn):

            x_adja = F.embed_id(adjacency, x_vertex, ignore_label=-1)

            h_adja = F.relu(self.W_vertex(F.sum(x_adja, 1)) +
                            self.W_edge(F.sum(x_edge, 1)))

            x_vertex_ = F.embed_id(vertex_, x_vertex, ignore_label=-1)
            x_vertex_ = F.reshape(x_vertex_, (V*degree, dim))
            x_adja = F.reshape(x_adja, (V*degree, dim))
            h_side = F.relu(self.W_vertex(x_vertex_+x_adja))

            """Update x_vertex."""
            x_vertex = F.sigmoid(F.relu(self.W_vertex(x_vertex)) + h_adja)

            """Update x_edge."""
            x_edge = F.reshape(x_edge, (V*degree, dim))
            x_edge = F.sigmoid(F.relu(self.W_edge(x_edge)) + h_side)
            x_edge = F.reshape(x_edge, (V, degree, dim))

        y = F.expand_dims(F.sum(x_vertex, 0), 0)

        return y

    def cnn(self, sequence):
        x = self.embed_ngram(sequence)
        x = F.reshape(x, (1, 1, x.shape[0], dim))
        h_1 = F.relu(self.Convo1(x))
        h_1 = F.reshape(h_1, (1, 1, h_1.shape[2], dim))
        h_2 = F.relu(self.Convo2(h_1))
        h_2 = F.reshape(h_2, (1, 1, h_2.shape[2], dim))
        h_3 = F.relu(self.Convo3(h_2))
        H = F.reshape(h_3, (h_3.shape[2], dim))
        return H

    def attention(self, y, H):
        y = F.tanh(self.W_attention(y))
        H = F.tanh(self.W_attention(H))
        a = F.sigmoid(F.linear(y, H))
        r = F.matmul(a, H)
        return r

    def __call__(self, data, train=True):

        vertex, edge, adjacency, vertex_, sequence, t = data

        y_molecule = self.gnn(vertex, edge, adjacency, vertex_)
        H_protein = self.cnn(sequence)
        y_protein = self.attention(y_molecule, H_protein)
        r = F.concat((y_molecule, y_protein))
        z = self.W_z(r)

        if train:
            loss = F.softmax_cross_entropy(z, t)
            return loss
        else:
            return list(F.softmax(z).data[0]), t[0]


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optimizers.Adam(lr)
        self.optimizer.setup(model)
        self.optimizer.add_hook(optimizer.WeightDecay(weight_decay))

    def train(self, dataset):
        np.random.shuffle(dataset)
        loss_total = 0
        for data in dataset:
            self.model.zerograds()
            loss = self.model(data)
            loss.backward()
            self.optimizer.update()
            loss_total += loss.data
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        z_list, t_list = [], []
        for data in dataset:
            z, t = self.model(data, train=False)
            z_list.append(z)
            t_list.append(t)

        """Evaluation metrics (AUC, precision, and recall)."""
        score_list, label_list = [], []
        for z in z_list:
            score_list.append(z[1])  # Predicted score.
            label_list.append(np.argmax(z))  # Predicted label.
        auc = roc_auc_score(t_list, score_list)
        precision = precision_score(t_list, label_list)
        recall = recall_score(t_list, label_list)

        return auc, precision, recall

    def result(self, epoch, loss, auc_dev, auc_test,
               precision, recall, file_result):
        if (epoch == 1):
            f = open(file_result, 'w')
            f.write('Epoch\tLoss\tAUC_dev\t' +
                              'AUC_test\tPrecision\tRecall\n')
        else:
            f = open(file_result, 'a')
        result = map(str, [epoch, loss, auc_dev, auc_test, precision, recall])
        f.write('\t'.join(result) + '\n')
        f.close()

    def save(self, model, file_name):
        serializers.save_hdf5(file_name, model)


def load_dataset(data, radius, ngram):
    return np.load('../dataset/' + DATASET + '/input/radius' +
                   radius + '_ngram' + ngram + '/' + data + '.npy')


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    (DATASET, radius, ngram, dim, window, layer_gnn, layer_cnn,
     lr, lr_decay, weight_decay, setting) = sys.argv[1:]
    (dim, window, layer_gnn,
     layer_cnn) = map(int, [dim, window, layer_gnn, layer_cnn])
    (lr, lr_decay,
     weight_decay) = map(float, [lr, lr_decay, weight_decay])
    file_result = '../output/result/' + setting + '.txt'
    file_model = '../output/model/' + setting

    """Load data."""
    vertex = load_dataset('vertex', radius, ngram)
    edge = load_dataset('edge', radius, ngram)
    adjacency = load_dataset('adjacency', radius, ngram)
    vertex_ = load_dataset('vertex_', radius, ngram)
    sequence = load_dataset('sequence', radius, ngram)
    interaction = load_dataset('interaction', radius, ngram)

    """Make datasets."""
    dataset = list(zip(vertex, edge, adjacency, vertex_,
                       sequence, interaction))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    """Set a model."""
    u = 100  # The number of unknown vertices, edges, and words.
    n_vertex = load_dataset('n_vertex', radius, ngram) + u
    n_edge = load_dataset('n_edge', radius, ngram) + u
    n_word = load_dataset('n_word', radius, ngram) + u
    output_dim = 2  # Binary classification (interact or not).
    model = CompoundProteinInteraction()
    trainer = Trainer(model)
    tester = Tester(model)

    """Training the model."""
    print('Epoch Loss AUC_dev AUC_test Precision Recall')

    for epoch in range(1, 100):

        if epoch % 5 == 0:  # Every 5 epoch.
            trainer.optimizer.alpha *= lr_decay

        loss = trainer.train(dataset_train)
        auc_dev = tester.test(dataset_dev)[0]
        auc_test, precision, recall = tester.test(dataset_test)

        tester.result(epoch, loss, auc_dev, auc_test,
                      precision, recall, file_result)
        tester.save(model, file_model)

        print(epoch, loss, auc_dev, auc_test, precision, recall)
