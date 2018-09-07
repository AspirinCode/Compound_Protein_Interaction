from collections import defaultdict
import os
import pickle
import sys

import numpy as np

from rdkit import Chem


def pad(x_list, p):
    """Pad x_list with p."""
    len_max = max(map(len, x_list))
    pad_list = []
    for x in x_list:
        len_x = len(x)
        if (len_x < len_max):
            x += [p] * (len_max-len_x)
        pad_list.append(x)
    pad_array = np.array(pad_list)
    if (pad_array.dtype == 'int64'):
        return np.array(pad_array, np.int32)
    elif (pad_array.dtype == 'float64'):
        return np.array(pad_array, np.float32)


def make_atomlist(mol):
    atom_list = [atom_dict[a.GetSymbol()] for a in mol.GetAtoms()]
    return np.array(atom_list, np.int32)


def make_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def make_adjalist(mol):
    adja_mat = Chem.GetAdjacencyMatrix(mol)
    adja_list = [list(np.where(adja == 1)[0]) for adja in adja_mat]
    return adja_list


def Weisfeiler_Lehman(atom_list, i_jbond_dict, radius):
    """Extract the r-radius vertices and r-radius edges
    from a graph with WeisfeilerLehman-like algorithm."""

    if (radius == 0):
        vertex_list = atom_list
        edge_list = []
        for i, j_bond in i_jbond_dict.items():
            edge_list.append([bond for j, bond in j_bond])

    else:
        for _ in range(radius):

            """r-radius vertex."""
            vertex_list = []
            for i, j_bond in i_jbond_dict.items():
                v = (atom_list[i], tuple(sorted([(atom_list[j], bond)
                     for j, bond in j_bond])))
                v = vertex_dict[v]
                vertex_list.append(v)

            """r-radius edge."""
            edge_list = []
            for i, j_bond in i_jbond_dict.items():
                e_list = []
                for j, bond in j_bond:
                    e = (bond, tuple(sorted((atom_list[i], atom_list[j]))))
                    e = edge_dict[e]
                    e_list.append(e)
                edge_list.append(e_list)

            """Update atom_list and i_jbond_dict."""
            atom_list = vertex_list
            i_jedge_dict = defaultdict(lambda: [])
            for i, j_bond in i_jbond_dict.items():
                for k, (j, bond) in enumerate(j_bond):
                    i_jedge_dict[i].append((j, edge_list[i][k]))
            i_jbond_dict = i_jedge_dict

    return np.array(vertex_list, np.int32), pad(edge_list, -1)


def make_vertex_(adja_list):
    """Make vertex_, which is a vertex matrix corresponding to an edge matrix
    to efficiently compute an edge transition function
    introduced in our GNN."""
    vertex_ = [[i for a in adja] for i, adja in enumerate(adja_list)]
    return pad(vertex_, -1)


def make_sequence(sequence, ngram):
    """Split a sequence into n-gram words."""
    word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
    sequence = [word_dict[word] for word in word_list]
    return np.array(sequence, np.int32)


def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'))


if __name__ == "__main__":

    DATASET, radius, ngram = sys.argv[1:]
    radius, ngram = map(int, [radius, ngram])
    directory = '../dataset/' + DATASET + '/original/'
    data_file = open(directory + 'smiles_sequence_interaction.txt', 'r')

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    vertex_dict = defaultdict(lambda: len(vertex_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    Vertex, Edge, Adjacency, Vertex_ = [], [], [], []
    Sequence = []
    Interaction = []

    no, n_positive, n_negative = 0, 0, 0

    for data in data_file:

        smiles, sequence, interaction = data.strip().split(' ')

        if interaction == '1':
            n_positive += 1
        elif interaction == '0':
            n_negative += 1

        """Atom list."""
        mol = Chem.MolFromSmiles(smiles)
        atom_list = make_atomlist(mol)

        """Note that we exclude '.' data in smiles."""
        if ('.' not in smiles):
            no += 1
            print('Sample:', no)

            """Bond dictionary."""
            i_jbond_dict = make_ijbonddict(mol)

            """The r-radius vertex and edge."""
            if (len(atom_list) == 1):
                vertex = np.array(atom_list, np.int32)
                edge = np.array([[-1]], np.int32)
                adja_list = [[-1]]
            else:
                vertex, edge = Weisfeiler_Lehman(
                               atom_list, i_jbond_dict, radius)
                adja_list = make_adjalist(mol)
            Vertex.append(vertex)
            Edge.append(edge)

            """Vertex_."""
            vertex_ = make_vertex_(adja_list)
            Vertex_.append(vertex_)

            """Pad the adjacency list."""
            adja = pad(adja_list, -1)
            Adjacency.append(adja)

            """Split the sequence."""
            sequence = make_sequence(sequence, ngram)
            Sequence.append(sequence)

            """Label (interact or not)."""
            interaction = np.array([int(interaction)], np.int32)
            Interaction.append(interaction)

    """The number of WL vertices and edges."""
    if (radius == 0):
        vertex_dict, edge_dict = atom_dict, bond_dict
    n_vertex, n_edge = len(vertex_dict), len(edge_dict)
    n_word = len(word_dict)

    """Save dataset."""
    radius, ngram = str(radius), str(ngram)
    directory = ('../dataset/' + DATASET + '/input/radius' +
                 radius + '_ngram' + ngram + '/')
    if not os.path.isdir(directory):
        os.mkdir(directory)
    np.save(directory + 'vertex', Vertex)
    np.save(directory + 'edge', Edge)
    np.save(directory + 'adjacency', Adjacency)
    np.save(directory + 'vertex_', Vertex_)
    np.save(directory + 'sequence', Sequence)
    np.save(directory + 'interaction', Interaction)
    np.save(directory + 'n_vertex', n_vertex)
    np.save(directory + 'n_edge', n_edge)
    np.save(directory + 'n_word', n_word)

    """Save dictionaries (for analysis)."""
    pickle_dump(atom_dict, directory + 'atom_dict.pickle')
    pickle_dump(bond_dict, directory + 'bond_dict.pickle')
    pickle_dump(vertex_dict, directory + 'vertex_dict' + '.pickle')
    pickle_dump(edge_dict, directory + 'edge_dict' + '.pickle')
    pickle_dump(word_dict, directory + 'word_dict' + '.pickle')

    print('-'*30)
    print('The preprocess is finished!\nDataset information is as follows.')
    print('Dataset: ', DATASET)
    print('Radius: ', radius)
    print('N-gram: ', ngram)
    print('The number of r-radius vertices: ', n_vertex)
    print('The number of r-radius edges: ', n_edge)
    print('The number of n-gram: ', n_word)
    print('The number of positive samples: ', n_positive)
    print('The number of negative samples: ', n_negative)
    print('-'*30)
