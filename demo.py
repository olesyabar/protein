import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM

with open('yeast_cell_membrane.txt', 'r') as mem_fi:
    mem_seqs = [line.strip() for line in mem_fi]
with open('yeast_nucleolus.txt', 'r') as nuc_fi:
    nuc_seqs = [line.strip() for line in nuc_fi]

def reduce_sequence(sequences):
    aa_dict = {'I': 'O', 'V': 'O', 'L': 'O', 'F': 'O', 'C': 'O', 'M': 'O', 'A': 'O', # Hydrophobic amino acids
               'G': 'N', 'T': 'N', 'W': 'N', 'S': 'N', 'Y': 'N', 'P': 'N', # Neutral amino acids
               'H': 'I', 'E': 'I', 'Q': 'I', 'D': 'I', 'N': 'I', 'K': 'I', 'R': 'I'} # Hydrophilic amino acids
    reduced_sequences = []
    for sequence in sequences:
        reduced_sequence = ''
        for amino_acid in sequence:
            reduced_sequence += aa_dict[amino_acid]
        reduced_sequences.append(reduced_sequence)
    return reduced_sequences

example_seq = ['DCDVIINELCHRLGGEYAKLCCNPVKLSE']
res = reduce_sequence(example_seq)
print(res)

class Option1:
    """
    Sequence processing - Extract numerical features
    Machine Learning algorithm - Support vector machine using feature extraction
    """
    def run(self):
        test_size = [0.25, 0.33, 0.37, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.5, 0.55, 0.6, 0.66, 0.75, 0.99]

        def calc_composition(sequences):
            compositions = []
            for sequence in sequences:
                composition = []
                for aa in ['O', 'I']:
                    composition.append(sequence.count(aa)/len(sequence))
                compositions.append(composition)
            return compositions

        print(example_seq, calc_composition(res))

        red_mem = reduce_sequence(mem_seqs)
        red_nuc = reduce_sequence(nuc_seqs)

        comps = calc_composition(red_mem) + calc_composition(red_nuc)
        y = [0]*len(mem_seqs) + [1]*len(nuc_seqs)

        x = np.array(comps)
        y = np.array(y)

        for i in test_size:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i, random_state=42)

            clf = svm.LinearSVC()
            clf.fit(x_train, y_train)

            accuracy = clf.score(x_test, y_test)
            print("The accuracy for model with size {} is {}".format(str(i), str(accuracy)))

        def plot_contour(clf, x, y):
            plt.scatter(x[:, 0], x[:, 1], c=y, zorder=9, cmap=plt.cm.Paired,
                        edgecolor='black', s=5)
            x_min = x[:, 0].min()
            x_max = x[:, 0].max()
            y_min = x[:, 1].min()
            y_max = x[:, 1].max()
            xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.pcolormesh(xx, yy, Z > 0, cmap=plt.cm.Paired)
            plt.contour(xx, yy, Z, colors=['white'],
                        linestyles=['-'], levels=[0], zorder=10)
            plt.xlabel('Hydrophobicity')
            plt.ylabel('Hydrophilicity')
            plt.show()

        #plot_contour(clf, x_train, y_train)


class Option2:
    """
    Sequence processing - One hot encoding
    Machine Learning algorithm - LSTM neural network
    """
    def run(self):
        def one_hot_sequences(sequences):
            oh_vecs = []
            oh = {'O': [1, 0, 0, 0], 'N': [0, 1, 0, 0], 'I': [0, 0, 1, 0], '0': [0, 0, 0, 1]}
            for sequence in sequences:
                oh_vec = []
                for aa in sequence:
                    oh_vec.append(oh[aa])
                oh_vecs.append(oh_vec)
            return oh_vecs

        def one_hot_y(y):
            oh_vecs = []
            oh = {0: [0, 1], 1: [1, 0]}
            for category in y:
                oh_vecs.append(oh[category])
            return oh_vecs

        example_seq = ['IOIONN']
        print(one_hot_sequences(example_seq))
        example_y = [0, 0, 1, 1]
        print()
        print(one_hot_y(example_y))

        def truncate_and_pad(sequences, cutoff):
            new_sequences = []
            for sequence in sequences:
                if len(sequence) > cutoff:
                    new_sequences.append(sequence[0:cutoff])
                else:
                    padding = cutoff - len(sequence)
                    new_sequences.append(sequence + '0'*padding)
            return new_sequences

        sequence_length = 100

        red_mem = reduce_sequence(mem_seqs)
        red_nuc = reduce_sequence(nuc_seqs)
        oh_mem = one_hot_sequences(truncate_and_pad(red_mem, sequence_length))
        oh_nuc = one_hot_sequences(truncate_and_pad(red_nuc, sequence_length))
        x = np.array(oh_mem + oh_nuc)
        y = [0]*len(red_mem) + [1]*len(red_nuc)
        y = np.array(one_hot_y(y))

        print(x.shape)
        print(y.shape)

        n_sequences = x.shape[0]
        n_categories = y.shape[1]
        data_dim = x.shape[2] # This is the number of unique letters in your alphabet plus 1 for the padding value

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        model = Sequential()
        model.add(LSTM(32, input_shape=(sequence_length, data_dim)))
        model.add(Dense(n_categories, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

Option1().run()