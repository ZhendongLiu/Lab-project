
import sys
import scipy
from scipy import sparse as spa

from nltk.corpus import wordnet as wn 

import pickle
cor_name = sys.argv[1]

vocab = pickle.load(open("data/text8_out/{}_vocab.pkl".format(cor_name),"rb"))
types = pickle.load(open("data/text8_out/{}_types.pkl".format(cor_name),"rb"))
a_sets = pickle.load(open("data/text8_out/{}_a_sets.pkl".format(cor_name),"rb"))
vocab_to_types = pickle.load(open("data/text8_out/{}_vocab_to_types.pkl".format(cor_name),'rb')) # word to list 
vocab_to_a_sets = pickle.load(open("data/text8_out/{}_vocab_to_a_sets.pkl".format(cor_name),'rb')) # word to an a_sets, can be modified to a list of a_sets


W = scipy.zeros((len(vocab),len(types)))

for i in range(len(vocab)):
	for j in range(len(types)):
		if types[j] in vocab_to_types[vocab[i]]:
			W[i][j] = 1

W = spa.csc_matrix(W)
spa.save_npz('data/{}_W.npz'.format(cor_name),W)

A = scipy.zeros((len(vocab),len(a_sets)))

for i in range(len(vocab)):
	for j in range(len(a_sets)):
		if a_sets[j] == vocab_to_a_sets[vocab[i]]:
			A[i][j] = 1

A = spa.csc_matrix(A)
spa.save_npz('data/{}_A.npz'.format(cor_name),A)

S = scipy.zeros((len(a_sets),len(types)))

for i in range(len(a_sets)):
	for j in range(len(types)):
		if types[j] in a_sets[i]:
			S[i][j] = 1

S = spa.csc_matrix(S)
spa.save_npz('data/{}_S.npz'.format(cor_name),S)

type_to_idx = dict()

for i in range(len(types)):
	type_to_idx[types[i]] = i

T = scipy.zeros((len(types),len(types)))

for i in range(len(types)):
	ss = types[i]
	ss = wn.synset(ss[8:-2])

	hypers = ss.hypernyms()
	hypos = ss.hyponyms()
	T[i][i] = 1
	for hyper in hypers:
		if str(hyper) in types:
			j = type_to_idx[str(hyper)]
			T[i][j] = 0.25

	hypos = ss.hyponyms()
	for hypo in hypos:
		if str(hypo) in types:
			j = type_to_idx[str(hypo)]
			T[i][j] = 0.75

T = spa.csc_matrix(T)
spa.save_npz('data/{}_T.npz'.format(cor_name),T)

vocab_to_idx = dict()
for i in range(len(vocab)):
	vocab_to_idx[vocab[i]] = i

pickle.dump(vocab_to_idx, open("data/{}_vocab_to_idx.pkl".format(cor_name),'wb'))


