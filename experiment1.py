'''
experiment config:
	train on text8
	test on 5000 semcor sentences

	algorithm as given by rik

	feature:
		replace low frequency words with UNK(less than 5)
		vocab only consider those with a-set

'''
import scipy
from scipy import sparse as spa
import pickle
import util_codes.simple_tagger as st 
from util_codes.utils import *

from nltk.corpus import wordnet as wn 

import numpy

import sys

cor_name = sys.argv[1]
sent_size = int(sys.argv[2])
test_semcor(sent_size)

vocab = pickle.load(open("data/{}_vocab.pkl".format(cor_name),"rb"))
types = pickle.load(open("data/{}_types.pkl".format(cor_name),"rb"))
a_sets = pickle.load(open("data/{}_a_sets.pkl".format(cor_name),"rb"))
vocab_to_types = pickle.load(open("data/{}_vocab_to_types.pkl".format(cor_name),'rb')) # word to list 
vocab_to_a_sets = pickle.load(open("data/{}_vocab_to_a_sets.pkl".format(cor_name),'rb'))
vocab_to_idx = pickle.load(open("data/{}_vocab_to_idx.pkl".format(cor_name),'rb'))

n_gram_count_word = pickle.load(open("data/{}_n_gram_count_word_size3.pkl".format(cor_name),'rb'))
sub_n_gram_count_word = pickle.load(open("data/{}_sub_n_gram_count_word_size3.pkl".format(cor_name),'rb'))

#spa.save_npz('data/{}_W.npz'.format(cor_name),W)
W = spa.load_npz('data/{}_W.npz'.format(cor_name))
W = W.todense()
A = spa.load_npz('data/{}_A.npz'.format(cor_name))
S = spa.load_npz('data/{}_S.npz'.format(cor_name))

def word_distribution_to_td(word_d, word_index):
	'''
	word_d : W x 1
	W: w x t
	A: w x a
	S: a x t
	'''

	from scipy.sparse import csc_matrix

	

	flat_a_set_distribution = word_d * a_sparse

	flat_t_distribution = flat_a_set_distribution * s_sparse

	t_in_w_distribution = W[word_index] * flat_t_distribution

	return t_in_w_distribution


def word_distribution_from_ngram(ngram):
	#use laplace
	N1 = len(n_gram_count_word)
	N2 = len(sub_n_gram_count_word)

	V = len(vocab)

	B1 = V * 3
	B2 = V * 2

	sub_gram = list(ngram)
	sub_gram = tuple(sub_gram[:-1])


	if sub_gram in sub_n_gram_count_word:
		sub_count = sub_n_gram_count_word[sub_gram]
	else:
		sub_count = 0

	p2 = laplace_estimate(sub_count,N2, B2)

	distribution = []


	for w in vocab:
		ngram = list(ngram)
		ngram.pop(2)
		ngram.insert(2,w)
		ngram = tuple(ngram)

		if ngram in n_gram_count_word:
			ngram_count = n_gram_count_word[ngram]
		else:
			ngram_count = 0

	
		p1 = laplace_estimate(ngram_count,N1, B1)
		distribution.append(p1/p2)

	return scipy.array(distribution)

def whole_sent_distribution(sentence):
	'''

	input: a raw sentence
	return:
	'''
	words, taggings, dic = pre_process_sent(sentence)
	words.insert(0,"_START_")

	n_grams = ngrams(words, 3)
	words_distributions = ["_START_"]

	print(n_grams)
	for g in n_grams:
		
		word = g[2]
		
		if word == '_END_':
			break

		if type(dic[word]) == str:
			words_distributions.append(dic[word])
			continue

		word_distribution = word_distribution_from_ngram(g)
		type_distribution = word_distribution_to_td(word_distribution, vocab_to_idx[word])

		distribution = dict()
		
		for i in range(len(type_distribution)):
			if type_distribution[i] != 0:
				distribution[types[i]] = type_distribution[i]

		distribution = sort_result(distribution)
		words_distributions.append(distribution)

	
	return [i for i in zip(words[1:], words_distributions)]

def laplace_estimate(count, N, B):

	return (count+1)/(N+B)

def sort_result(r):
	pairs = [i for i in r.items()]

	lst = list(pairs) 
	n = len(lst) 
	for i in range(n): 
		for j in range(0, n-i-1): 
			if lst[j][1] < lst[j+1][1]: 
				lst[j], lst[j+1] = lst[j+1], lst[j] 

	return lst


def test_semcor(size):
	from nltk.corpus import semcor

	sentences = semcor.chunk_sents()
	sentences = sentences[:size]

	senses = pickle.load(open("data/sense.pkl",'rb'))
	idx = 0

	total = 0 #total words that the model should give answer to
	top1 = 0 # number of words that the top-1 rated sense is correct
	top2 = 0 # number of words that the top-2 rated sense is correct
	trivil = 0 #number of words that the model gives non-zero distribution but there is only one sense

	for sentence in sentences:

		wn_tagging = senses[idx]
		idx += 1

		words = list()
		word_to_type = dict()

		for sublist, wn_tag in zip(sentence, wn_tagging):

			if len(sublist) == 1:
				word = sublist[0]
				words.append(word)
				string, pos_tag, lemma = parse_tagged_chunks(wn_tag)

				lemma = clean_lemma(lemma)

				if '.' in lemma:
					try:
						synset = wn.synset(lemma)
					except:
						continue

					if not (synset.pos() == 'v' or synset.pos() == 'n'):
						continue

					print(synset)
					
					if word not in word_to_type:
						word_to_type[word] = [synset]
					else:
						word_to_type[word].append(synset)
			else:
				for word in sublist:
					words.append(word)

		sentence_str = ' '.join(words)
		print("sentence No.{}\n{}".format(idx, sentence_str))
		words_distributions = whole_sent_distribution(sentence_str)

		print(word_to_type)
		occurance = dict()
		for word, distribution in words_distributions:
			
			if type(distribution) == str:
				print("\tword:{}, tag:{}".format(word, distribution))
				#idx1 += len(word.split())
			
			#fetch type
			else:
				try:
					#if this word has a wn-tagging
					if word in word_to_type:
						if word not in occurance:
							occurance[word] = 0
						else:
							occurance[word] += 1
						t = word_to_type[word][occurance[word]]
					else:
						continue
					
				except:
					continue
				#onts = right_tag[idx1]

				#if onts == "NON":
				#	idx1 += 1
				#	continue
				#print("\tright word:{}".format(words[idx1]))
				#idx1 += 1
				total += 1

				print("\tword:{}, correct sense:{}".format(word, t))

				for i in range(len(distribution)):
						print("\t\t{}".format(distribution[i]))
				first_second = 1
				

				for i in range(len(distribution)):
						if i == 0 and distribution[i][1] == 0:
							zero_distribution += 1
							break
						elif first_second == 1:
							if str(distribution[i][0]) == str(t):
								top1 += 1
								top2 += 1
								if len(distribution) == 1:
									trivil += 1
							if i + 1 < len(distribution):
								if distribution[i][1] != distribution[i+1][1]:
									first_second += 1
							
						elif first_second == 2 and distribution[i][1] != 0:
							if str(distribution[i][0]) == str(t):
								top2 += 1
							if i+1 < len(distribution):
								if distribution[i][1] != distribution[i+1][1]:
									first_second += 1


			
					

		print("total tasks:{}".format(total))
		#print("zero_distribution:{}, percentage:{}".format(zero_distribution, zero_distribution/total))
		print("top1 correct     :{}, percentage:{}".format(top1,top1/total))
		print("top2 correct     :{}, percentage:{}".format(top2, top2/total))
		print("trivial          :{}, percentage:{}".format(trivil, trivil/total))
		

