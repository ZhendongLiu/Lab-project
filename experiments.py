import pickle

import util_codes.simple_tagger as st
from util_codes.utils import *

#from pytrips.ontology import load as load_ontology
#from pytrips.tools import nlp

from nltk.corpus import wordnet as wn

import numpy

import sys
#TODO:
#1. check the case where size = 1
#2. finish test_semcor
#3. remember to consider all test cases 

#ont = load_ontology()
cor_name = sys.argv[1]

n_gram_size = 3

vocab = pickle.load(open("data/{}_vocab.pkl".format(cor_name),"rb"))
types = pickle.load(open("data/{}_types.pkl".format(cor_name),"rb"))
a_sets = pickle.load(open("data/{}_a_sets.pkl".format(cor_name),"rb"))
vocab_to_types = pickle.load(open("data/{}_vocab_to_types.pkl".format(cor_name),'rb')) # word to list 
vocab_to_a_sets = pickle.load(open("data/{}_vocab_to_a_sets.pkl".format(cor_name),'rb')) # word to an a_sets, can be modified to a list of a_sets
#should be loaded from files
#learning git

'''
vocab_to_idx = dict()
for i in range(len(vocab)):
	vocab_to_idx[vocab[i]] = i

pickle.dump(vocab_to_idx,open("data/vocab_to_idx.pkl",'wb'))

'''
vocab_to_idx = pickle.load(open("data/vocab_to_idx.pkl",'rb'))


#if we are doing n_gram for types
'''
all_A_sets = pickle.load(open("data/all_A_sets_size{}.pkl".format(n_gram_size),"rb"))
id_to_A_sets = pickle.load(open("data/id_to_A_sets_size{}.pkl".format(n_gram_size),'rb'))
n_gram_count = pickle.load(open('data/n_gram_count_size{}.pkl'.format(n_gram_size),'rb'))
s_to_sets = pickle.load(open('data/s_to_sets_size{}.pkl'.format(n_gram_size),'rb'))
sub_n_gram_count = pickle.load(open('data/sub_n_gram_count_size{}.pkl'.format(n_gram_size),'rb'))
'''
#if we are doint n_gram for words
n_gram_count_word = pickle.load(open("data/n_gram_count_word_size3.pkl",'rb'))
sub_n_gram_count_word = pickle.load(open("data/sub_n_gram_count_word_size3.pkl",'rb'))


'''
W = numpy.zeros((len(vocab), len(types)))

print("48")
for i in range(len(vocab)):
	for j in range(len(types)):
		if types[j] in vocab_to_types[vocab[i]]:
			W[i][j] = 1
'''
from scipy.sparse import csc_matrix

W = numpy.load(open("data/W.pkl",'rb'))

'''
print("54")
A = numpy.zeros((len(vocab),len(a_sets)))

for i in range(len(vocab)):
	for j in range(len(a_sets)):
		if a_sets[j] == vocab_to_a_sets[vocab[i]]:
			A[i][j] = 1
print("61")
'''
A = numpy.load(open("data/A.pkl",'rb'))
a_sparse = csc_matrix(A)

'''
S = numpy.zeros((len(a_sets),len(types)))

for i in range(len(a_sets)):
	for j in range(len(types)):
		if types[j] in a_sets[i]:
			S[i][j] = 1
print("51")
'''
S = numpy.load(open("data/S.pkl",'rb'))
s_sparse = csc_matrix(S)

def word_distribution_to_td(word_d, word_index):
	'''
	word_d : W x 1
	W: w x t
	A: w x a
	S: a x t
	'''

	from scipy.sparse import csc_matrix

	

	flat_a_set_distribution = word_d * a_sparse

	#a_sets_given_context = word_d.T.reshape(len(word_d),1) * A # w x a
	#flat_a_set_distribution = numpy.sum(a_sets_given_context, axis = 0) #1 x a
		

	flat_t_distribution = flat_a_set_distribution * s_sparse

	t_in_w_distribution = W[word_index] * flat_t_distribution

	print("95")
	return t_in_w_distribution

def word_distribution_from_ngram(ngram):
	'''
	for word we are using two previous words as context
	and laplace estimater
	'''
	N1 = len(n_gram_count_word)
	N2 = len(sub_n_gram_count_word)

	V  = len(vocab)

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
		

	return numpy.array(distribution)


def whole_sent_distribution_1(sentence):
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










def main():
	'''
	python3 decoder1.py [size] semcor 
	python3 decoder1.py [size] "[sent]"
	python3 decoder1.py [size] "[sent]" word
	'''
	import sys
	sent_size = int(sys.argv[1])
	
	

	if sys.argv[2] == "semcor":
		test_semcor_v1(sent_size)


def laplace_estimate(count, N, B):

	return (count+1)/(N+B)
	



def distribution_one_gram(gram, posi, files):
	'''
	only consider the case where gram[posi] is a set of senses
	it should be garantee that gram[posi] is an a-set
	

				

	'''

	V = len(all_A_sets)
	ID = V
	
	N1 = len(n_gram_count)
	N2 = len(sub_n_gram_count)

	B1 = V ** 3
	B2 = V ** 2

	senses = list(gram[posi])
	this_id = all_A_sets[gram[posi]]
	sub_gram = list(gram)
	sub_gram.pop(posi)

	
	sub_gram1 = list()

	for g in sub_gram:
		if g in all_A_sets:
			sub_gram1.append(all_A_sets[g])
		else:
			sub_gram1.append(ID)
			ID += 1

	sub_gram = sub_gram1

	sub_gram = sort(sub_gram)
	sub_gram = gram_id(sub_gram)

	sub_gram_count = 0

	if sub_gram in sub_n_gram_count:
		sub_gram_pd = sub_n_gram_count[sub_gram]
	
	sub_gram_p = laplace_estimate(sub_gram_count, N2, B2)

	distribution = dict()

	gram1 = list()

	for i in gram:
		if i in all_A_sets:
			gram1.append(all_A_sets[i])
		else:
			gram1.append(ID)
			ID += 1

	gram = gram1

	for s in senses:
	
		s = str(s)

		if s not in s_to_sets:
			continue
		a_sets = s_to_sets[s]
		#print("214:{}".format( this_id in a_sets))
		a_sets = [id_to_A_sets[i] for i in a_sets if i in id_to_A_sets]
		p = 0

		for a_set in a_sets:
			gram = list(gram)
			gram.pop(posi)
			gram.insert(posi, all_A_sets[a_set])
			gram = tuple(gram)
			gram = sort(gram)
			gram_ID = gram_id(gram)

			#print(gram_ID)

			gram_count = 0

			if gram_ID in n_gram_count:
				gram_count = n_gram_count[gram_ID]
				
			if not gram_count:
					continue
			gram_p = laplace_estimate(gram_count, N1,B1)

			p1 = (gram_p/sub_gram_p)
			p += p1 


		distribution[s] = p

	return distribution

def sort_result(r):
	pairs = [i for i in r.items()]

	lst = list(pairs) 
	n = len(lst) 
	for i in range(n): 
		for j in range(0, n-i-1): 
			if lst[j][1] < lst[j+1][1]: 
				lst[j], lst[j+1] = lst[j+1], lst[j] 

	return lst

def naive_whole_sent_distribution(sent, n_gram_size,files):

	#this function provides the baseline score

	s_to_sets = files[3]
	words, taggings, dic = pre_process_sent(sent)
	
	grams = ngrams(taggings, n_gram_size)
	words_distributions = []

	for w_posi in range(len(words)):
		
		word = words[w_posi]
		
		


		if type(dic[word]) == str:
			words_distributions.append(dic[word])
			continue

		else:
			distribution = dict()
			a_set = taggings[w_posi]
			for s in a_set:
				num = len(s_to_sets[s])
				distribution[s] = num



		distribution = sort_result(distribution)
		words_distributions.append(distribution)

	return [i for i in zip(words, words_distributions)]



def whole_sent_distribution(sent, n_gram_size, files):
	'''
	preprocess the sentence as in the preprocessing.py

	returns [(word, distribution)]

	'''

	words, taggings, dic = pre_process_sent(sent)
	
	grams = ngrams(taggings, n_gram_size)
	words_distributions = []

	for w_posi in range(len(words)):
		
		word = words[w_posi]
		
		if type(taggings[w_posi]) == str:
			words_distributions.append(taggings[w_posi])
			continue

		#indexs = [i for i in range(len(words)) if str(words[i]) == word]
		#dic1 = {i:[] for i in indexs}
		idx = 0	

		grams_set = [] #n_grams that contains this word

		#ISSUE
		for g in grams:
			if idx <= w_posi and idx + n_gram_size > w_posi:
				grams_set.append((g, w_posi-idx))
			idx += 1

		distribution = dict()

		for g in grams_set:
			gram = g[0]
			posi = g[1]
			d = distribution_one_gram(gram, posi,files)
			if len(distribution) == 0:
				distribution = d
			else:
				for t in d:
					distribution[t] += d[t]

		distribution = sort_result(distribution)
		words_distributions.append(distribution)

	return [i for i in zip(words, words_distributions)]

def parse_tagged_chunks(string):
	lemma_string = 'Lemma'
	lemma = ''
	position = string.find(lemma_string)

	if position != -1:
		start_lemma = position + len(lemma_string) + 1
		end_lemma = string.find(')')
		lemma = string[start_lemma + 1 : end_lemma - 1]
		string = string[end_lemma + 2 : ]

	tag_pos_start = string.rfind('(')
	tag_pos_end = string.find(')', tag_pos_start)
	string = string[tag_pos_start + 1 : tag_pos_end]
	position = string.find(' ')
	pos_tag = string[: position]
	string = string[position + 1: ]

	return string, pos_tag, lemma

def clean_lemma(lemma):
	split_lemma = lemma.split('.')
	return '.'.join(split_lemma[:3])

def test_semcor_v1(size):
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
		words_distributions = whole_sent_distribution_1(sentence_str)

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
		


def test_semcor(n_gram_size,files):
	'''
	note:
		for now I skip the named entity which includes days etc.

	'''

	from nltk.corpus import semcor

	

	sentences = semcor.chunk_sents()
	#training_size = int(len(sentences)*(9/10))
	#sentences = sentences[training_size:]
	senses = pickle.load(open("data/sense.pkl",'rb'))
	#senses = senses[training_size:]
	idx = 0

	total = 0 #total words that the model should give answer to
	zero_distribution = 0 # number of words that the model give all-zero distribution
	top1 = 0 # number of words that the top-1 rated sense is correct
	top2 = 0 # number of words that the top-2 rated sense is correct
	trivil = 0 #number of words that the model gives non-zero distribution but there is only one sense
	noisy = 0 #number of words where the "correct" sense mapped from wn-tag is not in the a-set
	sense_mapping_fail = 0
	
	for sentence in sentences:
		
		wn_tagging = senses[idx]
		idx += 1
		
		word_to_onts = dict()
		words = list()
		
		for sublist, wn_tag in zip(sentence, wn_tagging):
			
			if len(sublist) == 1:
				word = sublist[0]
				words.append(word)
				string, pos_tag, lemma = parse_tagged_chunks(wn_tag)
				#print("word: " + word + " tag:" + wn_tag)
				lemma = clean_lemma(lemma)
				
				if '.' in lemma:
					
					#print("{}:{}".format(word,lemma))
					try:
						synset = wn.synset(lemma)
					except:
						continue
					
					if not (synset.pos() == 'v' or synset.pos() == 'n'):
						continue

					onts = list(set(ont[synset]))
					onts = [str(s) for s in onts]
					print("{},{},{}\n".format(word, lemma, onts))
					if word not in word_to_onts:
						word_to_onts[word] = [onts]
					else:
						word_to_onts[word].append(onts)
			else:
				for w in sublist:
					words.append(w)

		sentence_str = ' '.join(words)
		print("sentence No.{}\n{}".format(idx, sentence_str))
		words_distributions = whole_sent_distribution(sentence_str, n_gram_size, files)
		

		
		occurance = dict()
		for word, distribution in words_distributions:
			if type(distribution) == str:
				print("\tword:{}, tag:{}".format(word, distribution))
				#idx1 += len(word.split())
			else:
				try:
					#if this word has a wn-tagging
					if word in word_to_onts:
						if word not in occurance:
							occurance[word] = 0
						else:
							occurance[word] += 1
						onts = word_to_onts[word][occurance[word]]
					else:
						continue
					if len(onts) == 0:
						#no mapping from wn to trips senses
						sense_mapping_fail += 1
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

				print("\tword:{}, correct sense:{}".format(word, onts))

				for i in range(len(distribution)):
						print("\t\t{}".format(distribution[i]))
				first_second = 1
				not_noisy = False 
				for i in distribution:
						if i[0] in onts:
							not_noisy = True

				if not not_noisy:
						noisy += 1
						continue
				for i in range(len(distribution)):
						if i == 0 and distribution[i][1] == 0:
							zero_distribution += 1
							break
						elif first_second == 1:
							if distribution[i][0] in onts:
								top1 += 1
								top2 += 1
								if len(distribution) == 1:
									trivil += 1
							if i + 1 < len(distribution):
								if distribution[i][1] != distribution[i+1][1]:
									first_second += 1
							
						elif first_second == 2 and distribution[i][1] != 0:
							if distribution[i][0] in onts:
								top2 += 1
							if i+1 < len(distribution):
								if distribution[i][1] != distribution[i+1][1]:
									first_second += 1


			
					

		print("total tasks:{}".format(total))
		#print("zero_distribution:{}, percentage:{}".format(zero_distribution, zero_distribution/total))
		print("top1 correct     :{}, percentage:{}".format(top1,top1/total))
		print("top2 correct     :{}, percentage:{}".format(top2, top2/total))
		print("trivial          :{}, percentage:{}".format(trivil, trivil/total))
		#print("noisy            :{}, percentage:{}".format(noisy, noisy/total))














main()










		
