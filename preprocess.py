from nltk.corpus import semcor
from nltk.corpus import wordnet as wn

import pytrips.ontology
from pytrips.tools import nlp

import util_codes.simple_tagger as st
from util_codes.utils import *

import pickle
import numpy




def create_semcor_data_files(length):

	print("loading semcor...")
	sentences = semcor.chunk_sents()
	senses = [[str(c) for c in s] for s in semcor.tagged_sents(tag = 'both')[:length]]
	
	with open('data/sense.pkl','wb') as outfile:
		pickle.dump(senses, outfile, pickle.HIGHEST_PROTOCOL)

	print("semcor loaded")
	return sentences[:length]

def semcor_raw_sentences(length):

	sentences = create_semcor_data_files(length)

	raw_sentences = list()

	idx = 0
	for sentence in sentences:
		
		print(idx)
		idx += 1

		flat_sentence = [token for sublist in sentence for token in sublist]
		flat_sentence_string = ' '.join(flat_sentence)
		flat_sentence_string.replace('-','')
		raw_sentences.append(flat_sentence_string)

	return raw_sentences



def pickle_from_raw_texts(sentences):
	'''
	preprocess a list of raw sentences 
	'''
	result = list()
	idx = 0

	# 
	
	vocab = set()
	types = set()
	a_sets = set()
	vocab_to_types = dict()
	vocab_to_a_sets = dict()


	for sentence in sentences:

		token_to_a_set = tag_a_sentence(sentence)

		print(idx)
		print(sentence)

		idx += 1
		words, taggings, dic = pre_process_sent(sentence)
		
		#store each word, sense, a_set and the mapping from word to types, from word to a_set
		pairs = []
		for word, tag in zip(words, taggings):
			
			pair = (word,tag)
			pairs.append(pair)
			
			vocab.add(word)
			a_sets.add(tag)
			vocab_to_a_sets[word] = tag

			if word not in vocab_to_types:
				vocab_to_types[word] = set()

			if type(tag) == frozenset:
				for t in tag:
					types.add(t)
					vocab_to_types[word].add(t)
			else:
				vocab_to_types[word].add(tag)

		#end loop

		result.append(pairs)

	
	for i in vocab_to_types.items():
		vocab_to_types[i[0]] = frozenset(i[1])

	#print(list(vocab)[:10])
	#print(list(types)[:10])
	#print(list(a_sets)[:10])
	

	#for i in vocab_to_types.items():
	#	print(i)

	#for j in vocab_to_a_sets.items():
	#	print(j)


	training_size = int(len(result)*(9/10))

	training = result[:training_size]
	testing = result[training_size:]
	#result = tuple(result)

	with open('data/sentences.pkl','wb') as outfile:
		pickle.dump(result, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/training.pkl','wb') as outfile:
		pickle.dump(training, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/testing.pkl','wb') as outfile:
		pickle.dump(testing, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/vocab.pkl','wb') as outfile:
		pickle.dump(list(vocab), outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/types.pkl','wb') as outfile:
		pickle.dump(list(types), outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/a_sets.pkl','wb') as outfile:
		pickle.dump(list(a_sets), outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/vocab_to_types.pkl','wb') as outfile:
		pickle.dump(vocab_to_types, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/vocab_to_a_sets.pkl','wb') as outfile:
		pickle.dump(vocab_to_a_sets, outfile, pickle.HIGHEST_PROTOCOL)





def main():
	import sys

	if len(sys.argv) == 1:
		pickle_from_raw_texts(semcor_raw_sentences(37175))
	else:
		data_length = int(sys.argv[1])
		pickle_from_raw_texts(semcor_raw_sentences(data_length))

main()