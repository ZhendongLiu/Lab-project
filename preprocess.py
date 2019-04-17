from nltk.corpus import semcor
from nltk.corpus import wordnet as wn

import pytrips.ontology
from pytrips.tools import nlp

import util_codes.simple_tagger as st
from util_codes.utils import *

import pickle




def create_semcor_data_files():

	print("loading semcor...")
	sentences = semcor.chunk_sents()
	#senses = [[str(c) for c in s] for s in semcor.tagged_sents(tag = 'both')[:37176]]
	
	#with open('data/sense.pkl','wb') as outfile:
	#	pickle.dump(senses, outfile, pickle.HIGHEST_PROTOCOL)

	print("semcor loaded")
	return sentences

def semcor_raw_sentences():

	sentences= create_semcor_data_files()

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
	
	result = list()
	idx = 0

	for sentence in sentences:

		token_to_a_set = tag_a_sentence(sentence)

		print(idx)
		print(sentence)

		idx += 1
		words, taggings, dic = pre_process_sent(sentence)
		
		pairs = [i for i in zip(words, taggings)]
		pairs = tuple(pairs)
		result.append(pairs)

	training_size = int(len(result)*(9/10))

	training = result[:training_size]
	testing = result[training_size:]
	result = tuple(result)

	with open('data/sentences.pkl','wb') as outfile:
		pickle.dump(result, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/training.pkl','wb') as outfile:
		pickle.dump(training, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/testing.pkl','wb') as outfile:
		pickle.dump(testing, outfile, pickle.HIGHEST_PROTOCOL)



pickle_from_raw_texts(semcor_raw_sentences())