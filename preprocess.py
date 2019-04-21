from nltk.corpus import semcor
from nltk.corpus import wordnet as wn

import pytrips.ontology
from pytrips.tools import nlp

import util_codes.simple_tagger as st
from util_codes.utils import *

import pickle
import numpy



def text8_raw_sentences(length):
	
	'''
	may want to replace the lower frequency words
	'''
	from gensim.summarization import textcleaner

	text_file = open("text")
	text = text_file.read()

	sentences = textcleaner.get_sentences(text)
	sentences = [i for i in sentences]

	if length != -1:
		return sentences[:length]
	else:
		return sentences


def create_semcor_data_files(length):

	print("loading semcor...")
	sentences = semcor.chunk_sents()
	senses = [[str(c) for c in s] for s in semcor.tagged_sents(tag = 'both')[:length]]
	
	with open('data/sense.pkl','wb') as outfile:
		pickle.dump(senses, outfile, pickle.HIGHEST_PROTOCOL)

	print("semcor loaded")
	
	if length != -1:

		return sentences[:length]
	else:
		return sentences

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



def pickle_from_raw_texts(sentences, cor_name):
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

		

		
		words, taggings, dic = pre_process_sent(sentence)
		
		if len(words) <= 20:
			continue
		
		print(idx)
		print('|'.join(words))
		idx += 1
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

	with open('data/{}_sentences.pkl'.format(cor_name),'wb') as outfile:
		pickle.dump(result, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/{}_training.pkl'.format(cor_name),'wb') as outfile:
		pickle.dump(training, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/{}_testing.pkl'.format(cor_name),'wb') as outfile:
		pickle.dump(testing, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/{}_vocab.pkl'.format(cor_name),'wb') as outfile:
		pickle.dump(list(vocab), outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/{}_types.pkl'.format(cor_name),'wb') as outfile:
		pickle.dump(list(types), outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/{}_a_sets.pkl'.format(cor_name),'wb') as outfile:
		pickle.dump(list(a_sets), outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/{}_vocab_to_types.pkl'.format(cor_name),'wb') as outfile:
		pickle.dump(vocab_to_types, outfile, pickle.HIGHEST_PROTOCOL)

	with open('data/{}_vocab_to_a_sets.pkl'.format(cor_name),'wb') as outfile:
		pickle.dump(vocab_to_a_sets, outfile, pickle.HIGHEST_PROTOCOL)





def main():
	import sys

	#length = -1 if you want all
	name = sys.argv[1]
	length = int(sys.argv[2])

	if name == 'semcor':
		pickle_from_raw_texts(semcor_raw_sentences(length),name)
	else:
		pickle_from_raw_texts(text8_raw_sentences(length),name)
	
main()