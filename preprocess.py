from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
import pytrips.ontology
from nltk.corpus import wordnet as wn 
import simple_tagger as st
import pickle
from pytrips.tools import nlp
from codes.utils import tag_a_sentence, pre_process_sent

def create_semcor_data_files():

	print("loading semcor...")
	sentences = semcor.chunk_sents()
	senses = [[str(c) for c in s] for s in semcor.tagged_sents(tag = 'both')[:37176]]
	
	with open('sense.pkl','wb') as outfile:
		pickle.dump(senses, outfile, pickle.HIGHEST_PROTOCOL)

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


		'''
		words = ['_START_']
		dic = {'_START_':'_START_'}

		doc = nlp(sentence)
		
		print(idx)
		print(sentence)

		idx += 1
		i = 0

		while i < len(doc):
			token = doc[i]

			if token.ent_type_ != '':
				tag = token.ent_type_
				ent = []
				while i < len(doc) and doc[i].ent_type_ != '' and doc[i].ent_type_ == tag:
					ent.append(str(doc[i]))
					i += 1
				ent_name = ' '.join(ent)
				words.append(ent_name)
				dic[ent_name] = "_{}_".format(tag)
				continue

			elif token.like_num:
				words.append(str(token))
				dic[str(token)] = "_NUMBER_"
				

			elif token.is_punct:
				words.append(str(token))
				dic[str(token)] = "_PUNCT_"
				

			elif token.is_stop:
				words.append(str(token))
				dic[str(token)] = '_STOP_'

			else:
				if str(token) in token_to_a_set and len(token_to_a_set[str(token)]) != 0:
					words.append(str(token))
					dic[str(token)] = frozenset(token_to_a_set[str(token)])
				else:
					words.append(str(token))
					dic[str(token)] = 'POS::' + token.pos_ 
			i += 1

		words.append('_END_')
		dic['_END_'] = '_END_'
		taggings = [dic[word] for word in words]
		'''

		words, taggings, dic = pre_process_sent(sentence)
		
		pairs = [i for i in zip(words, taggings)]
		pairs = tuple(pairs)
		result.append(pairs)


		
	training_size = int(len(result)*(9/10))

	training = result[:training_size]
	testing = result[training_size:]
	result = tuple(result)

	with open('sentences.pkl','wb') as outfile:
		pickle.dump(result, outfile, pickle.HIGHEST_PROTOCOL)

	with open('training.pkl','wb') as outfile:
		pickle.dump(training, outfile, pickle.HIGHEST_PROTOCOL)

	with open('testing.pkl','wb') as outfile:
		pickle.dump(testing, outfile, pickle.HIGHEST_PROTOCOL)



pickle_from_raw_texts(semcor_raw_sentences())