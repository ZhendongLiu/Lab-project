import pickle
from pytrips.ontology import load as load_ontology
import simple_tagger as st
from pytrips.tools import nlp
from nltk.corpus import wordnet as wn

#TODO:
#1. check the case where size = 1
#2. finish test_semcor
#3. remember to consider all test cases 

ont = load_ontology()

n_gram_size = 1

def main():
	'''
	python3 decoder1.py [size] semcor 
	python3 decoder1.py [size] "[sent]"
	python3 decoder1.py [size] "[sent]" word
	'''
	import sys
	n_gram_size = int(sys.argv[1])

	files = list()
	

	all_A_sets = pickle.load(open("all_A_sets_size{}.pkl".format(n_gram_size),"rb"))
	files.append(all_A_sets)
	id_to_A_sets = pickle.load(open("id_to_A_sets_size{}.pkl".format(n_gram_size),'rb'))
	files.append(id_to_A_sets)
	n_gram_count = pickle.load(open('n_gram_count_size{}.pkl'.format(n_gram_size),'rb'))
	files.append(n_gram_count)
	s_to_sets = pickle.load(open('s_to_sets_size{}.pkl'.format(n_gram_size),'rb'))
	files.append(s_to_sets)
	sub_n_gram_count = pickle.load(open('sub_n_gram_count_size{}.pkl'.format(n_gram_size),'rb'))
	files.append(sub_n_gram_count)

	if sys.argv[2] == "semcor":
		test_semcor(n_gram_size, files)





def ngrams(tokens,n):
	'''
	return all n-grams of tokens as a list of tuples
	'''
	return [i for i in zip(*[tokens[i:] for i in range(n)])]

def sort(tu):
	lst = list(tu) 
	n = len(lst) 
	for i in range(n): 
		for j in range(0, n-i-1): 
			if lst[j] > lst[j+1]: 
				lst[j], lst[j+1] = lst[j+1], lst[j] 
	return tuple(lst)	

def gram_id(n_gram):
	'''
	assume n_gram is alrealy a tuple of integers
	'''
	return str('.'.join([str(i) for i in n_gram]))

def fetch_trips_types(sentence):
	'''
	return dictionary of word mapped to its TRIPS A-set
	reference: original code
	'''
	words_with_ontologies = {}
	tagging = st.tag_sentence(sentence)
	for j, i in tagging:
		if str(i) not in words_with_ontologies:
			j = frozenset([str(s) for s in j])

			words_with_ontologies[str(i)] = j
	return words_with_ontologies

def pre_process_sent(sent):
	
	token_to_a_set = fetch_trips_types(sent)
	words = ['_START_']
	dic = {'_START_':'_START_'} #potential problem: is it possible that one word has different tags?
	doc = nlp(sent)

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
			i += 1
			continue
			#words.append(str(token))
			
			#dic[str(token)] = "_PUNCT_"
				

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

	return words, taggings, dic

def laplace_estimate(count, N, B):

	return (count+1)/(N+B)
	



def distribution_one_gram(gram, posi, files):
	'''
	only consider the case where gram[posi] is a set of senses
	it should be garantee that gram[posi] is an a-set

				

	'''
	all_A_sets = files[0]
	id_to_A_sets = files[1]
	n_gram_count = files[2]
	s_to_sets = files[3]
	sub_n_gram_count = files[4]

	V = len(all_A_sets)
	ID = V
	
	N1 = len(n_gram_count)
	N2 = len(sub_n_gram_count)

	B1 = V ** 3
	B2 = V ** 2

	senses = list(gram[posi])
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


def test_semcor(n_gram_size,files):
	'''
	note:
		for now I skip the named entity which includes days etc.

	'''

	from nltk.corpus import semcor

	

	sentences = semcor.chunk_sents()
	#training_size = int(len(sentences)*(9/10))
	#sentences = sentences[training_size:]
	senses = pickle.load(open("sense.pkl",'rb'))
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
		print("zero_distribution:{}, percentage:{}".format(zero_distribution, zero_distribution/total))
		print("top1 correct     :{}, percentage:{}".format(top1,top1/total))
		print("top2 correct     :{}, percentage:{}".format(top2, top2/total))
		print("trivial          :{}, percentage:{}".format(trivil, trivil/total))
		print("noisy            :{}, percentage:{}".format(noisy, noisy/total))














main()










		