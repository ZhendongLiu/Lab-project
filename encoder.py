'''


laplace








'''

import pickle

#util functions
def ngrams(tokens, n):
	'''
	return all n-grams of given tokens as a list of tuples
	'''
	return [i for i in zip(*[tokens[i:] for i in range(n)])]

def sub_grams(n_gram):
	'''
	given an n-gram, return all the sub n-1 grams as a list of tuples, if size is 1, than return empty
	'''
	if len(n_gram) == 1:
		return list()
	
	result = list()
	for i in range(len(n_gram)):
		lst = list(n_gram)
		lst.pop(i)
		result.append(tuple(lst))
	return result


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


def load_sentences():
	
	sentences = pickle.load(open("training.pkl","rb"))
	words = list()
	tags = list()

	for sentence in sentences:
		a_sentence = list()
		sub_tags = list()

		for pairs in sentence:
			if pairs[1] == '_PUNCT_':
				continue
			else:
				a_sentence.append(pairs[0])
				sub_tags.append(pairs[1])

		words.append(a_sentence)
		tags.append(sub_tags)

	return words, tags

def count(n_gram_size):
	words, sentences = load_sentences()

	n_gram_count = dict() #N1
	sub_n_gram_count = dict() #N2
	all_A_sets = dict() #vocab size V
	id_to_A_sets = dict()
	s_to_sets = dict()

	ID = 0
	idx = 0

	for sentence in sentences:
		
		print(idx)
		idx += 1
		tokens = list()

		for a in sentence:

			if a in all_A_sets:
				tokens.append(all_A_sets[a])
			else:
				all_A_sets[a] = ID
				id_to_A_sets[ID] = a 
				tokens.append(ID)

				if type(a) == frozenset:
					for sense in a:
						if sense not in s_to_sets:
							s_to_sets[sense] = [ID]
						else:
							s_to_sets[sense].append(ID)

				ID += 1

		n_grams = ngrams(tokens, n_gram_size)

		for g in n_grams:

			g = sort(g)
			g_id = gram_id(g)

			sub_gs = sub_grams(g)
			sub_gs = [sort(sg) for sg in sub_gs]
			sub_gs = [gram_id(i) for i in sub_gs]

			if g_id in n_gram_count:
				n_gram_count[g_id] += 1
			else:
				n_gram_count[g_id] = 1

			for sg in sub_gs:
				if sg in sub_n_gram_count:
					sub_n_gram_count[sg] += 1
				else:
					sub_n_gram_count[sg] = 1

	'''
	n_gram_count = dict() #N1
	sub_n_gram_count = dict() #N2
	all_A_sets = dict() #vocab size V
	id_to_A_sets = dict()
	s_to_sets = dict()

	'''
	pickle.dump(all_A_sets, open('all_A_sets_size{}.pkl'.format(n_gram_size),'wb'))
	pickle.dump(id_to_A_sets, open('id_to_A_sets_size{}.pkl'.format(n_gram_size),'wb'))
	pickle.dump(n_gram_count,open('n_gram_count_size{}.pkl'.format(n_gram_size),'wb'))
	pickle.dump(s_to_sets, open('s_to_sets_size{}.pkl'.format(n_gram_size),'wb'))
	pickle.dump(sub_n_gram_count,open('sub_n_gram_count_size{}.pkl'.format(n_gram_size),'wb'))

def main():
	import sys
	n_gram_size = int(sys.argv[1])
	
	count(n_gram_size)


main()













