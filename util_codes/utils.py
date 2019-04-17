from pytrips.tools import nlp
from util_codes import simple_tagger as st

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

def pre_process_sent(sent):
	
	token_to_a_set = tag_a_sentence(sent)
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

def tag_a_sentence(sentence):
	words_with_ontologies = {}
	tagging = st.tag_sentence(sentence)

	for j, i in tagging:
		if str(i) not in words_with_ontologies:
			j = frozenset([str(s) for s in j])

			words_with_ontologies[str(i)] = j
	return words_with_ontologies

def tag_a_sentence_wn(sentence):
	pass



