# turn off all the logging
import logging
logging.basicConfig(level=logging.CRITICAL)

from pytrips.ontology import load as load_ontology
from pytrips.helpers import Normalize
from pytrips.tools import nlp
from nltk.corpus import wordnet as wn



#import spacy
#nlp = spacy.load("en_core_web_lg")

ont = load_ontology()

def tag_word(token):
    pos = Normalize.spacy_pos(token.pos_)
    word = token.text.lower()
    lemma = token.lemma_

    if pos not in "nvar": # if the pos is not in wordnet
        return set(),word

    # word lookup:
    wlookup = ont[("q::"+word, pos)]
    llookup = ont[("q::"+lemma, pos)]

    res = wlookup["lex"] 
    res += wlookup["wn"]
    res += llookup["lex"]
    res += llookup["wn"]

    return (frozenset(res),word)

def tag_word_wn(token):
    pos = Normalize.spacy_pos(token.pos_)
    word = token.text.lower()
    lemma = token.lemma_

    if pos not in "nvar":
        return frozenset(),word

    
    return frozenset(wn.synsets(word, pos)), word

def tag_sentence(sentence):
    sentence = nlp(sentence)
    
    return [tag_word_wn(token) for token in sentence]

