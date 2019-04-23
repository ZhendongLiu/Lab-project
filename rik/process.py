import calendar, time
from pytrips.tools import nlp
from pytrips.helpers import Normalize
from collections import defaultdict, Counter
import json


nlp.disable_pipes("ner")
nlp.disable_pipes("parser")

nvar = "nvar"


class CTR:
    def __init__(self, count=None):
        if count:
            self.__count = count
        else:
            self.__count = Counter()

    @staticmethod
    def tokenform(token):
        pos = Normalize.spacy_pos(token.pos_)
        tag = token.tag_
        lemma = token.lemma_
        return "{}.{}.{}".format(lemma,tag,pos)

    def count(self, token):
        token = CTR.tokenform(token)
        if token[-1] in nvar:
            self.__count[token] += 1
        return token

    def get_count(self, token):
        return self.__count[CTR.tokenform(token)]

    def pruned(self, thresh=40, lower=False):
        new_count = Counter()
        if lower:
            low = lambda x: x.lower()
        else:
            low = lambda x: x
        for key in self.__count.keys():
            if self.__count[key] > thresh:
                new_count[low(key)] += self.__count[key]
        json.dump(new_count, open('counter-thresh{}.txt'.format(str(thresh)), 'w'))

    def sentence(self, sent):
        return " ".join([self.count(t) for t in sent])
            

def estimate(target, i, seconds):
    hrs = str(((target-i) * (seconds/i))/3600) + ".00"
    hrs = hrs.split(".")
    return hrs[0] + "." + hrs[1][:3]

def main(data):
    ctr = CTR()
    i = 0
    ct = calendar.timegm(time.gmtime())
    pt = calendar.timegm(time.gmtime())
    output = open("tagged.txt", 'w')
    with open(data, "r") as text:
        for line in text:
            i += 1
            sent = nlp(line)
            output.write(ctr.sentence(sent) + "\n")
            ct = calendar.timegm(time.gmtime())
            if i % 1000 == 0:
                print("                                                        \r", end="")
                print("[{}]{}".format(str(estimate(8467707, i, ct - pt)), str(i)), end="")
    output.close()
    return ctr

if __name__ == "__main__":
    ctr = main('smltext')

    ctr.pruned(20)
    ctr.pruned(30)
    ctr.pruned(40)
    ctr.pruned(50)
