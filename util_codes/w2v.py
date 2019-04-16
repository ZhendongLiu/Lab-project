import gensim.downloader as api
from gensim.models import Word2Vec

model = Word2Vec.load('word2vec.model')

for i in model.predict_output_word(context_words_list = "coffee delicious".split()):
	print(i)
