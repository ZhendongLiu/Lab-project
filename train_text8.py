from gensim.summarization import textcleaner

text_file = open('text')
text = text_file.read()

sentences = textcleaner.get_sentences(text)

