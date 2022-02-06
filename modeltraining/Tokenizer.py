from utils.Singleton import SingletonMeta

import nltk

class Tokenizer(metaclass = SingletonMeta):
    def __init__(self) -> None:
        pass

    def tokenize(self, text):
        stopwords = ['a','the','of','on','in','an','and','is','at','are','as','be','but','by','for','it','no','not','or',
                     'such','that','their','there','these','to','was','with','they','will',  'v', 've', 'd']#, 's']
        punct = [',', '.', '!', ';', ':', '?', "'", '"']
        tok = ' '.join([w for w in nltk.tokenize.casual_tokenize(text, preserve_case = False) if w not in stopwords])
        sent = tok.split(' ')
        if(len(sent) == 1):
            tok = sent[0]
        else:
            tok = '_'.join(sent)
        # cleaned = re.sub('[\W_]+', ' ', str(text).encode('ascii', 'ignore').decode('ascii')).lower()
        # feature_one = re.sub(' +', ' ', cleaned).strip()
        
        # for x in stopwords:
        #     feature_one = feature_one.replace(' {} '.format(x), ' ')
        #     if feature_one.startswith('{} '.format(x)):
        #         feature_one = feature_one[len('{} '.format(x)): ]
        #     if feature_one.endswith(' {}'.format(x)):
        #         feature_one = feature_one[:-len(' {}'.format(x))]
        return tok
