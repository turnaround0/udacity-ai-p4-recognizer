import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for idx in range(test_set.num_items):
        X, lengths = test_set.get_item_Xlengths(idx)

        best_logL = float('-inf')
        best_word = str()
        words_logL = dict()
        
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
            except Exception as e:
                logL = float('-inf')

            words_logL[word] = logL

            if logL > best_logL:
                best_logL, best_word = logL, word
                
        probabilities.append(words_logL)
        guesses.append(best_word)
        
    return probabilities, guesses
