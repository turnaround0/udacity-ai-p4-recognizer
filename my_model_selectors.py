import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    [parameters]
    L: the likelihood of the fitted model
    p: the number of parameters
    N: the number of data points (= sample size)
    -2logL term: decreases with increasing model complexity (more parameters)
    plogN term: the penalties, increase with increasing complexity
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_bic_score = float('inf')
        best_model = self.base_model(self.n_constant)

        # num_states: for n between self.min_n_components and self.max_n_components
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(num_states)

            # logL: log(the likelihood of the fitted model)
            try:
                logL = model.score(self.X, self.lengths)
            except Exception as e:
                continue

            # N: the number of data points (= sample size)
            N = sum(self.lengths)

            # p: the number of free parameters
            # http://hmmlearn.readthedocs.io/en/latest/api.html
            # Attributes of GaussianHMM
            #  transmat_: (array, shape (n_components, n_components)) Matrix of transition probabilities between states.
            #   since they add up to 1.0, the last row can be calculated from others,
            #   so it is n_components * (n_components - 1).
            #  startprob_: (array, shape (n_components, )) Initial state occupation distribution.
            #   since they add up to 1.0, it is (n_components - 1).
            #  means_: (array, shape (n_components, n_features)) Mean parameters for each state.
            #  covars_: (array) Covariance parameters for each state. (n_components, n_features) if “diag”
            # p = #transmat_ + #startprob_ + #means_ + #covars_
            #   = n_components * (n_components - 1) + n_components - 1 + n_components * n_features + n_components * n_features
            p = num_states ** 2 + 2 * num_states * model.n_features - 1

            # BIC = -2 * logL + p * logN
            bic_score = -2 * logL + p * np.log(N)

            if bic_score < best_bic_score:
                best_bic_score, best_model = bic_score, model
    
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_dic_score = float('-inf')
        best_model = self.base_model(self.n_constant)

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(num_states)
            try:
                logL = model.score(self.X, self.lengths)
            except Exception as e:
                # if cannot calculate this word, there is no reason to get anti-likelihood term.
                continue

            # getting average of anti-likelihood
            num_antiLogL, sum_antiLogL = 0, 0
            for word in self.words:
                if word != self.this_word:
                    X, lengths = self.hwords[word]
                    try:
                        sum_antiLogL += model.score(X, lengths)
                        num_antiLogL += 1
                    except Exception as e:
                        continue

            if num_antiLogL:
                dic_score = logL - (sum_antiLogL / num_antiLogL)
            else:
                dic_score = float('-inf')

            if dic_score > best_dic_score:
                best_dic_score, best_model = dic_score, model

        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        num_split = 3
        best_logL = float('-inf')
        best_model = self.base_model(self.n_constant)

        # if number of sample is lower than number of split, it is impossible to split
        if len(self.sequences) < num_split:
            return best_model      

        # class sklearn.model_selection.KFold(n_splits=3, shuffle=False, random_state=None)
        #   Provides train/test indices to split data in train/test sets.
        #   method: split(X, y=None, groups=None)
        #     Generate indices to split data into training and test set.
        kf = KFold(n_splits=num_split)

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            sum_logL, num_tries = 0, 0

            for train_index, test_index in kf.split(self.sequences):
                # method: combine_sequences(split_index_list, sequences)
                #   concatenate sequences referenced in an index list and returns tuple of the new X,lengths
                #   returns: tuple of list, list in format of X, lengths use in hmmlearn
                X_train, X_train_len = combine_sequences(train_index, self.sequences)
                X_test, X_test_len  = combine_sequences(test_index, self.sequences)

                # X, lengths represent training set of base_model
                backup_X, backup_lengths = self.X, self.lengths
                self.X, self.lengths = X_train, X_train_len

                # re-establish hmm model using new training data
                train_model = self.base_model(num_states)
                self.X, self.lengths = backup_X, backup_lengths
                
                # logL: log Likelihood of cross-validation folds
                try:
                    sum_logL += train_model.score(X_test, X_test_len)
                    num_tries += 1
                except Exception as e:
                    continue

            # based on average log Likelihood of cross-validation folds
            if num_tries > 0:
                avg_logL = sum_logL / num_tries
            else:
                avg_logL = float('-inf')
            
            if avg_logL > best_logL:
                # this hmm model should reflect all training data
                model = self.base_model(num_states)
                best_logL, best_model = avg_logL, model
        
        return best_model
