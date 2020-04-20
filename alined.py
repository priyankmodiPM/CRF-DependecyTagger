from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable, Function
from torch import autograd

import math

import time
import _pickle as cPickle

import urllib
import os
import sys
from tqdm import tqdm
import codecs
import re
import numpy as np

parameters = OrderedDict()
parameters['tag_scheme'] = "BIO"
parameters['lower'] = True
parameters['zeros'] = False
parameters['char_dim'] = 60
parameters['word_dim'] = 300
parameters['word_lstm_dim'] = 300
parameters['all_emb'] = 1
parameters['crf'] = 0 #Disabled 
parameters['dropout'] = 0.3
parameters['epoch'] = 20
parameters['weights'] = ""
parameters['name'] = sys.argv[3]
parameters['gradient_clip'] = 5.0
parameters['char_mode'] = "CNN"
parameters['languages'] = int(sys.argv[2])
parameters['lambd'] = 0.5 		# For the gradient reversal layer
models_path = "./models/"

parameters['use_gpu'] = torch.cuda.is_available()

use_gpu = parameters['use_gpu']
print(use_gpu)
parameters['reload'] = "./models/pre-trained-model"

START_TAG = '<s>'
STOP_TAG  = '</s>'
unk = 'unk'

mapping_file = "../data/mapping.pkl"

name = parameters['name']
model_name = models_path + name

if not os.path.exists(models_path):
    os.makedirs(models_path)

class Model_Data:
    def __init__(self, name):
        self.name = name
        self.all_words = []
        self.all_tags = []
        self.lang_vocab = {}
        self.joint_embeds = {}
        self.max_length = 0
        self.train_data, self.train_targets, self.valid_data, self.valid_targets = [], [], [], []
        self.test_data, self.test_targets = [], []
        self.word2index , self.muse_eng, self.muse_ita, self.muse_esp = {}, {}, {}, {}
        self.n_words = 0	
        self.index2word = {}
        self.char2index = {}
        self.index2char = {}
        self.tag2index = {}
        self.index2tag = {}

def load_saved_data(model_name):
    model = Model_Data("Dataset")
    model = torch.load(model_name)
    print("[INFO] -> Train Data Sentences.")
    print(len(model.train_data))
    print("[INFO] -> Valid Data Sentences.")
    print(len(model.valid_data))
    print("[INFO] -> Test Data Sentences.")
    print(len(model.test_data))
    return model

def lower_case(x, lower=False):
    if lower:
        return x.lower()
    else:
        return x

def prepare_dataset(sentences, targets, word_to_id, char_to_id, tag_to_id, dtype, lower=False):
    data = []
    for index, s in enumerate(sentences):
        if s[0] == '':
            continue
        str_words = s[0].split()
        words = []
        # print(str_words)
        # chars = [[char_to_id[c] for c in w.lower()] for w in str_words]
        chars = []
        for w in str_words:
            if w.lower()[0] not in char_to_id.keys():
                continue
            else:
                chars.append([char_to_id[c] for c in w])
                words.append(word_to_id[w if w in word_to_id else unk])
        tags = [tag_to_id[tag] for tag in targets[index].split(" ")]
        # if dtype != "test":
        # 	lang = [lang2index[s[1]] for w in str_words]
        # else:
        # 	lang = [-1 for w in str_words]
        data.append({
            'str_words': str_words,
            'words' : words,
            'chars' : chars,
            'tags'	: tags,
            # 'lang'  : lang
            })
    return data

def index_to_embed(word_embeds, word_to_id, all_embeds):
    for word in word_to_id:
        if word in all_embeds:
            word_embeds[word_to_id[word]] = all_embeds[word]

    return word_embeds


''' Model Games Begin '''
def init_embedding(input_embedding):
    bias = np.sqrt(3.0/input_embedding.size(1))
    '''
    We start with the init_embedding function, which just initializes the 
    embedding layer by pooling from a random sample.
    '''
    nn.init.uniform_(input_embedding, -bias, bias)

def init_linear(input_linear):
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):
    '''
    weight_ih_l[k] : input-hidden weights of kth layer 
    if k = 0 --> shape : (hidden_size * input_size) 
    else     --> shape : (hidden_size * hidden_size)

    weight_hh_l[k] : hidden-hidden weights of kth layer
    shape : (hidden_size * hidden_size)
    '''
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        sampling_range = np.sqrt(6.0/(weight.size(0)/ 4+weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)

        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0/(weight.size(0)/ 4+weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)

    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0/(weight.size(0)/ 4+weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0/(weight.size(0) / 4+weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2*input_lstm.hidden_size] = 1
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2*input_lstm.hidden_size] = 1

        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2*input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2*input_lstm.hidden_size] = 1

class Highway(nn.Module):

    def __init__(self, inp_dim, out_dim, cuda=False):
        super(Highway, self).__init__()
        in_feas = inp_dim
        D = inp_dim
        self.cuda = False

        self.fc1 = self.init_Linear(in_fea=in_feas, out_fea=D, bias=True)
        # Highway gate layer  T in the Highway formula
        self.gate_layer = self.init_Linear(in_fea=in_feas, out_fea=D, bias=True)

        #new crap
        self.dropout = nn.Dropout(0.20)
        self.linear_layer = self.init_Linear(in_fea=in_feas, out_fea=out_dim, bias=True)


    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x):
        # normal layer in the formula is H
        normal_fc = torch.tanh(self.fc1(x))
        # transformation gate layer in the formula is T
        transformation_layer = torch.sigmoid(self.gate_layer(x))
        # carry gate layer in the formula is C
        carry_layer = 1 - transformation_layer
        # formula Y = H * T + x * C
        allow_transformation = torch.mul(normal_fc, transformation_layer)
        allow_carry = torch.mul(x, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)
        
        #new crap
        inf = self.dropout(information_flow)
        inf_flow = self.linear_layer(inf)
        return inf_flow


def init_embedding(input_embedding):
    bias = np.sqrt(3.0/input_embedding.size(1))
    '''
    We start with the init_embedding function, which just initializes the 
    embedding layer by pooling from a random sample.
    '''
    nn.init.uniform_(input_embedding, -bias, bias)

def argmax(vec):
    '''
    Max index in a vector
    '''
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def to_scalar(var):
    '''
    Pytorch tensor to scalar
    '''
    return var.view(-1).data.tolist()[0]

def get_lstm_features(self, sentence, chars2, chars2_length):
    
    self.hidden_w = self.init_hidden('w')
    self.hidden_c = self.init_hidden('c')

    chars_embeds = self.char_embeds(chars2).unsqueeze(1)
    chars_cnn_out3 = self.char_cnn3(chars_embeds)
    chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)
    chars_embeds = self.char_highway(chars_embeds) #new crap
    chars_embeds = chars_embeds.unsqueeze(1)
    
    # print(sentence)
    word_embeds = self.word_embeds(sentence)
    word_embeds = word_embeds.unsqueeze(1)  
    ## Word lstm
    ## Takes words as input and generates a output at each step
    
    lstm_out_w_old, self.hidden_w = self.lstm_w(word_embeds, self.hidden_w)
    lstm_out_c_old, self.hidden_c = self.lstm_c(chars_embeds, self.hidden_c)

    final_hidden = torch.cat((self.hidden_c[1],self.hidden_w[1]),dim=2)
    
    ## Reshaping the outputs from the lstm layer
    lstm_out_w = lstm_out_w_old.view(len(sentence), self.hidden_dim*2)
    lstm_out_c = lstm_out_c_old.view(len(sentence), (self.hidden_dim*2)//5)

    ## Add a highway network here to combine the outputs

    lstm_out = self.highway(torch.cat((lstm_out_w, lstm_out_c), dim=1))
    '''
    Insert Attention here
    '''
    lstm_feats = self.hidden2tag(lstm_out)
    
    return lstm_feats, lstm_out

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class Parallel_LSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, lang_num, char_to_ix, pre_word_embeds, char_out_dimension=40,char_embedding_dim=40, use_gpu=True,use_crf=True, char_mode='CNN'):
        '''
        Input parameters:
                vocab_size= Size of vocabulary (int)
                tag_to_ix = Dictionary that maps NER tags to indices
                embedding_dim = Dimension of word embeddings (int)
                hidden_dim = The hidden dimension of the LSTM layer (int)
                char_to_ix = Dictionary that maps characters to indices
                pre_word_embeds = Numpy array which provides mapping from word embeddings to word indices
                char_out_dimension = Output dimension from the CNN encoder for character
                char_embedding_dim = Dimension of the character embeddings
                use_gpu = defines availability of GPU, 
                    when True: CUDA function calls are made
                    else: Normal CPU function calls are made
                use_crf = parameter which decides if you want to use the CRF layer for output decoding
        '''
        
        super(Parallel_LSTM_CRF, self).__init__()
        #parameter initialization for the model
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension
        self.char_mode = char_mode
        self.lang_num = lang_num

        self.char_embedding_dim = char_embedding_dim
        self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
        init_embedding(self.char_embeds.weight)
        
       
        
        #Performing CNN encoding on the character embeddings

        self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))
        self.char_highway = Highway(self.out_channels, self.out_channels, self.use_gpu) #new crap

        #Creating Embedding layer with dimension of ( number of words * dimension of each word)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            #Initializes the word embeddings with pretrained word embeddings
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False
    
        #Initializing the dropout layer, with dropout specificed in parameters
        self.dropout = nn.Dropout(parameters['dropout'])
        
        #Lstm Layer:
        #input dimension: word embedding dimension + character level representation
        #bidirectional=True, specifies that we are using the bidirectional LSTM

        self.lstm_c = nn.LSTM(self.out_channels, hidden_dim//5, bidirectional=True)

        self.lstm_w = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        
        #Initializing the lstm layer using predefined function for initialization
        init_lstm(self.lstm_c)
        init_lstm(self.lstm_w)
        
        self.highway = Highway(2*hidden_dim + 2*hidden_dim//5, hidden_dim//2, True)

        # Linear layer which maps the output of the bidirectional LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim//2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        print(tag_to_ix[STOP_TAG])
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        #Initializing the linear layer using predefined function for initialization
        init_linear(self.hidden2tag) 

        self.hidden_w = self.init_hidden('w')
        self.hidden_c = self.init_hidden('c')


    def init_hidden(self, t):
        if t == 'w':
            if self.use_gpu:
                return (torch.randn(2, 1, self.hidden_dim).cuda(),torch.randn(2, 1, self.hidden_dim).cuda())
            else:
                return (torch.randn(2, 1, self.hidden_dim),torch.randn(2, 1, self.hidden_dim))
        elif t == 'c':
            if self.use_gpu:
                return (torch.randn(2, 1, self.hidden_dim//5).cuda(),torch.randn(2, 1, self.hidden_dim//5).cuda())
            else:
                return (torch.randn(2, 1, self.hidden_dim//5),torch.randn(2, 1, self.hidden_dim//5))



    #assigning the functions, which we have defined earlier
    _get_lstm_features = get_lstm_features

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        if self.use_gpu:
            init_alphas = torch.full((1, self.tagset_size), -10000.).cuda()
        else:
            init_alphas = torch.full((1, self.tagset_size), -10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        if self.use_gpu:
            score = torch.zeros(1).cuda()
            tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).cuda(), tags])
        else:
            score = torch.zeros(1)
            tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        if self.use_gpu:
            init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
        else:
            init_vvars = torch.full((1, self.tagset_size), -10000.)

        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, chars2, chars2_length,):
        feats, _ = self._get_lstm_features(sentence, chars2, chars2_length)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, chars2, chars2_length):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats, _ = self._get_lstm_features(sentence, chars2, chars2_length)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES
    
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    
    # We assume by default the tags lie outside a named entity
    # print(tags)
    default = tags["rh"]
    
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    
    chunks = []
    
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                # Initialize chunk for each entity
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                # If chunk class is B, i.e., its a beginning of a new named entity
                # or, if the chunk type is different from the previous one, then we
                # start labelling it as a new entity
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def evaluating(model, datas, best_F, best_P, best_R, data_model, out_file, dataset="Train"):
    '''
    The function takes as input the model, data and calcuates F-1 Score
    It performs conditional updates 
     1) Flag to save the model 
     2) Best F-1 score
    ,if the F-1 score calculated improves on the previous F-1 score
    '''
    # Initializations
    prediction = [] # A list that stores predicted tags
    save = False # Flag that tells us if the model needs to be saved
    new_F = 0.0 # Variable to store the current F1-Score (may not be the best)
    correct_preds, total_correct, total_preds = 0., 0., 0. # Count variables
    
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        
        if parameters['char_mode'] == 'CNN':
            d = {} 
            # Padding the each word to max word size of that sentence
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        
        # We are getting the predicted output from our model
        if use_gpu:
            val,out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length)
        else:
            val,out = model(dwords, chars2_mask, chars2_length)
        predicted_id = out
    
        
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks	  = set(get_chunks(ground_truth_id, data_model.tag2index))
        lab_pred_chunks = set(get_chunks(predicted_id, data_model.tag2index))

        # Updating the count variables
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds   += len(lab_pred_chunks)
        total_correct += len(lab_chunks)
    
    # Calculating the F1-Score
    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0

    if dataset == "Test":
        out_file.write("Data: " + str(dataset) + " New_F1 Score: " + str(new_F) + "\n")
        out_file.write("Best_F1 Score: " + str(best_F) + " Best Precision: " + str(best_P) + " Best Recall: " + str(best_R) + "\n\n")
    
    # If our current F1-Score is better than the previous best, we update the best
    # to current F1 and we set the flag to indicate that we need to checkpoint this model
    
    if new_F > best_F:
        best_F = new_F
        best_P = p
        best_R = r
        save = True

    return best_F, best_P, best_R, new_F, save

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(argv):
    data_model_name = sys.argv[1]
    out_file = open("./event_results/no_discrim_" + data_model_name, 'w')
    print(data_model_name)
    data_model = load_saved_data('./data_models/' + data_model_name)
    train_data = prepare_dataset(data_model.train_data, data_model.train_targets, data_model.word2index, data_model.char2index, data_model.tag2index, "train", parameters['lower'])
    dev_data = prepare_dataset(data_model.valid_data, data_model.valid_targets, data_model.word2index, data_model.char2index, data_model.tag2index, "val", parameters['lower'])
    test_data = prepare_dataset(data_model.test_data, data_model.test_targets, data_model.word2index, data_model.char2index, data_model.tag2index, "test", parameters['lower'])

    out_file.write("Sentences in train / dev  : " + str(len(train_data)) + " : " + str(len(dev_data)) + "\n")
    
    # Initialize Embedding Matrix

    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(data_model.word2index), parameters['word_dim']))
    word_embeds = index_to_embed(word_embeds, data_model.word2index, data_model.joint_embeds)

    data_model.tag2index[START_TAG] = 54
    data_model.tag2index[STOP_TAG] = 55
    print(data_model.tag2index)
    # print((data_model.word2index))
    model = Parallel_LSTM_CRF(vocab_size=len(data_model.word2index)+1,
                   tag_to_ix=data_model.tag2index,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   lang_num=parameters['languages'],
                   use_gpu=use_gpu,
                   char_to_ix=data_model.char2index,
                   pre_word_embeds=word_embeds,
                   use_crf=parameters['crf'],
                   char_mode=parameters['char_mode'])

    if use_gpu:
        model = model.cuda()

    learning_rate = 0.015
    momentum = 0.9
    number_of_epochs = parameters['epoch']
    decay_rate = 0.01
    gradient_clip = parameters['gradient_clip']
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)

    losses = []
    loss = 0.0
    best_dev_F = -1.0
    best_dev_P = -1.0
    best_dev_R = -1.0
    best_test_F = -1.0
    best_test_P = -1.0
    best_test_R = -1.0
    best_train_F = -1.0
    best_train_R = -1.0
    best_train_P = -1.0
    all_F = [[0, 0, 0]]
    eval_every = len(train_data)
    count = 0
    plot_every = len(train_data)

    tr = time.time()
    model.train(True)
    for epoch in range(1, number_of_epochs):
        out_file.write("Epoch :" + str(epoch) + "\n")
        print("EP: ", epoch)
        for index in np.random.permutation(len(train_data)):
            count += 1
            data = train_data[index]

            model.zero_grad()

            sentence_in = data['words']
            # print(data['str_words'])
            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']
            chars2 = data['chars']
            # langs = data['lang']

            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for m, c in enumerate(chars2):
                chars2_mask[m, :chars2_length[m]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

            targets = torch.LongTensor(tags)

            # lang_target = torch.LongTensor(langs)

            if use_gpu:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), chars2_length)
            else:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length)

            loss += neg_log_likelihood.data[0]/len(data['words'])
            neg_log_likelihood.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            if count % plot_every == 0:
                loss /= plot_every
                out_file.write(str(count) +  " : " + str(loss) + "\n")
                if losses == []:
                    losses.append(loss)
                losses.append(loss)
                loss = 0.0

            if count % (eval_every) == 0:
                model.train(False)
                #best_train_F, best_train_P, best_train_R, new_train_F, _ = evaluating(model, train_data, best_train_F, best_train_P, best_train_R, data_model, out_file, "Train")
                best_dev_F, best_dev_P, best_dev_R, new_dev_F, save  = evaluating(model, dev_data, best_dev_F, best_dev_P, best_dev_R, data_model, out_file, "Dev")
                best_test_F, best_test_P, best_test_R, new_test_F, _ = evaluating(model, test_data, best_test_F, best_test_P, best_test_R , data_model, out_file,  "Test")

                if save:
                    print("Saving Model to " + model_name)
                    print("Best Dev F1: ", best_dev_F)
                    print("Best Dev Precision: ", best_dev_P)
                    print("Best Dev Recall: ", best_dev_R)
                    print()
                    print("Best Test F1: ", best_test_F)
                    print("Best Test Precision: ", best_test_P)
                    print("Best Test Recall: ", best_test_R)
                    print("New F: ", new_test_F)
                    print()
                    torch.save(model.state_dict(), model_name)

                model.train(True)

            if count % len(train_data) == 0:
                adjust_learning_rate(optimizer, lr=learning_rate/(1+decay_rate*count/len(train_data)))

        print("Total Time Elapsed :", time.time()-tr, "\n")
    out_file.close()
    exit(0)

main(sys.argv)
