from argparse import ArgumentParser
import sys
import random
from modeltraining.preprocessing import load_embeddings_filtered_byvocab
import torch
import json
import numpy as np
import pandas as pd
import os
import pickle

import modeltraining.train_RWE as train_RWE
from model.RWE_Model import RWE_Model
from modeltraining.Tokenizer import Tokenizer

__BENCHMARK__ = './benchmark/'
__EXAMPLENUM__ = 5
__QNUM__ = 20

def trainmodel_getembedding():
    parser = ArgumentParser()
    parser.add_argument('-word_embeddings', '--input_word_embeddings', help='Input word embeddings path', required=True)
    parser.add_argument('-rel_embeddings', '--input_relation_embeddings', help='Input relation embeddings path', required=True)
    parser.add_argument('-output', '--output_path', help='Output path to store the output relational word embeddings or pretrained model', required=True)
    parser.add_argument('-hidden', '--hidden_size', help='Size of the hidden layer (default=2xdimensions-wordembeddings)', required=False, default=0)
    parser.add_argument('-dropout', '--drop_rate', help='Dropout rate', required=False, default=0.5)
    parser.add_argument('-epochs', '--epochs_num', help='Number of epochs', required=False, default=5)
    parser.add_argument('-interval', '--interval', help='Size of intervals during training', required=False, default=100)
    parser.add_argument('-batchsize', '--batchsize', help='Batch size', default=10)
    parser.add_argument('-devsize', '--devsize', help='Size of development data (proportion with respect to the full training set, from 0 to 1)', required=False, default=0.015)
    parser.add_argument("-lr", '--learning_rate', help='Learning rate for training', required=False, default=0.01)
    parser.add_argument('-model', '--output_model', help='True for output model, False for output pretrained word embeddings', required=True, default=True)
    parser.add_argument('-hp', '--hyperparameters', help='Output path to store the output hyperparameters, until folder', required=True)

    args = vars(parser.parse_args())
    word_embeddings_path=args['input_word_embeddings']
    rel_embeddings_path=args['input_relation_embeddings']
    output_path=args['output_path']
    hidden_size=int(args['hidden_size'])
    dropout=float(args['drop_rate'])
    epochs=int(args['epochs_num'])
    interval=int(args['interval'])
    batchsize=int(args['batchsize'])
    devsize=float(args['devsize'])
    lr=float(args['learning_rate'])
    model = bool(args['output_model'])
    hp_path=args['hyperparameters']
    if devsize>=1 or devsize<0: sys.exit("Development data should be between 0% (0.0) and 100% (1.0) of the training data")

    print ("Loading word vocabulary...")
    pre_word_vocab=train_RWE.load_word_vocab_from_relation_vectors(rel_embeddings_path)
    print ("Word vocabulary loaded succesfully ("+str(len(pre_word_vocab))+" words). Now loading word embeddings...")
    matrix_word_embeddings,word2index,index2word,dims_word=train_RWE.load_embeddings_filtered_byvocab(word_embeddings_path,pre_word_vocab)
    if(type(matrix_word_embeddings).__module__=='numpy'):
        np.save(output_path + 'matrix_word_embeddings.npy', matrix_word_embeddings)
    else:
        pickle.dump(matrix_word_embeddings, open(output_path + 'matrix_word_embeddings.pkl', 'wb'))
    if(type(word2index).__module__=='numpy'):
        np.save(output_path + 'word2index.npy', word2index)
    else:
        pickle.dump(word2index, open(output_path + 'word2index.pkl', 'wb'))
    if(type(index2word).__module__=='numpy'):
        np.save(output_path + 'index2word.npy', index2word)
    else:
        pickle.dump(index2word, open(output_path + 'index2word.pkl', 'wb'))
    pre_word_vocab.clear()
    print ("Word embeddings loaded succesfully ("+str(dims_word)+" dimensions). Now loading relation vectors...")
    matrix_input,matrix_output,dims_rels=train_RWE.load_training_data(rel_embeddings_path,matrix_word_embeddings,word2index)
    print ("Relation vectors loaded ("+str(dims_rels)+" dimensions), now spliting training and dev...")
    random.seed(21)
    s1 = random.getstate()
    random.shuffle(matrix_input)
    random.setstate(s1)
    random.shuffle(matrix_output)
    matrix_input_train,matrix_output_train,matrix_input_dev,matrix_output_dev=train_RWE.split_training_data(matrix_input,matrix_output,devsize,batchsize)
    with open(output_path + 'inputtrainmatrix.txt', 'w') as f:
        for item in matrix_input_train:
            f.write(str(item))
    with open(output_path + 'inputmatrix.txt', 'w') as f:
        for item in matrix_input:
            f.write(str(item))
    # for item in matrix_output_train:
    #     with open(output_path + 'outputmatrix.txt', 'w') as f:
    #         f.write(str(item))
        
    matrix_input.clear()
    matrix_output.clear()
    print ("Done preprocessing all the data, now loading and training the model...\n")
    
    if hidden_size==0: hidden_size=dims_word*2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("Device used: "+str(device))
    embedding_weights=torch.tensor(matrix_word_embeddings)
    matrix_word_embeddings.clear()
    tensor_input_train_1=torch.LongTensor([[x[0]] for x in matrix_input_train])
    tensor_input_train_2=torch.LongTensor([[x[1]] for x in matrix_input_train])
    matrix_input_train.clear()
    tensor_input_dev_1=torch.LongTensor([[x[0]] for x in matrix_input_dev])
    tensor_input_dev_2=torch.LongTensor([[x[1]] for x in matrix_input_dev])
    matrix_input_dev.clear()
    tensor_output_train=torch.FloatTensor(matrix_output_train)
    matrix_output_train.clear()
    tensor_output_dev=torch.FloatTensor(matrix_output_dev)
    matrix_output_dev.clear()
    hyperparams = dict()
    hyperparams['hidden_size'] = hidden_size
    hyperparams['dropout'] = dropout
    # hyperparams['dims_word'] = dims_word
    # hyperparams['dims_rels'] = dims_rels
    # hyperparams['embedding_weights'] = embedding_weights
    hyperparams_json = json.dumps(hyperparams)
    with open(hp_path + 'hyperparams.json', 'w') as f:
        json.dump(hyperparams_json, f)
    torch.save(dims_word, hp_path + 'dims_word.pt')
    torch.save(dims_rels, hp_path + 'dims_rels.pt')
    torch.save(embedding_weights, hp_path + 'embedding_weights.pt')
    print("Hyperparameters saved to " + hp_path)
    model, criterion = train_RWE.getRWEModel(dims_word,dims_rels,embedding_weights,hidden_size,dropout)
    print ("RWE model loaded.")
    optimizer = torch.optim.Adam(model.parameters(), lr)
    trainX1batches = train_RWE.getBatches(tensor_input_train_1.cuda(), batchsize)
    trainX2batches = train_RWE.getBatches(tensor_input_train_2.cuda(), batchsize)
    validX1Batches = train_RWE.getBatches(tensor_input_dev_1.cuda(), batchsize)
    validX2Batches = train_RWE.getBatches(tensor_input_dev_2.cuda(), batchsize)
    trainYBatches = train_RWE.getBatches(tensor_output_train.cuda(), batchsize)
    validYBatches = train_RWE.getBatches(tensor_output_dev.cuda(), batchsize)
    print ("Now starting training...\n")
    #with open(output_path + 'training_log.txt', 'w+') as f:
    # for x1, x2, y, i in zip(trainX1batches, trainX2batches, trainYBatches, range(1)):
            #np.savetxt(f, x1.cpu().numpy())
            #f.write("\n")
            #np.savetxt(f, x2.cpu().numpy())
            #f.write("\n")
            #np.savetxt(f, y.cpu().numpy())
            #f.write("\n")
            #f.write(str(len(x1)))
            #f.write("\n")
            #f.write(str(len(x2)))
            #f.write("\n")
            #f.write(str(len(y)))
            #f.write("\n")
            #f.write("=============================")
            #f.write("\n")
        # torch.save(x1, output_path + 'x1Tensor.pt')
        # torch.save(x2, output_path + 'x2Tensor.pt')
        # torch.save(y, output_path + 'yTensor.pt')
    output_model=train_RWE.trainEpochs(model, optimizer, criterion, (trainX1batches, trainX2batches, trainYBatches), (validX1Batches, validX2Batches, validYBatches), epochs, interval, lr)
    print ("\nTraining finished. Now loading relational word embeddings from trained model...")


    if(output_model):
        print ("\nSaving model...")
        torch.save(model.state_dict(), hp_path + 'model_state.pt')
        torch.save(model, hp_path + 'pretrained_model.pt')
        print ("Model saved to " + hp_path)
        print ("\nModel saved succesfully.")
    else:
        parameters=list(output_model.parameters())
        num_vectors=len(parameters[0])
        print ("Number of vectors: "+str(num_vectors))
        num_dimensions=len(parameters[0][0])
        print ("Number of dimensions output embeddings: "+str(num_dimensions))
        txtfile=open(output_path,'w',encoding='utf8')
        txtfile.write(str(num_vectors)+" "+str(num_dimensions)+"\n")
        if num_vectors!=embedding_weights.size()[0]: print ("Something is wrong in the input vectors: "+str(embedding_weights.size()[0])+" != "+str(num_vectors))
        for i in range(num_vectors):
            word=index2word[i]
            txtfile.write(word)
            vector=parameters[0][i].cpu().detach().numpy()
            for dimension in vector:
                txtfile.write(" "+str(dimension))
            txtfile.write("\n")
        txtfile.close()
        print ("\nFINISHED. Word embeddings stored at "+output_path)

    print('Testing with small DS')
    testword1 = 'germany'
    testword2 = 'german'
    testword3 = 'france'
    testword4 = 'french'
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    emb1 = model(torch.LongTensor([[word2index[testword1]]]).cuda(), torch.LongTensor([[word2index[testword2]]]).cuda())
    emb2 = model(torch.LongTensor([[word2index[testword3]]]).cuda(), torch.LongTensor([[word2index[testword4]]]).cuda())
    with open('pretrainedmodel/embs.txt', 'w') as f:
        f.write(str(emb1))
        f.write("\n")
        f.write(str(emb2))
    print(word2index[testword1])
    print(word2index[testword2])
    print(word2index[testword3])
    print(word2index[testword4])
    print(cos(emb1, emb2))

def loadmodel_calculateembedding():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = ArgumentParser()
    parser.add_argument('-im', '--input_model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('-hidden', '--hidden_size', help='Size of the hidden layer (default=2xdimensions-wordembeddings)', required=False, default=0)
    parser.add_argument('-dropout', '--drop_rate', help='Dropout rate', required=False, default=0.5)
    parser.add_argument('-epochs', '--epochs_num', help='Number of epochs', required=False, default=5)
    parser.add_argument('-interval', '--interval', help='Size of intervals during training', required=False, default=100)
    parser.add_argument('-batchsize', '--batchsize', help='Batch size', default=10)
    parser.add_argument('-devsize', '--devsize', help='Size of development data (proportion with respect to the full training set, from 0 to 1)', required=False, default=0.015)
    parser.add_argument("-lr", '--learning_rate', help='Learning rate for training', required=False, default=0.01)
    args = vars(parser.parse_args())

    hidden_size=int(args['hidden_size'])
    dropout=float(args['drop_rate'])
    epochs=int(args['epochs_num'])
    interval=int(args['interval'])
    batchsize=int(args['batchsize'])
    devsize=float(args['devsize'])
    lr=float(args['learning_rate'])

    # model = RWE_Model()
    # model.load_state_dict(torch.load(args['input_model']))
    dims_word = torch.load('pretrainedmodel/dims_word.pt')
    dims_rels = torch.load('pretrainedmodel/dims_rels.pt')
    embedding_weights = torch.load('pretrainedmodel/embedding_weights.pt')
    # model, criterion = train_RWE.getRWEModel(dims_word,dims_rels,embedding_weights,hidden_size,dropout)
    # model.load_state_dict(torch.load(args['input_model']))
    model = torch.load(args['input_model'])
    model.eval()
    matrix_word_embeddings = pickle.load(open('pretrainedmodel/RWE.modelmatrix_word_embeddings.pkl', 'rb'))
    word2index = pickle.load(open('pretrainedmodel/RWE.modelword2index.pkl', 'rb'))
    # print(model.state_dict())

    testfile = 'CountryToLanguage.csv'
    dataTable = pd.read_csv(os.path.join(__BENCHMARK__, testfile))
    example = dataTable.iloc[:__EXAMPLENUM__, :]
    XList = example.iloc[:, :-1].values.tolist()
    XList = [item for sublist in XList for item in sublist]
    Y = example.iloc[:, -1].values.tolist()
    Q = dataTable.iloc[__EXAMPLENUM__:__QNUM__, :-1].values.tolist()
    QY = dataTable.iloc[__EXAMPLENUM__:__QNUM__, -1].values.tolist()
    Q = [item for sublist in Q for item in sublist]

    tokenizer = Tokenizer()

    XList = [tokenizer.tokenize(sent) for sent in XList]
    Y = [tokenizer.tokenize(sent) for sent in Y]
    Q = [tokenizer.tokenize(sent) for sent in Q]

    # print(XList)
    # print(Y)
    # print(Q)

    vocab = set()
    for x in XList:
        vocab.add(x)
    for y in Y:
        vocab.add(y)
    print(XList)
    print(Y)
    print(vocab)
    inputpath = 'traindata/RWE_default.txt'
    # matrix_word_embeddings,word2index,index2word,dimensions = load_embeddings_filtered_byvocab(inputpath, vocab)
    # print(matrix_word_embeddings[9])
    # print(word2index)
    # print(word2index.keys()[:10])
    # i = 0
    # for key in word2index.keys():
    #     print(key, word2index[key])
        # i += 1
        # if i == 10:
        #     break
    # print(word2index)
    # print(len(matrix_word_embeddings))

    tensor1 = torch.Tensor(matrix_word_embeddings[word2index[XList[0]]]).cuda().view(-1, 300)
    tensor2 = torch.Tensor(matrix_word_embeddings[word2index[Y[0]]]).cuda().view(-1, 300)
    # tensor1 = torch.LongTensor(matrix_word_embeddings[5]).cuda().view(-1, 300)
    # tensor2 = torch.LongTensor(matrix_word_embeddings[6]).cuda().view(-1, 300)
    # tensor1 = model.word_embeddings(torch.LongTensor(word2index[XList[0]])).cuda()
    # tensor2 = model.word_embeddings(torch.LongTensor(word2index[Y[0]])).cuda()
    # tensor1 = torch.autograd.Variable(tensor1, requires_grad = False).to(torch.cuda.DoubleTensor)
    # tensor2 = torch.autograd.Variable(tensor2, requires_grad = False).to(torch.cuda.DoubleTensor)
    rel1 = model(tensor1, tensor2)
    rel1 = [tensor1, tensor2]

    tensor3 = torch.Tensor(matrix_word_embeddings[word2index[XList[1]]]).cuda().view(-1, 300)
    tensor4 = torch.Tensor(matrix_word_embeddings[word2index[Y[1]]]).cuda().view(-1, 300)
    # tensor3 = torch.LongTensor(matrix_word_embeddings[1]).cuda().view(-1, 300)
    # tensor4 = torch.LongTensor(matrix_word_embeddings[9]).cuda().view(-1, 300)
    # tensor3 = model.word_embeddings(torch.LongTensor(word2index[XList[1]])).cuda()
    # tensor4 = model.word_embeddings(torch.LongTensor(word2index[Y[1]])).cuda()
    # tensor3 = torch.autograd.Variable(tensor3, requires_grad = False).to(torch.cuda.DoubleTensor)
    # tensor4 = torch.autograd.Variable(tensor4, requires_grad = False).to(torch.cuda.DoubleTensor)
    rel2 = model(tensor3, tensor4)
    rel2 = [tensor3, tensor4]

    # print(rel1.shape)
    # print(rel2.shape)
    # print(tensor1)
    # print(tensor2)
    # print(tensor3)
    # print(tensor4)
    # print(model(rel1, rel2))
    # print(len(matrix_word_embeddings[5]))
    # print(len(matrix_word_embeddings[6]))
    # print(len(matrix_word_embeddings[1]))
    # print(len(matrix_word_embeddings[9]))

    cos = torch.nn.CosineSimilarity(dim = 0, eps = 1e-6)
    print(cos(rel1, rel2))
    # print(rel1)
    

def checkTensor():
    __PATH__ = './pretrainedmodel/'
    __FILE_PREF__ = 'RWE.model'
    __FILE_SUFF__ = 'Tensor.pt'
    filelist = ['dims_rels.pt', 'dims_word.pt']

    for file in filelist:
        tensor = torch.load(__PATH__ + file)
        print(file + ' tensor size: ' + str(tensor))

def main():
    trainmodel_getembedding()

if __name__ == '__main__':
    trainmodel_getembedding()
    # loadmodel_calculateembedding()
