from argparse import ArgumentParser
import sys
import random
import torch
import json

import modeltraining.train_RWE as train_RWE
from model.RWE_Model import RWE_Model

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
    output_model=train_RWE.trainEpochs(model, optimizer, criterion, (trainX1batches, trainX2batches, trainYBatches), (validX1Batches, validX2Batches, validYBatches), epochs, interval, lr)
    print ("\nTraining finished. Now loading relational word embeddings from trained model...")


    if(output_model):
        print ("\nSaving model...")
        torch.save(model.state_dict(), output_path)
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

def loadmodel_calculateembedding():
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

    model = RWE_Model()
    model.load_state_dict(torch.load(args['input_model']))
    print(model.state_dict())


def main():
    trainmodel_getembedding()

if __name__ == '__main__':
    trainmodel_getembedding()