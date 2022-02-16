# -*- coding: utf-8 -*-
import torch
from modeltraining.preprocessing import load_word_vocab_from_relation_vectors, load_embeddings_filtered_byvocab, load_training_data, split_training_data

from model.RWE_Model import RWE_Model

#Initialize RWE model
def getRWEModel(embedding_size_input, embedding_size_output, embedding_weights,hidden_size,dropout):
    vocab_size=(len(embedding_weights))
    model=RWE_Model(embedding_size_input, embedding_size_output, embedding_weights,hidden_size,dropout)
    criterion = torch.nn.MSELoss()
    return model.cuda(), criterion

#Train epochs
def trainEpochs(model, optimizer, criterion, trainBatches, validBatches, epochs=10, interval=100, lr=0.1):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, threshold = 1e-7, factor = 0.9)
    min_error=-1.0
    for epoch in range(1, epochs+1):
        print("\n     ----------    \n")
        print ("EPOCH "+str(epoch))
        print ("Starting training epoch "+str(epoch))
        trainIntervals(model, optimizer, criterion, trainBatches, interval, lr)
        validErr = validate(model, validBatches, criterion)
        scheduler.step(validErr)
        print("Validation error : " + str(validErr))
        if validErr<min_error or min_error==-1.0:
            new_model=model
            min_error=validErr
            print ("[Model at epoch "+str(epoch)+" obtained the lowest development error rate so far.]")
        #if epoch%5==0 or epoch==1: torch.save(model, "./"+outputModelFile + "-epoch" + str(epoch) + ".model")
        print("Epoch " + str(epoch) + " done")
    return new_model

#Train intervals
def trainIntervals(model, optimizer, criterion, batches, interval=100, lr=0.1):
    i = 0
    n = 0
    trainErr = 0
    for x1, x2, y in zip(*batches):
        model.train(); optimizer.zero_grad()
        trainErr += gradUpdate(model, x1, x2, y, criterion, optimizer, lr)
        i += 1
        if i == interval:
            n += 1;
            prev_train_err = trainErr
            trainErr = 0
            i = 0
    if i > 0 and prev_train_err != 0:
        print("Training error: "+ str(prev_train_err / float(i)))

#Validation phase
def validate(model, batches, criterion):
    evalErr = 0
    n = 0
    model.eval()
    for x1, x2, y in zip(*batches):
        y = torch.autograd.Variable(y, requires_grad=False)
        x1 = torch.autograd.Variable(x1, requires_grad=False)
        x2 = torch.autograd.Variable(x2, requires_grad=False)
        print_in_steps('pretrainedmodel/exactinput.txt', [x1, x2])
        print_in_steps('pretrainedmodel/exactinput.txt', x1.size())
        print_in_steps('pretrainedmodel/exactinput.txt', x2.size())
        print_in_steps('pretrainedmodel/exactinput.txt', '==========')
        output = model(x1, x2)
        error = criterion(output, y)
        evalErr += error.item()
        n += 1
    return evalErr / n

#Update gradient
def gradUpdate(model, x1, x2, y, criterion, optimizer, lr):
    output = model(x1,x2)
    error = criterion(output,y)
    error.backward()
    optimizer.step()
    return error.item()

#Get batches from training set
def getBatches(data, batchSize):
    embsize = int(data.size(-1))
    return data.view(-1, batchSize, embsize) 

def print_in_steps(file, data,  mode = 'a'):
    with open(file, mode) as f:
        if(isinstance(data, list)):
            for i in range(len(data)):
                f.write(str(data[i])+'\n')
                f.write('------\n')
        else:
            f.write(str(data) + '\n')