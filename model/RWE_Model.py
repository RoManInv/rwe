import torch

#RWE model
class RWE_Model(torch.nn.Module):
    def __init__(self, embedding_size_input, embedding_size_output, embedding_weights,hidden_size,dropout):
        super(RWE_Model, self).__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(embedding_weights).float()
        self.embeddings.weight.requires_grad = True
        self.linear1 = torch.nn.Linear(embedding_size_input*2, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden_size, embedding_size_output)
    def forward(self,input1,input2):
        embed1 = self.embeddings(input1)
        embed2 = self.embeddings(input2)
        out = self.linear1(torch.cat(((embed1*embed2), (embed1+embed2)/2), 2)).squeeze()
        out = self.relu(out)
        out = self.dropout(out)
        out= self.linear2(out)
        return out