import torch
import torch.nn as nn
import torch.nn.functional as F

def call_bn(bn, x):
    return bn(x)

########################################################################################################################################################

##CNNAE  CNN自动编码器

########################################################################################################################################################
class CNNAE(nn.Module):
    def __init__(self, input_channel=1, embedding_dim=10, input_dim = 29,dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNNAE, self).__init__()
        self.c1=nn.Conv1d(input_channel,64,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv1d(64,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv1d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv1d(256,128,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1)
        self.l_c1=nn.Linear(128,embedding_dim)
        
        self.bn1=nn.BatchNorm1d(64)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(128)
        self.bn4=nn.BatchNorm1d(256)
        self.bn5=nn.BatchNorm1d(128)
        self.bn6=nn.BatchNorm1d(128)
        
        self.l_c2=nn.Linear(embedding_dim,64)
        self.c11=nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1)
        self.c12=nn.Conv1d(128,256,kernel_size=3,stride=1, padding=1)
        self.c13=nn.Conv1d(256,128,kernel_size=3,stride=1, padding=1)
        self.c14=nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1)
        self.c15=nn.Conv1d(128,64,kernel_size=3,stride=1, padding=1)
        self.c16=nn.Conv1d(64,input_channel,kernel_size=3,stride=1, padding=1)
        
        self.l_c3=nn.Linear(32,input_dim)
        
        self.bn11=nn.BatchNorm1d(64)
        self.bn12=nn.BatchNorm1d(64)
        self.bn13=nn.BatchNorm1d(64)
        self.bn14=nn.BatchNorm1d(32)
        self.bn15=nn.BatchNorm1d(32)
        #self.bn16=nn.BatchNorm1d(256)
        
    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool1d(h, kernel_size=2, stride=2)
        h=F.dropout(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.avg_pool1d(h, kernel_size=h.data.shape[2])
        
        
        h = h.view(h.size(0),h.size(1))
        u = self.l_c1(h)
        #u = h
        h = self.l_c2(u)
        
        h=self.c11(h)
        h=F.leaky_relu(call_bn(self.bn11, h), negative_slope=0.01)
        h=self.c12(h)
        h=F.leaky_relu(call_bn(self.bn12, h), negative_slope=0.01)
        h=self.c13(h)
        h=F.leaky_relu(call_bn(self.bn13, h), negative_slope=0.01)
        h=F.max_pool1d(h, kernel_size=2, stride=2)
        h=F.dropout(h, p=self.dropout_rate)

        h=self.c14(h)
        h=F.leaky_relu(call_bn(self.bn14, h), negative_slope=0.01)
        h=self.c15(h)
        h=F.leaky_relu(call_bn(self.bn15, h), negative_slope=0.01)
        h=self.c16(h)
        x=self.l_c3(h)
        #h=F.leaky_relu(call_bn(self.bn16, h), negative_slope=0.01)
        #h=F.max_pool1d(h, kernel_size=2, stride=2)
        #x=F.dropout(h, p=self.dropout_rate)
        
        return x,u


########################################################################################################################################################

##全连接网络

########################################################################################################################################################
    
class Mutual_net(nn.Module):
    def __init__(self):
        super(Mutual_net, self).__init__()

        self.fc1 = nn.Linear(10, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 10)
        self.last = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = self.last(x)
        return x

########################################################################################################################################################

##全连接2

########################################################################################################################################################
    
class CNN(nn.Module):
    def __init__(self, input_channel=1, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1=nn.Conv1d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv1d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv1d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv1d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv1d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv1d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv1d(256,128,kernel_size=2,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm1d(128)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(128)
        self.bn4=nn.BatchNorm1d(256)
        self.bn5=nn.BatchNorm1d(256)
        self.bn6=nn.BatchNorm1d(256)
        self.bn7=nn.BatchNorm1d(512)
        self.bn8=nn.BatchNorm1d(256)
        self.bn9=nn.BatchNorm1d(128)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool1d(h, kernel_size=2, stride=2)
        h=F.dropout(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool1d(h, kernel_size=2, stride=2)
        h=F.dropout(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool1d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        if self.top_bn:
            logit=call_bn(self.bn_c1, logit)
        return logit
    

#################################################################################################################################################


##MLPAE

#################################################################################################################################################

class MLPAE(nn.Module):
    def __init__(self,input_dim = 29, embedding_dim = 10):
        super(MLPAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 32), nn.ReLU(True), nn.Linear(32, embedding_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, input_dim), nn.Tanh())
        

    def forward(self, x):
        u = self.encoder(x)
        x = self.decoder(u)
        return x,u.squeeze(1)

#################################################################################################################################################


##MLPAE_cls

#################################################################################################################################################


class MLPAE_cls(nn.Module):
    def __init__(self,input_dim = 29, embedding_dim = 15, output_dim = 12):
        super(MLPAE_cls, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 32), nn.ReLU(True), nn.Linear(32, embedding_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, input_dim), nn.Tanh())
        self.cls = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, output_dim)
            
        )

    def forward(self, x):
        u = self.encoder(x)
        x = self.decoder(u)
        y = self.cls(u)
        
        return x,u.squeeze(1),y

#################################################################################################################################################


##MLP for MoPro

#################################################################################################################################################

class MLPAE_for_DeepRe(nn.Module):
    def __init__(self,input_dim = 29, num_class = 15, low_dim = 12):
        super(MLPAE_for_DeepRe, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 32), nn.ReLU(True), nn.Linear(32, 32))                 
        self.decoder = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, input_dim), nn.Tanh())
        
        self.classifier = nn.Linear(32, num_class)
        self.l2norm = Normalize(2)
        
        #projection MLP
        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, low_dim)

    def forward(self, x):
        u = self.encoder(x)
        feat_ = u.view(u.size(0), -1)
        
        x_ = self.decoder(feat_)
        
        out = self.classifier(feat_)
        
        feat = F.relu(self.fc1(feat_))
        feat = self.fc2(feat)
        feat = self.l2norm(feat) 
        
        return out,feat,feat_,x_

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

