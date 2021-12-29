import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class SoftMax(nn.Module):

    def __init__(self, model, args):
        super().__init__()

        # embedding model
        self.embedding = model

        # classifier model
        self.classifier = nn.Linear(args.n_embed, args.n_keyword)

    def forward(self, x, y):
        x = self.embedding(x)
        x = self.classifier(x)

        return x



class ArcFace(nn.Module):
    def __init__(self, model, args):
        super(ArcFace, self).__init__()
        self.embedding = model
        self.s = args.s
        self.m = args.m
        self.W = nn.Parameter(torch.FloatTensor(args.n_keyword, args.n_embed))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        # compute embedding
        x = self.embedding(x)
        # normalize features
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)

        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class CosFace(nn.Module):
    def __init__(self, model, args):
        super(CosFace, self).__init__()
        self.embedding = model
        self.s = args.s
        self.m = args.m
        self.W = nn.Parameter(torch.FloatTensor(args.n_keyword, args.n_embed))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        # compute embedding
        x = self.embedding(x)
        # normalize features
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)

        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output




class AdaCos(nn.Module):
    def __init__(self, model, args):
        super(AdaCos, self).__init__()
        self.embedding = model
        self.s = math.sqrt(2) * math.log(args.n_keyword - 1)
        self.m = args.m
        self.W = nn.Parameter(torch.FloatTensor(args.n_keyword, args.n_embed))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        # compute embedding
        x = self.embedding(x)
        # normalize features
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)

        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / x.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output

class SphereFace(nn.Module):
    def __init__(self, model, args):
        super(SphereFace, self).__init__()
        self.embedding = model
        self.s = args.s
        self.m = args.m
        self.W = nn.Parameter(torch.FloatTensor(args.n_keyword, args.n_embed))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        # compute embedding
        x = self.embedding(x)
        # normalize features
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)

        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output