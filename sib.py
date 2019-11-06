import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import label_to_1hot, dni_linear, LinearDiag, FeatExemplarAvgBlock


class ClassifierSIB(nn.Module):
    """
    nKall: number of categories in train-set
    nKnovel: number of categories in an episode
    nFeat: feature dimension at the input of classifier
    q_steps: number of iteration used in weights refinement
    """
    def __init__(self, nKnovel, nFeat, q_steps):
        super(ClassifierSIB, self).__init__()

        self.nKnovel = nKnovel
        self.nFeat = nFeat
        self.q_steps = q_steps

        # bias & scale of classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # init_net
        self.favgblock = FeatExemplarAvgBlock(self.nFeat)
        self.wnLayerFavg = LinearDiag(self.nFeat)

        # dni grad_net
        self.dni = dni_linear(self.nKnovel, dni_hidden_size=self.nKnovel*8)

    def apply_classification_weights(self, features, cls_weights):
        '''
        (B x n x nFeat, B x nKnovel x nFeat) -> B x n x nKnovel
        '''
        features = F.normalize(features, p=2, dim=features.dim()-1, eps=1e-12)
        cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)

        cls_scores = self.scale_cls * torch.baddbmm(1.0, self.bias.view(1, 1, 1), 1.0,
                                                    features, cls_weights.transpose(1,2))
        return cls_scores

    def init_theta(self, features_supp, labels_supp_1hot):
        '''
        return theta is B * nKnovel x nFeat
        '''
        theta = self.favgblock(features_supp, labels_supp_1hot) # B x nKnovel x nFeat
        batch_size, nKnovel, num_channels = theta.size()
        theta = theta.view(batch_size * nKnovel, num_channels)
        theta = self.wnLayerFavg(theta) # weight each feature differently
        theta = theta.view(-1, nKnovel, num_channels)
        return theta

    def refine_theta(self, theta, features_query, lr):
        '''
        theta <-- self.init_theta()
        Refine theta by performing approx GD on L_query
        '''
        batch_size, num_examples = features_query.size()[:2]
        new_batch_dim = batch_size * num_examples

        for t in range(self.q_steps):
            cls_scores = self.apply_classification_weights(features_query, theta)
            cls_scores = cls_scores.view(new_batch_dim, -1) # B * n x nKnovel
            grad_logit = self.dni(cls_scores) # B * n x nKnovel
            grad = torch.autograd.grad([cls_scores], [theta],
                                       grad_outputs=[grad_logit],
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True)[0] # B x nKnovel x nFeat

            # perform GD
            theta = theta - lr * grad

        return theta

    def get_classification_weights(self, lr=None,
                                   features_supp=None, labels_supp_1hot=None, features_query=None):
        '''
        features_supp, labels_supp --> self.init_theta
        features_query --> self.refine_theta
        '''
        assert(features_supp is not None and features_query is not None)

        # generate weights for novel categories
        features_supp = F.normalize(features_supp, p=2, dim=features_supp.dim()-1, eps=1e-12)

        weight_novel = self.init_theta(features_supp, labels_supp_1hot)
        weight_novel = self.refine_theta(weight_novel, features_query, lr)

        return weight_novel


    def forward(self, lr=None,
                features_supp=None, labels_supp=None, features_query=None):
        '''
        features_supp: (B, nKnovel * nExamplar, nFeat)
        labels_supp: (B, nknovel * nExamplar) in [0, nKnovel - 1]
        features_query: (B, nKnovel * nTest, nFeat)
        '''
        labels_supp_1hot = label_to_1hot(labels_supp, self.nKnovel)
        cls_weights = self.get_classification_weights(lr, features_supp, labels_supp_1hot, features_query)
        cls_scores = self.apply_classification_weights(features_query, cls_weights)

        return cls_scores


if __name__ == "__main__":
    net = ClassifierSIB(nKall=64, nKnovel=5, nFeat=512, q_steps=3)
    net = net.cuda()

    features_supp = torch.rand((8, 5 * 1, 512)).cuda()
    features_query = torch.rand((8, 5 * 15, 512)).cuda()
    labels_supp = torch.randint(5, (8, 5 * 1)).cuda()
    lr = 1e-3

    cls_scores = net(lr, features_supp, labels_supp, features_query)
    print(cls_scores.size())

