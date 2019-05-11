from utils import orthongonal_init_, AddBias, FixedNormal

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(GaussianPolicy, self).__init__()

        self.fc_mean = orthongonal_init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CategoricalPolicy, self).__init__()
        self.affine1 = nn.Linear(state_dim, action_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        action_scores = self.affine1(x)
        return F.softmax(action_scores, dim=1)