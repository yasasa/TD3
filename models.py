import torch


def make_linear_layer(inp, outp, activation=torch.nn.ReLU):
    return torch.nn.Sequential(torch.nn.Linear(inp, outp),
                            activation())

def make_linear_dropout_layer(inp, outp, activation=torch.nn.ReLU, p=0.5):
    return torch.nn.Sequential(torch.nn.Linear(inp, outp),
                            activation(),
                            torch.nn.Dropout(p=p))

def _init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Model(torch.nn.Module):
    def __init__(self, config, init_weights=_init_weights, activation=torch.nn.ReLU):
        super(Model, self).__init__()
        model_list = [
            make_linear_layer(inp, outp, activation)
            for inp, outp in zip(config[:-2], config[1:-1])
        ]
        model_list.append(torch.nn.Linear(config[-2], config[-1]))
        self.model = torch.nn.Sequential(*model_list)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


#TODO: Test this
class BootstrapModel(torch.nn.Module):
    def __init__(self, branches, layer_config, init_weights=_init_weights, activation=torch.nn.ReLU):
        super(BootstrapModel, self).__init__()
        self.branches = branches
        self.branch_size = layer_config[-1]
        self.inp_size = layer_config[0]

        lc = layer_config
        self.hidden_layers = torch.nn.ModuleList([
            Model(layer_config, init_weights, activation)
            for _ in range(branches)
        ])

    def forward(self, x, mask):
        batch_size = x.shape[0]
        ys = []
        for module in self.hidden_layers:
            y = module(x)
            ys.append(y)
        elems = mask.nonzero()

        i = torch.tensor(list(zip(*elems)), dtype=torch.int64)
        _x = torch.cat(ys, dim=1)
        _x = _x.unfold(1, self.branch_size, self.branch_size)
        x = torch.zeros_like(_x)
        x[i[0], i[1]] = _x[i[0], i[1]]
        x = x.view(batch_size, -1)
        return x

    def multiforward(self, x, mask=None):
        xs = x.unfold(1, self.inp_size, self.inp_size)
        assert(xs.shape[1] == len(self.hidden_layers))
        ys = []
        for i, module in enumerate(self.hidden_layers):
            _x = xs[torch.arange(x.shape[0]), torch.tensor([i]).expand(x.shape[0]), ...]
            _x = _x.view(x.shape[0], -1)
            y = module(_x)
            ys.append(y)
        y = torch.cat(ys, dim=1)
        if mask:
            _x = y.unfold(1, self.branch_size, self.branch_size)
            x = torch.zeros_like(_x)
            x[i[0], i[1]] = _x[i[0], i[1]]
            y = x.view(batch_size, -1)
        return y

class BNN(torch.nn.Module):
    def __init__(self, layer_config, init_weights=_init_weights, activation=torch.nn.ReLU):
        super(BNN, self).__init__()
        model_config = [make_linear_dropout_layer(lc1, lc2) for lc1, lc2 in zip(layer_config[:-2], layer_config[1:])]
        model_config.append(torch.nn.Linear(layer_config[-2], layer_config[-1]))
        self.model = torch.nn.Sequential(*model_config)

    def set_dropout(self, mode=True):
        def f(layer):
            if type(layer) == torch.nn.Dropout:
                if mode:
                    layer.train()
                else:
                    layer.eval()
        self.model.apply(f)

    def forward(self, x):
        return self.model(x)




