import torch


def make_linear_layer(inp, outp, activation=torch.nn.ReLU):
    return torch.nn.Sequential(torch.nn.Linear(inp, outp),
                            activation())

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



