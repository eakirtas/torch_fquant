import torch as T


class MinMax(T.nn.Module):

    @T.no_grad()
    def __init__(self, x=None):
        super().__init__()

        self.register_buffer('vmin', T.tensor(0.0))
        self.register_buffer('vmax', T.tensor(0.0))

        if x is not None:
            self(x)

    @T.no_grad()
    def forward(self, x, do_step=True):
        if len(x.size()) > 1:
            self.vmin = T.min(x, dim=1)[0].mean()
            self.vmax = T.max(x, dim=1)[0].mean()
        else:
            self.vmin = T.min(x).mean()
            self.vmax = T.max(x).mean()

        return self.vmin, self.vmax

    def get_vmin(self):
        return self.vmin

    def get_vmax(self):
        return self.vmax
