import torch as T


class MinMaxSimpler(T.nn.Module):

    @T.no_grad()
    def __init__(self, x=None):
        super().__init__()

        self.register_buffer('vmin', T.tensor(0.0))
        self.register_buffer('vmax', T.tensor(0.0))

        if x is not None:
            self(x)

    @T.no_grad()
    def forward(self, x, do_step=True):
        self.vmin = T.min(x).mean()
        self.vmax = T.max(x).mean()

        return self.vmin, self.vmax

    def get_vmin(self):
        return self.vmin

    def get_vmax(self):
        return self.vmax
