import torch as T


class MinMaxStd(T.nn.Module):

    def __init__(self, alpha, x=None):
        super().__init__()

        self.register_buffer('alpha', T.tensor(alpha))

        self.register_buffer('vmin', T.tensor(0.0))
        self.register_buffer('vmax', T.tensor(0.0))

        if x is not None:
            self.vmin, self.vmax = self(x)

    @T.no_grad()
    def __call__(self, x, do_step=True):
        if len(x.size()) > 1:
            average_v = T.mean(x, dim=1).mean()
            std_v = T.std(x, dim=1, unbiased=False).mean()
        else:
            average_v = T.mean(x).mean()
            std_v = T.std(x, unbiased=False).mean()

        self.vmin = average_v - self.alpha * std_v
        self.vmax = average_v + self.alpha * std_v

        return self.vmin, self.vmax

    def get_vmin(self):
        return self.vmin

    def get_vmax(self):
        return self.vmax
