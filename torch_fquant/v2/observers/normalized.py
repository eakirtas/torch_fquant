import torch as T


class Normalized(T.nn.Module):

    @T.no_grad()
    def __init__(self,
                 alpha,
                 beta=2.0,
                 x=None,
                 inc_step=False,
                 edge_values=[-1, 1]):
        super().__init__()

        self.register_buffer('beta', T.tensor(beta))
        self.register_buffer('alpha', T.tensor(alpha))
        self.register_buffer('inc_step', T.tensor(inc_step))
        self.register_buffer('inc_step_store', T.tensor(inc_step))
        self.register_buffer('step_v', T.tensor(1.0))

        self.register_buffer('ema_w', T.tensor(0.0))
        self.register_buffer('ema_avg', T.tensor(0.0))
        self.register_buffer('ema_std', T.tensor(1.0))

        self.register_buffer(
            'vmin',
            T.tensor(self.ema_avg -
                     self.alpha * self.ema_std))  # TODO: Should be checked
        self.register_buffer(
            'vmax',
            T.tensor(self.ema_avg -
                     self.alpha * self.ema_std))  # TODO: Should be checked

        if x is not None:
            self.vmin, self.vmax = self(x)

    @T.no_grad()
    def __call__(self, x, do_step=True):
        self.ema_w = self.beta / (self.step_v + 1)

        if len(x.size()) == 1:
            avg_val = T.mean(x).mean()
            std_val = T.std(x, unbiased=False).mean()
        else:

            avg_val = T.mean(x, dim=1).mean()
            std_val = T.std(x, dim=1, unbiased=False).mean()

        self.ema_avg = self.ema_w * avg_val + (1.0 - self.ema_w) * self.ema_avg
        self.ema_std = self.ema_w * std_val + (1.0 - self.ema_w) * self.ema_std

        self.vmin = self.ema_avg - self.alpha * self.ema_std
        self.vmax = self.ema_avg + self.alpha * self.ema_std

        self.step(do_step)

        return self.vmin, self.vmax

    def step(self, do_step: bool):
        if self.inc_step and do_step:
            self.step_v += 1

    def get_vmin(self):
        return self.vmin

    def get_vmax(self):
        return self.vmax

    def train(self, mode: bool = True):
        if mode:
            self.inc_step = self.inc_step_store
        else:
            self.inc_step = T.tensor(False)

        return super().train(mode)

    def eval(self):
        self.inc_step = T.tensor(False)
        return super().eval()
