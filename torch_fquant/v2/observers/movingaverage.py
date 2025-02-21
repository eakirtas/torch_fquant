import torch as T


class MovingAverage(T.nn.Module):
    # TODO: Step should be increased only when it is on train mode
    @T.no_grad()
    def __init__(self, x=None, inc_step=True):

        super().__init__()

        self.register_buffer('step_v', T.tensor(1))
        self.register_buffer('inc_step', T.tensor(inc_step))
        self.register_buffer('inc_step_store', T.tensor(inc_step))
        self.register_buffer('ema_min', T.tensor(0.0))
        self.register_buffer('ema_max', T.tensor(0.0))

        if x is not None:
            self(x)

    @T.no_grad()
    def forward(self, x, do_step=True):
        if len(x.size()) > 1:
            vmax, _ = T.max(x, dim=1)
            vmin, _ = T.min(x, dim=1)
        else:
            vmax = T.max(x)
            vmin = T.min(x)

        ema_w = 2.0 / (self.step_v + 1)

        self.ema_min = ema_w * (vmin.mean()) + (1.0 - ema_w) * self.ema_min
        self.ema_max = ema_w * (vmax.mean()) + (1.0 - ema_w) * self.ema_max

        self.step(do_step)

        return self.ema_min, self.ema_max

    def step(self, do_step: bool):
        if self.inc_step and do_step:
            self.step_v += 1

    def get_vmin(self):
        return self.ema_min

    def get_vmax(self):
        return self.ema_max

    def train(self, mode: bool = True):
        if mode:
            self.inc_step = self.inc_step_store
        else:
            self.inc_step = T.tensor(False)

        return super().train(mode)

    def eval(self):
        self.inc_step = T.tensor(False)
        return super().eval()
