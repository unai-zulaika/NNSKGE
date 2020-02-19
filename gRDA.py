import torch
from torch.optim.optimizer import Optimizer, required


class gRDA(Optimizer):
    def __init__(self, params, lr=0.01, c=0.005, mu=0.7):
        defaults = dict(lr=lr, c=c, mu=mu)
        super(gRDA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(gRDA, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            c = group['c']
            mu = group['mu']

            for p in group['params']:

                if p.grad is None:
                    continue
                d_p = p.grad.data
                # TODO: Idea, if tensor W, no penalty
                if len(p.shape) == 3:
                    i=0
                    # print(d_p)
                    # p.data.add_(-0.1, d_p)
                else:
                    param_state = self.state[p]

                    if 'iter_num' not in param_state:
                        iter_num = param_state['iter_num'] = torch.zeros(1)
                        accumulator = param_state[
                            'accumulator'] = torch.FloatTensor(p.shape).to(
                                p.device)
                        l1_accumulation = param_state[
                            'l1_accumulation'] = torch.zeros(1)
                        accumulator.data = p.clone()
                    else:
                        iter_num = param_state['iter_num']
                        accumulator = param_state['accumulator']
                        l1_accumulation = param_state['l1_accumulation']
                    iter_num.add_(1)
                    accumulator.data.add_(-lr, d_p)

                    # l1 = c * torch.pow(torch.tensor(lr), 0.5 + mu) * torch.pow(iter_num, mu)
                    l1_diff = c * torch.pow(
                        torch.tensor(lr), mu + 0.5) * torch.pow(
                            iter_num, mu) - c * torch.pow(
                                torch.tensor(lr), mu + 0.5) * torch.pow(
                                    iter_num - 1, mu)
                    l1_accumulation += l1_diff

                    new_a_l1 = torch.abs(
                        accumulator.data) - l1_accumulation.to(p.device)
                    p.data = torch.sign(
                        accumulator.data) * new_a_l1.clamp(min=0)
                    # print(p.data)

        return loss


class custom_SGD(Optimizer):
    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(custom_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(custom_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # if len(p.shape) == 3:
                #     print(d_p)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss