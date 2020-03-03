import math

import torch
from torch.optim.optimizer import Optimizer, required


class gRDA_momentum(Optimizer):
    def __init__(self,
                 params,
                 lr=required,
                 c=0.005,
                 mu=0.7,
                 momentum=0,
                 reg='l1'):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr,
                        c=c,
                        mu=mu,
                        momentum=momentum,
                        dampening=0,
                        reg=reg)
        super(gRDA_momentum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(gRDA_momentum, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            reg = group['reg']
            lr = group['lr']
            c = group['c']
            mu = group['mu']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:

                if p.grad is None:
                    continue
                d_p = p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                        d_p = buf

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

                l1_diff = c * torch.pow(torch.tensor(
                    lr), mu + 0.5) * torch.pow(iter_num, mu) - c * torch.pow(
                        torch.tensor(lr), mu + 0.5) * torch.pow(
                            iter_num - 1, mu)
                l1_accumulation += l1_diff
                new_a_l1 = torch.abs(accumulator.data) - l1_accumulation.to(
                    p.device)

                if reg == 'l1':
                    p.data = torch.sign(
                        accumulator.data) * new_a_l1.clamp(min=0)
                elif reg == 'elasticnet':
                    p.data = 1 / (
                        1 + 20 * l1_accumulation.to(p.device)) * torch.sign(
                            accumulator.data) * new_a_l1.clamp(min=0)
                elif reg == 'g_lasso':
                    p.data = (1 - l1_accumulation.to(p.device) / torch.norm(
                        accumulator.data, p=2)) * accumulator.data

        return loss


class gRDAAdam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 c=0.005,
                 mu=0.7,
                 reg='l1'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad,
                        c=c,
                        mu=mu,
                        reg=reg)
        super(gRDAAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(gRDAAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
            reg = group['reg']
            lr = group['lr']
            c = group['c']
            mu = group['mu']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                grad = p.data.addcdiv_(-step_size, exp_avg, denom)

                # compute RDA grads
                if 'iter_num' not in state:
                    iter_num = state['iter_num'] = torch.zeros(1)
                    accumulator = state['accumulator'] = torch.FloatTensor(
                        p.shape).to(p.device)
                    l1_accumulation = state['l1_accumulation'] = torch.zeros(1)
                    accumulator.data = p.clone()
                else:
                    iter_num = state['iter_num']
                    accumulator = state['accumulator']
                    l1_accumulation = state['l1_accumulation']
                iter_num.add_(1)
                accumulator.data.add_(-lr, grad)

                # l1 = c * torch.pow(torch.tensor(lr), 0.5 + mu) * torch.pow(iter_num, mu)
                l1_diff = c * torch.pow(torch.tensor(
                    lr), mu + 0.5) * torch.pow(iter_num, mu) - c * torch.pow(
                        torch.tensor(lr), mu + 0.5) * torch.pow(
                            iter_num - 1, mu)
                l1_accumulation += l1_diff

                if reg == 'l1':
                    new_a_l1 = torch.abs(
                        accumulator.data) - l1_accumulation.to(p.device)
                    p.data = torch.sign(
                        accumulator.data) * new_a_l1.clamp(min=0)
                elif reg == 'elasticnet':
                    new_a_l1 = torch.abs(
                        accumulator.data) - l1_accumulation.to(p.device)
                    p.data = 1 / (
                        1 + 0.5 * l1_accumulation.to(p.device)) * torch.sign(
                            accumulator.data) * new_a_l1.clamp(min=0)
                elif reg == 'g_lasso':
                    new_a_l1 = torch.abs(
                        accumulator.data) - l1_accumulation.to(p.device)
                    p.data = (1 - l1_accumulation.to(p.device) / torch.norm(
                        accumulator.data, p=2)) * accumulator.data

        return loss
