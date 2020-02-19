from torch.optim.sgd import SGD
from torch.optim.optimizer import required
import torch


class RDA(SGD):
    def __init__(self,
                 model,
                 proxs,
                 lr=required,
                 alpha=0.9,
                 reg=10e-08,
                 momentum=0,
                 dampening=0,
                 nesterov=False):

        kwargs = dict(lr=lr,
                      momentum=momentum,
                      dampening=dampening,
                      weight_decay=0,
                      nesterov=nesterov)
        super().__init__(model.parameters(), **kwargs)

        self.average_subgradient = 0
        self.previous_average_subgradient = [0] * len(
            self.param_groups[0]['params'])
        self.step_number = 1
        self.beta = 0.2
        self.alpha = alpha
        self.reg = reg
        self.names = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.names.append(name)

        if len(proxs) != len(self.param_groups):
            raise ValueError(
                "Invalid length of argument proxs: {} instead of {}".format(
                    len(proxs), len(self.param_groups)))

        for group, prox in zip(self.param_groups, list(proxs)):
            group.setdefault('prox', prox)

    def step(self, closure=None):
        # perform a gradient step
        # optionally with momentum or nesterov acceleration
        super().step(closure=closure)
        # print(self.param_groups[0]['params'][0].grad)
        self.beta = self.alpha * self.step_number**self.alpha
        # print(self.beta)
        # TODO: No grad
        for i, p in enumerate(self.param_groups[0]['params']):
            gradient = p.grad

            if self.step_number == 1:
                self.previous_average_subgradient[i] = gradient.clone()

            else:

                self.average_subgradient = (
                    self.step_number -
                    1) / self.step_number * self.previous_average_subgradient[
                        i] + gradient/ self.step_number

                case_one_index = torch.where(
                    self.average_subgradient < -self.reg)
                case_two_index = torch.where(
                    torch.abs(self.average_subgradient) <= self.reg)
                case_three_index = torch.where(
                    self.average_subgradient > self.reg)

                # print(len(case_one_index[0]))
                # print(len(case_two_index[0]))
                # print(len(case_three_index[0]))
                # print(self.average_subgradient)
                values_case_one = self.average_subgradient[case_one_index]
                update_case_one = -self.step_number / self.beta * (
                    values_case_one + self.reg)

                values_case_three = self.average_subgradient[case_three_index]
                update_case_three = -self.step_number / self.beta * (
                    values_case_three - self.reg)

                p.data[case_one_index] = update_case_one
                p.data[case_two_index] = 0
                p.data[case_three_index] = update_case_three

                # TODO: check the clone
                self.previous_average_subgradient[i] = self.average_subgradient

                # print(self.names[i])
                # print(self.average_subgradient)

            # print(p.data.nonzero())
            # print(case_two_index)

        # if(self.step_number == 10):
        #     exit()
        self.step_number += 1