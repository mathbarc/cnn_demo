import torch
import torch.optim
import math
from typing import Dict


class ObjDetectionRampUpLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        rampup_period: int,
        power: int = 4,
    ):
        self._optimizer = optimizer

        self._lr_base = lr
        self._current_step = 0

        self._rampup_period = rampup_period
        self._power = power

    def step(self):
        lr = self.get_last_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        self._current_step += 1

    def get_last_lr(self):
        if self._current_step >= self._rampup_period:
            lr = self._lr_base

        elif self._current_step <= self._rampup_period:
            lr = self._lr_base * pow(
                (self._current_step / self._rampup_period), self._power
            )

        return lr


class YoloObjDetectionRampUpLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: Dict[int, float],
        peak_lr: float,
        rampup_period: int,
        minimal_rampup_lr: float = 1e-3,
        power: int = 4,
    ):
        self._optimizer = optimizer

        self._lr_base = peak_lr
        self._current_step = 0

        self._rampup_period = rampup_period
        self._power = power

        self._lr_steps = lr
        self._minimal_rampup_lr = minimal_rampup_lr

    def step(self):
        lr = self.get_last_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        self._current_step += 1

    def get_last_lr(self):
        if self._current_step >= self._rampup_period:
            lr = self._lr_base
            steps = list(self._lr_steps.keys())
            steps.sort()
            if self._current_step >= steps[0]:
                i = 0
                while i < len(steps):
                    step = steps[i]
                    if self._current_step >= step:
                        lr = self._lr_steps[step]
                        i += 1
                    else:
                        break

        elif self._current_step <= self._rampup_period:
            lr = self._minimal_rampup_lr + (
                self._lr_base - self._minimal_rampup_lr
            ) * pow((self._current_step / self._rampup_period), self._power)

        return lr


class ObjDetectionLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_base: float,
        lr_overshoot: float,
        overshoot_period: int,
    ):
        self._optimizer = optimizer

        self._lr_base = lr_base
        self._lr_overshoot = lr_overshoot

        self._current_step = 0
        self._overshoot_period = overshoot_period

        self._overshoot_amplitude = self._lr_overshoot - self._lr_base

    def step(self):
        lr = self.get_last_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        self._current_step += 1

    def get_last_lr(self):
        if self._current_step >= self._overshoot_period:
            lr = self._lr_base

        elif self._current_step < self._overshoot_period:
            lr = self._lr_base + self._overshoot_amplitude * math.sin(
                (math.pi) * (self._current_step / (self._overshoot_period))
            )

        return lr


class ObjDetectionExponentialDecayLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_base: float,
        lr_final: float,
        n_steps: int,
        rampup_period: int,
        power: int = 4,
    ):
        self._optimizer = optimizer

        self._lr_base = lr_base

        self._current_step = 0
        self._rampup_period = rampup_period
        self._decay_period = n_steps - rampup_period

        self._decay_amplitude = self._lr_base - lr_final
        self._lr_final = lr_final
        self._power = power

    def step(self):
        lr = self.get_last_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        self._current_step += 1

    def get_last_lr(self):
        if self._current_step > self._rampup_period:
            lr = self._lr_final + self._decay_amplitude * pow(
                (
                    (self._decay_period - (self._current_step - self._rampup_period))
                    / self._decay_period
                ),
                self._power,
            )

        elif self._current_step <= self._rampup_period:
            lr = self._lr_base * pow(
                (self._current_step / self._rampup_period), self._power
            )

        return lr


class ObjDetectionCosineDecayLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_base: float,
        lr_final: float,
        n_steps: int,
        rampup_period: int,
        power: int = 4,
    ):
        self._optimizer = optimizer

        self._lr_base = lr_base

        self._current_step = 0
        self._rampup_period = rampup_period
        self._decay_period = n_steps - rampup_period

        self._decay_amplitude = self._lr_base - lr_final
        self._lr_final = lr_final
        self._power = power

    def step(self):
        lr = self.get_last_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        self._current_step += 1

    def get_last_lr(self):
        if self._current_step > self._rampup_period:
            lr = self._lr_final + self._decay_amplitude * math.sin(
                (math.pi / 2)
                * ((self._current_step - self._rampup_period) / self._decay_period)
                + (math.pi / 2)
            )

        elif self._current_step <= self._rampup_period:
            lr = self._lr_base * pow(
                (self._current_step / self._rampup_period), self._power
            )

        return lr


class ObjDetectionLogisticDecayLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_base: float,
        lr_final: float,
        n_steps: int,
        rampup_period: int,
        power: int = 4,
    ):
        self._optimizer = optimizer

        self._lr_base = lr_base

        self._current_step = 0
        self._rampup_period = rampup_period
        self._decay_period = n_steps - rampup_period

        self._decay_amplitude = self._lr_base - lr_final
        self._lr_final = lr_final
        self._power = power

    def step(self):
        lr = self.get_last_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        self._current_step += 1

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + (math.exp(-x)))

    def get_last_lr(self):
        if self._current_step >= self._rampup_period - 1:
            expoent = (
                (self._decay_period - (self._current_step - self._rampup_period))
                / self._decay_period
            ) * 12 - 7
            lr = self._lr_final + self._decay_amplitude * self._sigmoid(expoent)

        elif self._current_step < self._rampup_period - 1:
            lr = self._lr_base * pow(
                (self._current_step / self._rampup_period), self._power
            )

        return lr


class ObjDetectionCosineAnnealingLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_base: float,
        lr_final: float,
        rampup_period: int,
        cosine_period: int,
        cosine_period_inc: float,
        power: int = 4,
    ):
        self._optimizer = optimizer

        self._lr_base = lr_base

        self._current_step = 0
        self._rampup_period = rampup_period

        self._cosine_period = cosine_period
        self._cosine_period_inc = cosine_period_inc

        self._decay_amplitude = self._lr_base - lr_final
        self._lr_final = lr_final

        self._cosine_interval_start = self._rampup_period
        self._power = power

    def step(self):
        lr = self.get_last_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        self._current_step += 1

    def get_last_lr(self):
        if self._current_step >= self._rampup_period:
            pos_in_interval = self._current_step - self._cosine_interval_start
            lr = self._lr_final + self._decay_amplitude * (
                math.cos((math.pi / 2) * (pos_in_interval / self._cosine_period))
            )
            if pos_in_interval == self._cosine_period:
                self._cosine_interval_start = self._current_step
                self._cosine_period *= self._cosine_period_inc

        elif self._current_step < self._rampup_period:
            lr = self._lr_base * pow(
                (self._current_step / self._rampup_period), self._power
            )

        return lr
