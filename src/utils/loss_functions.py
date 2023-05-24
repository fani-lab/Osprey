from typing import Any
from src.utils.commons import RegisterableObject

import torch
import torch.nn.functional as F

import numpy as np


class BaseLossCalculator(RegisterableObject):
    pass

class WeightedBinaryCrossEntropy(BaseLossCalculator):

    def __init__(self, pos_weight=1, *args, **kwargs) -> None:
        super().__init__()
        self.pos_weight = pos_weight
    
    @classmethod
    def short_name(cls) -> str:
        return "weighted-binary-cross-entropy"

    def __call__(self, predictions, targets):
        _p = torch.clamp(predictions, 1e-7, 1 - 1e-7)
        result = (-targets * torch.log(_p) * self.pos_weight + (1 - targets) * - torch.log(1 - _p)).sum()
        return result


class DynamicSuperLoss(BaseLossCalculator, torch.nn.Module):
    """ Dynamic Loss =
        (loss_func(x) - expectation(loss)) * conf

    The confidence is deterministically computed as
        conf = argmin_c [(loss_func(x) - expecation) * c + weight_decay * log(c)**2]
        i.e., as the fixpoint of the confidence in a dynamic system.

    ncls:       number of classes
    nsamples:   number of training instances

    wd_cls:     weight decay for the classes
    wd_ins:     weight decay for the instances

    smooth_cls: smoothing parameter for the class losses
    smooth_ins: smoothing parameter for the instance losses

    loss_func:  loss function that will be used.
    store_conf: if True, we store the instance confidence in the model at every epoch
    """
    
    def __init__(self, nsamples, ncls, loss_func=torch.nn.CrossEntropyLoss(), wd_cls=0, wd_ins=0, expectation=0,
                 smooth_cls=0, smooth_ins=0, smooth_init=0, mode='metaloss',
                 store_conf=False):
        torch.nn.Module.__init__(self)
        BaseLossCalculator.__init__(self)
        assert ncls > 0 and nsamples > 0, 'need to know the number of class and labels'

        self.smooth_cls = smooth_cls
        self.smooth_ins = smooth_ins
        self.smooth_init = smooth_init
        self.ncls = ncls
        self.class_smoother = Smoother(self.smooth_cls, ncls, init=self.smooth_init)
        self.nsamples = nsamples
        self.instance_smoother = Smoother(self.smooth_ins, nsamples, init=self.smooth_init)
        self.loss_func = loss_func

        assert hasattr(self.loss_func, 'reduction')
        self.loss_func.reduction = 'none'
        
        self.optimal_conf_ins = make_optimal_conf(wd_ins, mode)
        self.optimal_conf_cls = make_optimal_conf(wd_cls, mode)

        self.expectation = make_expectator(expectation)
        self.store_conf = store_conf


    def forward(self, preds, labels, indices, **kw):
        loss_batch = self.loss_func(preds, labels)

        conf_ins = conf_cls = 1

        if self.optimal_conf_ins:
            smoothed_loss_ins = self.instance_smoother(loss_batch.detach(), indices)
            threshold_ins = self.expectation( smoothed_loss_ins )
            conf_ins = self.optimal_conf_ins(smoothed_loss_ins -threshold_ins)
            self.expectation.update( smoothed_loss_ins, conf_ins )

        if self.optimal_conf_cls:
            smoothed_loss_cls = self.class_smoother(loss_batch.detach(), labels)
            threshold_cls = self.expectation( smoothed_loss_cls )
            conf_cls = self.optimal_conf_cls(smoothed_loss_cls -threshold_cls)
            self.expectation.update( smoothed_loss_cls, conf_cls )

        conf = conf_ins * conf_cls

        # compute the final loss
        loss = (loss_batch * conf).mean()

        return loss

    @classmethod
    def short_name(cls) -> str:
        return "dynamic-super-loss"


class Smoother(torch.nn.Module):
    def __init__(self, smoothing, nsamples, init=0):
        super().__init__()
        assert 0 <= smoothing < 1
        self.smoothing = smoothing
        self.nsamples = nsamples
        if self.smoothing:
            if isinstance(init, (int, float)):
                assert nsamples > 0
                init = torch.full([nsamples], init)
            self.register_buffer('memory', init.clone())

    def __call__(self, values, indices=None):
        if self.smoothing > 0:
            assert len(values) == len(indices)
            binned_values = torch.bincount(indices, weights=values, minlength=self.nsamples)
            bin_size = torch.bincount(indices, minlength=self.nsamples).float()
            nnz = (bin_size > 0) # which classes are represented
            means = binned_values[nnz] / bin_size[nnz] # means for each class
            alpha = self.smoothing ** bin_size[nnz]
            self.memory[nnz] = alpha * self.memory[nnz] + (1-alpha) * means # update
            return self.memory[indices]
        else:
            return values


loss_div_wd = np.float32([-1000, -0.7357585932962737, -0.7292385198866751, -0.7197861042909649,
    -0.7060825529685993, -0.6862159572880272, -0.6574145455480526, -0.6156599675844636,
    -0.5551266577364037, -0.46736905653740307, -0.34014329294487, -0.15569892914556094,
    0.11169756647530316, 0.4993531412919867, 1.0613531942004133, 1.8761075276533326,
    3.0572900212223724, 4.769698321281568, 7.252246278161051, 10.851297017399714,
    16.06898724880869, 23.63328498268829, 34.599555050301056, 50.497802769609315,
    73.54613907594951, 106.96024960367691, 155.40204460004963, 225.63008495214464,
    327.4425312511471, 475.0441754009414, 689.0282819387658, 999.249744])

conf = np.float32([1, 0.9991138577461243, 0.8724386692047119, 0.8048540353775024, 0.7398145198822021,
    0.6715637445449829, 0.5973713397979736, 0.5154045820236206, 0.42423248291015625,
    0.3226756751537323, 0.20976418256759644, 0.08473344892263412, -0.05296758562326431,
    -0.2036692053079605, -0.3674810528755188, -0.5443023443222046, -0.7338425517082214,
    -0.9356498718261719, -1.149145483970642, -1.3736592531204224, -1.6084641218185425,
    -1.8528070449829102, -2.1059343814849854, -2.367111921310425, -2.6356399059295654,
    -2.910861015319824, -3.1921679973602295, -3.479003667831421, -3.770861864089966,
    -4.067285060882568, -4.367861747741699, -4.67222261428833])


def get_optimal_conf(loss, weight_decay):
    assert weight_decay > 0
    return np.interp(loss / weight_decay, loss_div_wd, conf)

class OptimalConf(torch.nn.Module):
    """ Pytorch implementation of the get_optimal_conf() function above
    """
    def __init__(self, weight_decay=1,  mode='torch'):
        super().__init__()
        self.weight_decay = weight_decay
        self.mode = mode

        # transformation from: loss_div_wd[1:] --> [0, ..., len(loss_div_wd)-2]
        log_loss_on_wd = torch.log(torch.from_numpy(loss_div_wd[1:]) + 0.750256)
        step = (log_loss_on_wd[-1] - log_loss_on_wd[0]) / (len(log_loss_on_wd) - 1)
        offset = log_loss_on_wd[0]

        # now compute step and offset such that [0,30] --> [-1,1]
        self.log_step = step * (len(log_loss_on_wd) - 1) / 2
        self.log_offset = offset + self.log_step
        self.register_buffer('optimal_conf', torch.from_numpy(conf[1:]).view(1,1,1,-1))

    def __call__(self, loss):
        loss = loss.detach()
        if self.mode == 'numpy':
            conf = get_optimal_conf(loss.cpu().numpy(), self.weight_decay)
            r = torch.from_numpy(conf).to(loss.device)

        elif self.mode == 'torch':
            l = loss / self.weight_decay
            # using grid_sampler in the log-space of loss/wd
            l = ( torch.log(l + 0.750256) - self.log_offset ) / self.log_step
            l[torch.isnan(l)] = -1 # not defined before -0.75
            l = torch.stack((l, l.new_zeros(l.shape)), dim=-1).view(1,1,-1,2)
            r =F.grid_sample(self.optimal_conf, l, padding_mode="border", align_corners=True)
        return torch.exp(r.view(loss.shape))


class Constant(torch.nn.Module):
    def __init__(self, expectation):
        super().__init__()
        self.expectation = expectation

    def __call__(self, values):
        return self.expectation

    def update(self, values, weights=None):
        pass


def make_expectator(expectation):
    if expectation is None: return None
    if isinstance(expectation, str):
        expectation = eval(expectation)
    if isinstance(expectation, (int, float)):
        expectation = Constant(expectation)
    return expectation

def make_optimal_conf(wd, mode):
    if wd == 0:
        return None
    elif mode == 'metaloss':
        return OptimalConf(wd)
    else:
        raise ValueError('bad mode '+mode)
