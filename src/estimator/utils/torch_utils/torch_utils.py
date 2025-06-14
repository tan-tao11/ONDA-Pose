import math
import os
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.estimator.utils.utils import iprint


def init_seeds(seed=0):
    torch.manual_seed(seed)

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device="", apex=False, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), "CUDA unavailable, invalid device %s requested" % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    ng = len(device.split(","))  # torch.cuda.device_count()
    if cuda:
        c = 1024 ** 2  # bytes to MB

        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, "batch-size %g not multiple of GPU count %g" % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = "Using CUDA " + ("Apex " if apex else "")  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = " " * len(s)
            iprint(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                % (s, i, x[i].name, x[i].total_memory / c)
            )
    else:
        iprint("Using CPU")

    iprint("")  # skip a line

    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def is_parallel(model):
    # is model is parallel with DP or DDP
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0.0, 0.0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune

    iprint("Pruning model... ", end="")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)  # prune
            prune.remove(m, "weight")  # make permanent
    iprint(" %.3g global sparsity" % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True,
        ).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


def model_info(model, verbose=False, imgsz=64, device="cpu"):
    device = select_device(device)
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        iprint(
            "%5s %40s %9s %12s %20s %10s %10s"
            % (
                "layer",
                "name",
                "gradient",
                "parameters",
                "shape",
                "mu",
                "sigma",
            )
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            iprint(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )

    try:  # FLOPS
        from thop import profile

        flops = (
            profile(
                deepcopy(model),
                inputs=(torch.zeros(1, 3, imgsz, imgsz).to(device),),
                verbose=verbose,
            )[0]
            / 1e9
            * 2
        )
        fs = ", %.1f GFLOPS_%dx%d" % (flops, imgsz, imgsz)  # FLOPS
    except:
        fs = ""

    iprint("Model Summary: %g layers, %g parameters, %g gradients%s" % (len(list(model.parameters())), n_p, n_g, fs))


def load_classifier(name="resnet101", n=2):
    # Loads a pretrained model reshaped to n-class output
    model = models.__dict__[name](pretrained=True)

    # Display model properties
    input_size = [3, 224, 224]
    input_space = "RGB"
    input_range = [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for x in [input_size, input_space, input_range, mean, std]:
        iprint(x + " =", eval(x))

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        gs = 32  # (pixels) grid size
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Model Exponential Moving Average from
    https://github.com/rwightman/pytorch-image-models Keep a moving average of
    everything in the model state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        # FIXME: should exclude some constant parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
