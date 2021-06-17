import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm

import re
import json
import os
from itertools import combinations
from PIL import Image
from warnings import warn
from model.model import Model, set_eps, get_eps
from model.norm_dist import set_p_norm, get_p_norm


def parse_function_call(s):
    s = re.split(r'[()]', s)
    if len(s) == 1:
        return s[0], {}
    name, params, _ = s
    params = re.split(r',\s*', params)
    params = dict([p.split('=') for p in params])
    for key, value in params.items():
        try:
            params[key] = int(params[key])
        except ValueError:
            try:
                params[key] = float(params[key])
            except ValueError:
                pass
    return name, params

def get_network(device, dataset, net_name, input_size, input_channel, n_class, net_config=None, net_dim=None):
    if net_name.startswith("LipNet"):
        tokens = net_name.split("_")
        if len(tokens)< 2:
            tokens.append('MLPFeature(depth=5,width=5)')
        
        from model.bound_module import Predictor, BoundFinalIdentity
        from model.mlp import MLPFeature, MLP
        from model.conv import ConvFeature, Conv  

        model_name, params = parse_function_call(tokens[1])
        predictor_hidden_size = 512
        input_dim = [input_channel,input_size,input_size]
        if predictor_hidden_size > 0:
            model = locals()[model_name](input_dim=input_dim, **params)
            predictor = Predictor(model.out_features, predictor_hidden_size, n_class)
        else:
            model = locals()[model_name](input_dim=input_dim, num_classes=n_class, **params)
            predictor = BoundFinalIdentity()
        net = Model(model, predictor, eps=0).to(device)
    else:
        assert False, 'Unknown network!'
    #net.determine_dims(torch.randn((2, input_channel, input_size, input_size), dtype=torch.float).to(device))
    return net



def get_net(device, dataset, net_name, input_size, input_channel, n_class, load_model=None, net_config=None, balance_factor=1, net_dim=None):
    net = get_network(device, dataset, net_name, input_size, input_channel, n_class, net_config=net_config, net_dim=net_dim).to(device) #, feature_extract=-1).to(device)

    #if n_class == 1 and isinstance(net.blocks[-1], Linear) and net.blocks[-1].bias is not None:
    #    net.blocks[-1].linear.bias.data = torch.tensor(-norm.ppf(balance_factor/(balance_factor+1)),
    #                                                   dtype=net.blocks[-1].linear.bias.data.dtype).view(net.blocks[-1].linear.bias.data.shape)

    init_slopes(net, device, trainable=False)
    return net


def my_cauchy(*shape):
    return torch.clamp(torch.FloatTensor(*shape).cuda().cauchy_(), -1e7, 1e7)



class Statistics:

    def __init__(self, window_size, tb_writer, log_dir=None, post_fix=None):
        self.window_size = window_size
        self.tb_writer = tb_writer
        self.values = {}
        self.steps = 0
        self.log_dir = log_dir
        self.post_fix = "" if post_fix is None else post_fix

    def update_post_fix(self, post_fix=""):
        self.post_fix = post_fix

    def report(self, metric_name, value):
        metric_name = metric_name + self.post_fix
        if metric_name not in self.values:
            self.values[metric_name] = []
        self.values[metric_name] += [value]

    def export_to_json(self, file, epoch=None):
        epoch = self.steps if epoch is None else epoch
        data = {"epoch_%d/%s"%(epoch, k): np.mean(v) for k, v in self.values.items() if len(v)>0}
        data = self.parse_keys(data)
        with open(file, 'a') as f:
            json.dump(data, f, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None,
                      indent="\t", separators=None, default=None, sort_keys=True)

    def report_hist(self, metric_name, values):
        metric_name = metric_name + self.post_fix
        self.tb_writer.add_histogram(metric_name, values.view(-1).detach().cpu().numpy(), self.steps)

    def print_stats(self):
        print('==============')
        for k, v in self.values.items():
            print('%s: %.5lf' % (k, np.mean(v)))

    def get(self, k, no_post_fix=False):
        k = k if no_post_fix else k+self.post_fix
        return np.mean(self.values[k])

    def update_tb(self, epoch=None):
        if self.log_dir is not None:
            self.export_to_json(os.path.join(self.log_dir, "training_log.json"), epoch=epoch)
        for k, v in self.values.items():
            if len(v) == 0: continue
            self.tb_writer.add_scalar(k, np.mean(v), self.steps if epoch is None else epoch)
            self.values[k] = []
        self.steps += 1

    def parse_keys(self, data_old):
        data_new = {}
        for k, v in data_old.items():
            data_new = self.add_to_leveled_dict(data_new, k, v)
        return data_new

    def add_to_leveled_dict(self, data_dict, label, data):
        if "/" not in label:
            data_dict[label] = data
            return data_dict
        labels = label.split("/")
        if labels[0] in data_dict:
            data_dict[labels[0]] = self.add_to_leveled_dict(data_dict[labels[0]],"/".join(labels[1:]),data)
        else:
            data_dict[labels[0]] = self.add_to_leveled_dict({},"/".join(labels[1:]),data)
        return data_dict


def init_slopes(net, device, trainable=False):
    for param_name, param_value in net.named_parameters():
        if 'deepz' in param_name:
            param_value.data = -torch.ones(param_value.size()).to(device)
            param_value.requires_grad_(trainable)


def count_vars(args, net):
    var_count = 0
    var_count_t = 0
    var_count_relu = 0

    for p_name, params in net.named_parameters():
        if "weight" in p_name or "bias" in p_name:
            var_count += int(params.numel())
            var_count_t += int(params.numel() * params.requires_grad)
        elif "deepz_lambda" in p_name:
            var_count_relu += int(params.numel())

    args.var_count = var_count
    args.var_count_t = var_count_t
    args.var_count_relu = var_count_relu

    print('Number of parameters: ', var_count)


def write_config(args, file_path):
    f=open(file_path,'w+')
    for param in [param for param in dir(args) if not param[0] == "_"]:
        f.write("{:<30} {}\n".format(param + ":", (str)(getattr(args, param))))
    f.close()


class AdvAttack:

    def __init__(self, eps=2./255, n_steps=1, step_size=1.25, adv_type="pgd"):
        self.eps = eps
        self.n_steps = n_steps
        self.step_size = step_size
        self.adv_type = adv_type

    def update_params(self, eps=None, n_steps=None, step_size=None, adv_type=None):
        self.eps = self.eps if eps is None else eps
        self.n_steps = self.n_steps if n_steps is None else n_steps
        self.step_size = self.step_size if step_size is None else step_size
        self.adv_type = self.adv_type if adv_type is None else adv_type

    def get_params(self):
        return self.eps, self.n_steps, self.step_size, self.adv_type


def get_lp_loss(blocks, p=1, input_size=1, scale_conv=True):
    lp_loss = 0
    N = input_size

    for block in blocks:
        if isinstance(block,Conv2d):
            conv = block.conv
            N = max(np.floor((N + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) /
                             conv.stride[0]), 0.0) + 1
            lp_loss += block.weight.abs().pow(p).sum() * ((N * N) if scale_conv else 1)
        elif isinstance(block, Linear):
            lp_loss += block.weight.abs().pow(p).sum()

    return lp_loss


class MyVisionDataset(torchvision.datasets.vision.VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, data=None, targets=None, classes=None,
                 class_to_idx=None, orig_idx=None, sample_weights=None, yield_weights=True, loader=None):
        super(MyVisionDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        self.loader = loader

        self.data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data

        if isinstance(data, torch.Tensor) and self.data.ndim == 4 and self.data.shape[2] == self.data.shape[3]:
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = targets.cpu().detach().tolist() if isinstance(targets, torch.Tensor) else targets
        self.targets = targets.tolist() if isinstance(targets, np.ndarray) else targets

        sample_weights = len(targets)*[1.] if sample_weights is None else sample_weights
        self.sample_weights = sample_weights.cpu().detach().tolist() if isinstance(sample_weights, torch.Tensor) else sample_weights
        self.sample_weights = sample_weights.tolist() if isinstance(sample_weights, np.ndarray) else sample_weights
        self.yield_weights = yield_weights

        if orig_idx is None:
            self.orig_idx = np.arange(0, len(targets))
        else:
            self.orig_idx = orig_idx

        self.classes = classes
        self.class_to_idx = class_to_idx

    @staticmethod
    def from_idx(dataset, idx, sample_weights=None, train=True):
        new_len = len(idx) # For debugmode
        new_data = np.array([v[0] for v,i in zip(dataset.samples[:new_len],idx) if i]) if not hasattr(dataset,"data") \
                   else dataset.data[:new_len][idx]
        new_targets = np.array(dataset.targets)[:new_len][idx].tolist()
        new_weights = None if (not hasattr(dataset,"sample_weights") or dataset.sample_weights is None) else \
                      np.array(dataset.sample_weights)[:new_len][idx].tolist()
        yield_weights = False if not hasattr(dataset,"yield_weights") else dataset.yield_weights
        old_orig_idx = dataset.orig_idx if hasattr(dataset,"orig_idx") else np.arange(0,len(dataset))
        new_orig_idx = old_orig_idx[:new_len][idx]
        loader = dataset.loader if hasattr(dataset, "loader") else None
        new_dataset = MyVisionDataset(dataset.root, train, dataset.transform, dataset.target_transform, new_data,
                                      new_targets, dataset.classes, dataset.class_to_idx, new_orig_idx, new_weights,
                                      yield_weights, loader)
        if sample_weights is not None:
            new_dataset.set_weights(sample_weights)
        return new_dataset

    @staticmethod
    def from_idx_and_targets(dataset, idx, new_targets, classes, sample_weights=None):
        new_len = len(idx) # For debugmode
        new_data = np.array([v[0] for v,i in zip(dataset.samples[:new_len],idx) if i]) if not hasattr(dataset,"data") \
                   else dataset.data[:new_len][idx]
        assert new_data.shape[0] == len(new_targets)
        new_targets = new_targets.cpu().detach().numpy() if isinstance(new_targets,torch.Tensor) else new_targets
        new_weights = None if (not hasattr(dataset,"sample_weights") or dataset.sample_weights is None) else \
                      np.array(dataset.sample_weights)[:new_len][idx].tolist()
        yield_weights = False if not hasattr(dataset,"yield_weights") else dataset.yield_weights

        class_to_idx = {int(k) : classes[i] for i,k in enumerate(np.unique(np.array(new_targets)))}
        old_orig_idx = dataset.orig_idx if hasattr(dataset,"orig_idx") else np.arange(0,len(dataset))
        new_orig_idx = old_orig_idx[:new_len][idx]
        loader = dataset.loader if hasattr(dataset,"loader") else None
        new_dataset = MyVisionDataset(dataset.root, dataset.train, dataset.transform, dataset.target_transform, new_data,
                               new_targets, classes, class_to_idx, new_orig_idx, new_weights, yield_weights, loader)
        if sample_weights is not None:
            new_dataset.set_weights(sample_weights)
        return new_dataset

    def set_weights(self, sample_weights=None, yield_weights=True):
        sample_weights = len(self.targets) * [1.] if sample_weights is None else sample_weights
        self.sample_weights = sample_weights.cpu().detach().tolist() if isinstance(sample_weights,
                                                                                   torch.Tensor) else sample_weights
        self.sample_weights = sample_weights.tolist() if isinstance(sample_weights, np.ndarray) else sample_weights
        self.yield_weights = yield_weights


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.loader is not None:
            img = self.loader(img)
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.yield_weights:
            weight = self.sample_weights[index]
            return img, target, weight

        return img, target

    def __len__(self):
        return len(self.data)



def random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_result_dir(args):
    result_dir = args.result_dir
    mp = {
        'dataset': '',
        'model': '',
        'predictor_hidden_size': 'hid',
        'loss': None,
        'p_start': 'p',
        'p_end': 'p_end',
        'batch_size': 'bs',
        'beta': None,
        'beta2': None,
        'epsilon': None,
        'start_epoch': None,
        'checkpoint': None,
        'gpu': None,
        'dist_url': None,
        'world_size': None,
        'rank': None,
        'print_freq': None,
        'result_dir': None,
        'filter_name': '',
        'seed': '',
        'visualize': None,
    }
    for arg in vars(args):
        if arg in mp and mp[arg] is None:
            continue
        value = getattr(args, arg)
        if type(value) == bool:
            value = 'T' if value else 'F'
        if type(value) == list:
            value = str(value).replace(' ', '')
        name = mp.get(arg, arg)
        result_dir += name + str(value) + '_'
    return result_dir

def create_result_dir(args):
    result_dir = get_result_dir(args)
    id = 0
    while True:
        result_dir_id = result_dir + '_%d'%id
        if not os.path.exists(result_dir_id): break
        id += 1
    os.makedirs(result_dir_id)
    return result_dir_id

class Logger(object):
    def __init__(self, dir):
        self.fp = open(dir, 'w+')
    def __del__(self):
        self.fp.close()
    def print(self, *args, **kwargs):
        print(*args, file=self.fp, **kwargs)
        print(*args, **kwargs)

class TableLogger(object):
    def __init__(self, path, header):
        import csv
        self.fp = open(path, 'w')
        self.logger = csv.writer(self.fp, delimiter='\t')
        self.logger.writerow(header)
        self.header = header
    def __del__(self):
        self.fp.close()
    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.fp.flush()

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

