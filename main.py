
import time
import json
import numpy as np
import re
import math
import os
import torch
import torch.nn as nn
import csv
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.args_factory import get_args
from src.loaders import get_loaders, get_statistics
from src.utils import (Statistics, count_vars, write_config, AdvAttack, MyVisionDataset)
from tqdm import tqdm
import random
from warnings import warn
from src.deepTrunk_networks import MyDeepTrunkNet
from src.zonotope import HybridZonotope
from model.model import Model, set_eps, get_eps
from model.norm_dist import set_p_norm, get_p_norm
from src.args_factory import get_args
from src.attack import AttackPGD
from src.utils import random_seed, create_result_dir, Logger, TableLogger, AverageMeter
from src.adamw import AdamW


seed = 100

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.set_printoptions(precision=10)
np.random.seed(seed)
random.seed(seed)

def run(args=None):
    device = 'cuda' if torch.cuda.is_available() and (not args.no_cuda) else 'cpu'

    num_train, train_loader, test_loader, input_size, input_channel, n_class = get_loaders(args)


    lossFn = nn.CrossEntropyLoss(reduction='none')
    def evalFn(x): return torch.max(x, dim=1)[1]
    ## Create CombinedNet and load model
    dTNet = MyDeepTrunkNet.get_deepTrunk_net(args, device, lossFn, evalFn, input_size, input_channel, n_class)

    timestamp = int(time.time())
    model_signature = '%s/%s/%d/%s_%.5f/%d' % (args.dataset, args.exp_name, args.exp_id, args.net, args.train_eps, timestamp)
    model_dir = args.root_dir + 'models_new/%s' % (model_signature)
    args.model_dir = model_dir


    print("Saving model to: %s" % model_dir)
    count_vars(args, dTNet)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    tb_writer = SummaryWriter(model_dir)
    stats = Statistics(len(train_loader), tb_writer, model_dir)
    args_file = os.path.join(model_dir, 'args.json')
    with open(args_file, 'w') as fou:
        json.dump(vars(args), fou, indent=4)
    write_config(args, os.path.join(model_dir, 'run_config.txt'))


    train_deepTrunk(dTNet, args, device, stats, train_loader, test_loader)

    with args.cert:
        with torch.no_grad():
            cert_deepTrunk_net(dTNet, args, device, test_loader, stats, log_ind=True)
    #if args.cert:
    #    with torch.no_grad():
    #        cert_deepTrunk_net(dTNet, args, device, test_loader if args.test_set == "test" else train_loader,
    #                           stats, log_ind=True, break_on_failure=False, epoch=epoch)

def train_deepTrunk(dTNet, args, device, stats, train_loader, test_loader):
    n_epochs = args.n_epochs

    ## Get a duplicate of the data loaders that will be changed
    train_set_spec = MyVisionDataset.from_idx(train_loader.dataset, np.ones_like(train_loader.dataset.targets).astype(bool), train=True)
    train_set_spec.set_weights(None)
    train_loader_spec = torch.utils.data.DataLoader(train_set_spec, batch_size=args.train_batch,
                                                    shuffle=~args.debug, num_workers=train_loader.num_workers, pin_memory=True, drop_last=True)
    test_set_spec = MyVisionDataset.from_idx(test_loader.dataset,
                                             np.ones_like(test_loader.dataset.targets).astype(bool), train=False)
    test_set_spec.set_weights(None)
    test_loader_spec = torch.utils.data.DataLoader(test_set_spec, batch_size=args.test_batch,
                                                   shuffle=False, num_workers=train_loader.num_workers, pin_memory=True, drop_last=False)

    for k, exit_idx in enumerate(dTNet.exit_ids[1::]):
        model = dTNet.gate_nets[exit_idx]
        print("getting dataset for gate training")
        train_loader, test_loader = get_gated_data_loaders(args, device,
                                                                        dTNet.branch_nets[exit_idx],
                                                                        train_loader_spec, test_loader_spec)
        loss_name, params = parse_function_call(args.loss)
        
        loss = Loss(globals()[loss_name](**params), args.kappa)
        #loss = torch.nn.functional.binary_cross_entropy_with_logits()
        result_dir = args.result_dir
        output_flag = True
        if output_flag:
            logger = Logger(os.path.join(result_dir, 'log.txt'))
            for arg in vars(args):
                logger.print(arg, '=', getattr(args, arg))
            logger.print(train_loader.dataset.transform)
            logger.print(model)
            logger.print('number of params: ', sum([p.numel() for p in model.parameters() if p.requires_grad == True]))
            logger.print('Using loss', loss)
            train_logger = TableLogger(os.path.join(result_dir, 'train.log'), ['epoch', 'loss', 'acc'])
            test_logger = TableLogger(os.path.join(result_dir, 'test.log'), ['epoch', 'loss', 'acc'])
        else:
            logger = train_logger = test_logger = None
        if args.eps_test is None:
            args.eps_test = default_eps[args.dataset]
        if args.eps_train is None:
            args.eps_train = args.eps_test
        mean, std = get_statistics(args.dataset.upper())
        args.eps_train /= std
        args.eps_test /= std
        up = torch.FloatTensor((1 - mean) / std).view(-1, 1, 1).to(device)
        down = torch.FloatTensor((0 - mean) / std).view(-1, 1, 1).to(device)
        attacker = AttackPGD(model, args.eps_test, step_size=args.eps_test / 4, num_steps=20, up=up, down=down)
        args.epochs = [int(epoch) for epoch in args.epochs.split(',')]

        ## optimizer的参数是否要加载？
        optimizer = AdamW(model, lr=args.lr, weight_decay=args.wd, betas=(args.beta1,args.beta2), eps=args.epsilon)
        schedule = create_schedule(args, len(train_loader), model, loss, optimizer)

        if args.visualize and output_flag:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(result_dir)
        else: writer = None

        for epoch in range(args.start_epoch, args.n_epochs):
            train_loss, train_acc = train(model, loss, epoch, train_loader, optimizer, schedule,
                                        logger, train_logger, device, args.print_freq, attacker)
            test_loss, test_acc = test(model, loss, epoch, test_loader, logger, test_logger, device, args.print_freq)
            if writer is not None:
                writer.add_scalar('curve/p', get_p_norm(model), epoch)
                writer.add_scalar('curve/train loss', train_loss, epoch)
                writer.add_scalar('curve/test loss', test_loss, epoch)
                writer.add_scalar('curve/train acc', train_acc, epoch)
                writer.add_scalar('curve/test acc', test_acc, epoch)
            if epoch % 50 == 49:
                if logger is not None:
                    logger.print('Generate adversarial examples on training dataset and test dataset (fast, inaccurate)')
                robust_train_acc = gen_adv_examples(dTNet,attacker, train_loader, device, logger, fast=True)
                robust_test_acc = gen_adv_examples(dTNet, attacker, test_loader, device, logger, fast=True)
                if writer is not None:
                    writer.add_scalar('curve/robust train acc', robust_train_acc, epoch)
                    writer.add_scalar('curve/robust test acc', robust_test_acc, epoch)
            if epoch % 5 == 0:
                certified_acc = certified_test(dTNet, args.eps_test, up, down, test_loader, logger, device).float().mean().item()
                if writer is not None:
                    writer.add_scalar('curve/certified acc', certified_acc, epoch)
            if epoch > args.epochs[-1] - 3:
                if logger is not None:
                    logger.print("Generate adversarial examples on test dataset")
                gen_adv_examples(dTNet, attacker, test_loader, device, logger)

        schedule(args.epochs[-1], 0)
        
        print("Calculate certified accuracy on training dataset and test dataset")
        certified_test(dTNet, args.eps_test, up, down,  train_loader, logger, device)
        certified_test(dTNet, args.eps_test, up, down,  test_loader, logger, device)

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(result_dir, 'model.pth'))

def cert_deepTrunk_net(dTNet, args, device, data_loader, stats, log_ind=True, break_on_failure=False, epoch=None, domains=None):
    dTNet.eval()
    n_exits = dTNet.n_branches
    tot_corr = 0
    tot_adv_corr = 0
    tot_samples = 0

    adv_attack_test = AdvAttack(eps=args.test_eps, n_steps=args.test_att_n_steps,
                                step_size=args.test_att_step_size, adv_type="pgd")

    if log_ind:
        data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=data_loader.batch_size,
                                    shuffle=False, num_workers=data_loader.num_workers, pin_memory=True,
                                    drop_last=False)
        csv_log = open(os.path.join(args.model_dir, "cert_log.csv"), mode='w')
        log_writer = csv.writer(csv_log, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        
        
    ver_corr = 0
    pbar = tqdm(data_loader, dynamic_ncols=True)
    for it, (inputs, targets) in enumerate(pbar):
        if args.debug and it == 50:
            pbar.close()
            break

        inputs, targets = inputs.to(device), targets.to(device)
        tot_samples += inputs.size()[0]
        nat_out, gate_out, exit_mask, nat_branches = dTNet(inputs)
        res = (outputs.max(dim=1)[1] == labels)
        if dTNet.trunk_net is None:
            exit_mask[-2] = exit_mask[-2].__or__(exit_mask[-1])
            exit_mask = exit_mask[:-1]
        nat_ok = targets.eq(dTNet.evalFn(nat_out[torch.stack(exit_mask, dim=1)])).detach()
        _, adv_ok, _, _, adv_branches = dTNet.get_adv_loss(inputs, targets, adv_attack_test)
        select_mask_adv, min_gate_out, max_gate_out = dTNet.get_adv_loss(inputs, targets, adv_attack_test, mode="reachability")

        tot_select_hist += select_mask_adv.sum(dim=0).detach()
        tot_select_n += (torch.arange(1, n_exits + 2, device=device).unsqueeze(dim=0) == select_mask_adv.sum(
            dim=1).unsqueeze(dim=1)).sum(dim=0).detach()


        tot_corr += nat_ok.int().sum().item()
        tot_adv_corr += adv_ok.int().sum().item()

    print('nat_corr:%.4f, adv_corr:%.4f, %ver_corr:%.4f' %(tot_corr/tot_samples, tot_adv_corr/tot_samples, ver_corr))

    
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

def cross_entropy():
    return F.cross_entropy

# The hinge loss function is a combination of max_hinge_loss and average_hinge_loss.
def hinge(mix=0.75):
    def loss_fun(outputs, targets):
        return mix * outputs.max(dim=1)[0].clamp(min=0).mean() + (1 - mix) * outputs.clamp(min=0).mean()
    return loss_fun

class Loss():
    def __init__(self, loss, kappa):
        self.loss = loss
        self.kappa = kappa
    def __call__(self, *args):
        margin_output = args[0] - torch.gather(args[0], dim=1, index=args[-1].view(-1, 1))
        if len(args) == 2:
            return self.loss(margin_output, args[-1])
        # args[1] which corresponds to worse_outputs, is already a margin vector.
        return self.kappa * self.loss(args[1], args[-1]) + (1 - self.kappa) * self.loss(margin_output, args[-1])

def create_schedule(args, batch_per_epoch, model, loss, optimizer):
    epoch0, epoch1, epoch2, epoch3, tot_epoch = args.epochs
    speed = math.log(args.p_end / args.p_start)
    def num_batches(epoch, minibatch=0):
        return epoch * batch_per_epoch + minibatch

    def schedule(epoch, minibatch):
        ratio = max(num_batches(epoch - epoch1, minibatch) / num_batches(tot_epoch - epoch1), 0)
        lr_now = 0.5 * args.lr * (1 + math.cos((ratio * math.pi)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now

        ratio = min(max(num_batches(epoch - epoch1, minibatch) / num_batches(epoch3 - epoch1), 0), 1)
        if ratio >= 1:
            p_norm = float('inf')
        else:
            p_norm = args.p_start * math.exp(speed * ratio)
        set_p_norm(model, p_norm)

        if epoch2 > 0:
            ratio = min(max(num_batches(epoch - epoch0, minibatch) / num_batches(epoch2), 0), 1)
        else:
            ratio = 1.0
        set_eps(model, args.eps_train * ratio)
        loss.kappa = args.kappa

    return schedule

def train(net, loss_fun, epoch, trainloader, optimizer, schedule, logger, train_logger, device, print_freq, attacker):
    if logger is not None:
        logger.print('Epoch %d training start' % (epoch))
    net.train()
    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    train_loader_len = len(trainloader)

    for batch_idx, data in enumerate(trainloader):
        if len(data)==2:
            inputs, targets = data
            sample_weights = torch.ones(size=targets.size(),dtype=torch.float)
        else:
            inputs, targets, sample_weights = data
        schedule(epoch, batch_idx)
        data_time.update(time.time() - start)
        inputs = inputs.to(device)
        targets = targets.to(device)
        #perturb = attacker.find(inputs, targets)
        outputs = net(inputs)
        #print('******',outputs,worse_outputs)
        #print(targets.size())
        #print(outputs.size())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.view(-1), targets.float())
        #print(loss.size())
        with torch.no_grad():
            losses.update(loss.data.item(), targets.size(0))
            accs.update(cal_acc(outputs.data, targets, 0).mean().item(), targets.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        if (batch_idx + 1) % print_freq == 0 and logger is not None:
            logger.print('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'lr {lr:.4f}\tp {p:.2f}\teps {eps:.4f}\tkappa{kappa:.4f}\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                         epoch, batch_idx + 1, train_loader_len, batch_time=batch_time,
                         lr=optimizer.param_groups[0]['lr'],
                         p=get_p_norm(net), eps=get_eps(net), kappa=loss_fun.kappa,
                         loss=losses, acc=accs))
        start = time.time()

    loss, acc = losses.avg, accs.avg

    if train_logger is not None:
        train_logger.log({'epoch': epoch, 'loss': loss, 'acc': acc})
    if logger is not None:
        logger.print('Epoch {0}:  train loss {loss:.4f}   acc {acc:.4f}'
                     '   lr {lr:.4f}   p {p:.2f}   eps {eps:.4f}   kappa {kappa:.4f}'.format(
                     epoch, loss=loss, acc=acc, lr=optimizer.param_groups[0]['lr'],
                     p=get_p_norm(net), eps=get_eps(net), kappa=loss_fun.kappa))
    return loss, acc

@torch.no_grad()
def test(net, loss_fun, epoch, testloader, logger, test_logger, device, print_freq):
    net.eval()
    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    test_loader_len = len(testloader)

    for batch_idx, data in enumerate(testloader):
        if len(data)==2:
            inputs, targets = data
            sample_weights = torch.ones(size=targets.size(),dtype=torch.float)
        else:
            inputs, targets, sample_weights = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.view(-1), targets)
        losses.update(loss.mean().item(), targets.size(0))
        accs.update(cal_acc(outputs, targets, 0).item(), targets.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
        if (batch_idx + 1) % print_freq == 0 and logger is not None:
            logger.print('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                         batch_idx + 1, test_loader_len, batch_time=batch_time, loss=losses, acc=accs))

    loss, acc = losses.avg, accs.avg

    if test_logger is not None:
        test_logger.log({'epoch': epoch, 'loss': loss, 'acc': acc})
    if logger is not None:
        logger.print('Epoch %d:  '%epoch + 'test loss  ' + f'{loss:.4f}' + '   acc ' + f'{acc:.4f}')
    return loss, acc

def get_gated_data_loaders(args, device, net, train_loader, test_loader, threshold=0):

    gate_train_loader, class_list = get_gated_data(args, device, net, train_loader, is_train=True, threshold=threshold)

    gate_test_loader, _ = get_gated_data(args, device, net, test_loader, is_train=False, threshold=threshold)

    return gate_train_loader, gate_test_loader

def get_gated_data(args, device, net, data_loader,is_train=True, threshold=0):

    class_list = None

    data_set_tmp = MyVisionDataset.from_idx(data_loader.dataset, np.ones_like(data_loader.dataset.targets).astype(bool))
    data_loader_tmp = torch.utils.data.DataLoader(data_set_tmp, batch_size=data_loader.batch_size,
                                                  shuffle=False, num_workers=data_loader.num_workers, pin_memory=True, drop_last=False)


    mean, std = get_statistics(args.dataset.upper())        
    if args.eps_test is None:
        args.eps_test = default_eps[args.dataset]
    if args.eps_train is None:
        args.eps_train = args.eps_test
    args.eps_train /= std
    args.eps_test /= std
    up = torch.FloatTensor((1 - mean) / std).view(-1, 1, 1).to(device)
    down = torch.FloatTensor((0 - mean) / std).view(-1, 1, 1).to(device)

    threshold_cert = certified_test(net, args.eps_test, up, down, data_loader,None, device)
    #print(threshold_cert)
    #gate_target = (threshold_cert > threshold).cpu().detach()
    gate_target = threshold_cert.cpu().detach()
    #print(gate_target)

    data_set = MyVisionDataset.from_idx_and_targets(data_loader.dataset, torch.ones_like(gate_target).numpy(),
                                                        gate_target.float(), ["gate_easy_cert", "gate_difficult_cert"])
                                
    if len(data_set) == 0:
        return None, class_list
    else:
        return torch.utils.data.DataLoader(data_set, batch_size=args.train_batch if is_train else args.test_batch,
                                       shuffle=~args.debug if is_train else False, num_workers=data_loader.num_workers,
                                       pin_memory=True, drop_last=False), class_list

def cal_acc(outputs, targets, threshold= 0):
    predicted = torch.max(outputs.data, threshold)[1]
    return (predicted == targets).float().mean()

def certified_test(net, eps, up, down, testloader, logger, device):
    save_p = get_p_norm(net)
    save_eps = get_eps(net)
    set_eps(net, eps)
    set_p_norm(net, float('inf'))
    net.eval()
    outputs = []
    labels = []
    pbar = tqdm(testloader, dynamic_ncols=True)

    for batch_idx, data in enumerate(pbar):
        if len(data)==2:
            inputs, targets = data
            sample_weights = torch.ones(size=targets.size(),dtype=torch.float)
        else:
            inputs, targets, sample_weights = data
        inputs, targets = inputs.to(device), targets.to(device)
        lower = torch.max(inputs - eps, down)
        upper = torch.min(inputs + eps, up)

        outputs.append(net(inputs, lower=lower, upper=upper, targets=targets.long())[1])
        labels.append(targets)

        
        
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    res = (outputs.max(dim=1)[1] == labels)
    if logger is not None:
        logger.print(' certified acc ' + f'{res.float().mean().item():.4f}')
        
    set_p_norm(net, save_p)
    set_eps(net, save_eps)
    return res

def gen_adv_examples(model, attacker, test_loader, device, logger, fast=False):
    model.eval()
    correct = 0
    tot_num = 0
    size = len(test_loader)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device,dtype=torch.long)
        result = torch.ones(targets.size(0), dtype=torch.bool, device=targets.device)
        for i in range(1):
            perturb = attacker.find(inputs, targets)
            with torch.no_grad():
                outputs = model(perturb)
                predicted = torch.max(outputs.data, 1)[1]
                result &= (predicted == targets)
        correct += result.float().sum().item()
        tot_num += inputs.size(0)
        if fast and batch_idx * 10 >= size: break

    acc = correct / tot_num * 100

    if logger is not None:
        logger.print('adversarial attack acc ' + f'{acc:.4f}')
    return acc


def main():
    args = get_args()
    run(args=args)


if __name__ == '__main__':
    main()
