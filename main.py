
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
from src.utils import random_seed, Logger, TableLogger, AverageMeter
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
    model_signature = '%s/%s/%d/%d' % (args.dataset, args.experiment_name, args.exp_id, timestamp)
    model_dir = args.root_dir + 'models_new/%s' % (model_signature)
    args.model_dir = model_dir

    if args.eps_test is None:
        args.eps_test = default_eps[args.dataset]
    if args.eps_train is None:
        args.eps_train = args.eps_test
    mean, std = get_statistics(args.dataset.upper())
    args.eps_train /= std
    args.eps_test /= std

    if args.mode == 'train':
        print("Saving model to: %s" % model_dir)
    #count_vars(args, dTNet)
    #count_vars(args, dTNet.branch_nets[0])
    count_vars(dTNet.gate_nets[0])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    tb_writer = SummaryWriter(model_dir)
    stats = Statistics(len(train_loader), tb_writer, model_dir)
    args_file = os.path.join(model_dir, 'args.json')
    with open(args_file, 'w') as fou:
        json.dump(vars(args), fou, indent=4)
    write_config(args, os.path.join(model_dir, 'run_config.txt'))

    if args.mode == 'train':

        train_deepTrunk(dTNet, args, device, stats, train_loader, test_loader)

    if args.mode == 'test':
        if args.load_branch_model == None or args.load_trunk_model == None or args.load_gate_model == None:
            print("Pretrained Model is needed in test mode!")
            exit(1)
        p_norm = float('inf')
        for k, exit_idx in enumerate(dTNet.exit_ids[1::]):
            set_p_norm(dTNet.branch_nets[exit_idx], p_norm)
            set_p_norm(dTNet.gate_nets[exit_idx], p_norm)

        cert_deepTrunk_net(dTNet, args, device, test_loader, stats, log_ind=True)


def train_deepTrunk(dTNet, args, device, stats, train_loader, test_loader):

    ## Get a duplicate of the data loaders that will be changed
    train_set_spec = MyVisionDataset.from_idx(train_loader.dataset, np.ones_like(train_loader.dataset.targets).astype(bool), train=True)
    train_set_spec.set_weights(None)
    train_loader_spec = torch.utils.data.DataLoader(train_set_spec, batch_size=args.train_batch,
                                                    shuffle=True, num_workers=train_loader.num_workers, pin_memory=True, drop_last=True)
    test_set_spec = MyVisionDataset.from_idx(test_loader.dataset,
                                             np.ones_like(test_loader.dataset.targets).astype(bool), train=False)
    test_set_spec.set_weights(None)
    test_loader_spec = torch.utils.data.DataLoader(test_set_spec, batch_size=args.test_batch,
                                                   shuffle=False, num_workers=train_loader.num_workers, pin_memory=True, drop_last=False)

    for k, exit_idx in enumerate(dTNet.exit_ids[1::]):
        model = dTNet.gate_nets[exit_idx]

        mean, std = get_statistics(args.dataset.upper())
        print("getting dataset for gate training")
        train_loader, test_loader = get_gated_data_loaders(args, device,
                                                                        dTNet.branch_nets[exit_idx],
                                                                        train_loader_spec, test_loader_spec)
        loss_name, params = parse_function_call(args.loss)
        

        loss = Loss(globals()[loss_name](**params), args.kappa)
        result_dir = args.result_dir
        output_flag = True
        if output_flag:
            logger = Logger(os.path.join(result_dir, 'log.txt'))
            for arg in vars(args):
                logger.print(arg, '=', getattr(args, arg))
            logger.print(train_loader.dataset.transform)
            logger.print(model)
            logger.print('number of params: ', sum([p.numel() for p in model.parameters()]))
            logger.print('number of params requires gradients: ', sum([p.numel() for p in model.parameters() if p.requires_grad == True]))
            logger.print('Using loss', args.loss)
            train_logger = TableLogger(os.path.join(result_dir, 'train.log'), ['epoch', 'loss', 'acc'])
            test_logger = TableLogger(os.path.join(result_dir, 'test.log'), ['epoch', 'loss', 'acc'])
        else:
            logger = train_logger = test_logger = None

        up = torch.FloatTensor((1 - mean) / std).view(-1, 1, 1).to(device)
        down = torch.FloatTensor((0 - mean) / std).view(-1, 1, 1).to(device)
        attacker = AttackPGD(model, args.eps_test, step_size=args.eps_test / 4, num_steps=20, up=up, down=down)
        args.epochs = [int(epoch) for epoch in args.epochs.split(',')]


        optimizer = AdamW(model, lr=args.lr, weight_decay=args.wd, betas=(args.beta1,args.beta2), eps=args.epsilon)
        schedule = create_schedule(args, len(train_loader), model, loss, optimizer)


        print("Now test gate_net's certified acc")

        certified_acc = certified_test(model, args.eps_test, up, down, test_loader, logger, device , threshold = args.threshold, n_class = 2).float().mean().item()
        total_acc = 0
        total_samples =0 
        tot_1 = 0
        tot_0 = 0
        save_p = get_p_norm(model)
        save_eps = get_eps(model)
        set_eps(model, args.eps_test)
        set_p_norm(model, float('inf'))
        model.eval()
        pbar = tqdm(test_loader, dynamic_ncols=True)
        for it, data in enumerate(pbar):
            if len(data)==2:
                inputs, targets = data
                sample_weights = torch.ones(size=targets.size(),dtype=torch.float)
            else:
                inputs, targets, sample_weights = data

            inputs, targets = inputs.to(device), targets.to(device)

            total_samples += targets.size()[0]
            outputs = model(inputs)
            total_acc += cal_acc(outputs.data, targets,0).float().sum().item()
            tot_1 += (outputs.max(dim=1)[1] == 1).float().sum().item()
            tot_0 += (outputs.max(dim=1)[1] == 0).float().sum().item()
        set_p_norm(model, save_p)
        set_eps(model, save_eps)
        print("gate_net's natrual acc: %.4f" %(total_acc/total_samples))
        print("output 1 percent :%.4f" %(tot_1/total_samples))
        print("output 0 percent :%.4f" %(tot_0/total_samples))

        if args.visualize and output_flag:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(result_dir)
        else: writer = None
        dTNet.gate_nets[0].train()
        for epoch in range(args.start_epoch, args.epochs[-1]):

            train_loss, train_acc = train(dTNet.gate_nets[0], loss, epoch, train_loader, optimizer, schedule,
                                        logger, train_logger, device, args.print_freq, attacker, args.threshold)
            test_loss, test_acc = test(dTNet.gate_nets[0], loss, epoch, test_loader, logger, test_logger, device, args.print_freq, args.threshold)
            if writer is not None:
                writer.add_scalar('curve/p', get_p_norm(dTNet.gate_nets[0]), epoch)
                writer.add_scalar('curve/train loss', train_loss, epoch)
                writer.add_scalar('curve/test loss', test_loss, epoch)
                writer.add_scalar('curve/train acc', train_acc, epoch)
                writer.add_scalar('curve/test acc', test_acc, epoch)
            if epoch % 50 == 49:
                if logger is not None:
                    logger.print('Generate adversarial examples on training dataset and test dataset (fast, inaccurate) ')
                robust_train_acc = gen_adv_examples(dTNet.gate_nets[0],attacker, train_loader, device, logger, fast=True, threshold=args.threshold , n_class =2)
                robust_test_acc = gen_adv_examples(dTNet.gate_nets[0], attacker, test_loader, device, logger, fast=True, threshold=args.threshold, n_class =2)
                if writer is not None:
                    writer.add_scalar('curve/robust train acc', robust_train_acc, epoch)
                    writer.add_scalar('curve/robust test acc', robust_test_acc, epoch)
            if epoch % 5 == 4:
                certified_acc = certified_test(dTNet.gate_nets[0], args.eps_test, up, down, test_loader, logger, device, threshold=args.threshold, n_class =2).float().mean().item()
                if writer is not None:
                    writer.add_scalar('curve/certified acc', certified_acc, epoch)
            if epoch > args.epochs[-1] - 3:
                if logger is not None:
                    logger.print("Generate adversarial examples on test dataset")
                gen_adv_examples(dTNet.gate_nets[0], attacker, test_loader, device, logger, threshold=args.threshold, n_class =2)

        schedule(args.epochs[-1], 0)
        

        torch.save({
            'state_dict': dTNet.gate_nets[0].state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(result_dir, 'gate_model.pth'))

        
def cert_deepTrunk_net(dTNet, args, device, data_loader, stats, log_ind=True, break_on_failure=False, epoch=None, domains=None):

    tot_trunk_corr = 0
    tot_gate_corr = 0
    tot_branch_corr = 0
    tot_samples = 0
    ## Use models like Resnet, need to un_normalize.
    mean = [0.4914, 0.4822, 0.4465]
    std = 0.2009
    def normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
        return t
    def un_normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] * std) + mean[0]
        t[:, 1, :, :] = (t[:, 1, :, :] * std) + mean[1]
        t[:, 2, :, :] = (t[:, 2, :, :] * std) + mean[2]
        return t
    
    dTNet.trunk_net.eval()
    dTNet.gate_nets[0].eval()
    dTNet.branch_nets[0].eval()
    mean, std = get_statistics(args.dataset.upper())
    up = torch.FloatTensor((1 - mean) / std).view(-1, 1, 1).to(device)
    down = torch.FloatTensor((0 - mean) / std).view(-1, 1, 1).to(device)

    ## test trunk_net,branch_net's Acc
    pbar = tqdm(data_loader, dynamic_ncols=True)
    for it, data in enumerate(pbar):
        if len(data)==2:
            inputs, targets = data
            sample_weights = torch.ones(size=targets.size(),dtype=torch.float)
        else:
            inputs, targets, sample_weights = data

        inputs, targets = inputs.to(device), targets.to(device)

        tot_samples += targets.size()[0]
        with torch.no_grad():
            trunk_out = dTNet.trunk_net(inputs.clone())
        tot_trunk_corr += cal_acc(trunk_out.data, targets, threshold = args.threshold, n_class =10 ).float().sum().item()
        with torch.no_grad():
            branch_out = dTNet.branch_nets[0](inputs)
        tot_branch_corr += cal_acc(branch_out.data, targets, threshold = args.threshold, n_class =10 ).float().sum().item()
    print("trunk_net nat_acc: %.4f, branch_net nat_acc: %.4f" %(tot_trunk_corr/tot_samples, tot_branch_corr/tot_samples))
    
    

    mean, std = get_statistics(args.dataset.upper())
    up = torch.FloatTensor((1 - mean) / std).view(-1, 1, 1).to(device)
    down = torch.FloatTensor((0 - mean) / std).view(-1, 1, 1).to(device)

    
    tot_corr = 0
    tot_ver_corr = 0
    tot_adv_corr = 0
    tot_samples = 0 

    if log_ind:
        data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=data_loader.batch_size,
                                    shuffle=False, num_workers=data_loader.num_workers, pin_memory=True,
                                    drop_last=False)
        csv_log = open(os.path.join(args.model_dir, "cert_log.csv"), mode='w')
        log_writer = csv.writer(csv_log, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        

    
                        
    attacker_gate = AttackPGD(dTNet.gate_nets[0], args.eps_test, step_size=args.eps_test / 4, num_steps=20, up=up, down=down)
    attacker_branch = AttackPGD(dTNet.branch_nets[0], args.eps_test, step_size=args.eps_test / 4, num_steps=20, up=up, down=down)
    attacker_trunk = AttackPGD(dTNet.trunk_net, args.eps_test, step_size=args.eps_test*std / 4, num_steps=20, up=up, down=down)    
    #with torch.enable_grad():        
    #    trunk_adv_acc = gen_adv_examples(dTNet.trunk_net, attacker_trunk, data_loader, device, None, fast=False, threshold=args.threshold, n_class =10)


   # print("trunk_adv_acc :%.4f " %trunk_adv_acc)
    pbar = tqdm(data_loader, dynamic_ncols=True)
    for it, data in enumerate(pbar):
        if len(data)==2:
            inputs, targets = data
            sample_weights = torch.ones(size=targets.size(),dtype=torch.float)
        else:
            inputs, targets, sample_weights = data

            
        inputs, targets = inputs.to(device), targets.to(device)
        tot_samples += inputs.size()[0]

        with torch.enable_grad():    
            perturb_gate_1 = attacker_gate.find(inputs.clone(), torch.ones_like(targets).int())       
            perturb_gate_0 = attacker_gate.find(inputs.clone(), torch.zeros_like(targets).int())              
            perturb_trunk = attacker_trunk.find(inputs.clone(), targets.clone())              
            perturb_branch = attacker_branch.find(inputs.clone(), targets.clone())
            
        all_perturb_gate_1 = dTNet.gate_nets[0](perturb_gate_1)
        all_perturb_gate_0 = dTNet.gate_nets[0](perturb_gate_0)
        all_perturb_trunk = dTNet.trunk_net(perturb_trunk)
        all_perturb_branch = dTNet.branch_nets[0](perturb_branch)
        all_choose = certify(dTNet.gate_nets[0], inputs, torch.ones_like(targets.view(1,-1)),args.eps_test, up, down, device , threshold=args.threshold, n_class =2)
        all_certified = certify(dTNet.branch_nets[0], inputs, targets, args.eps_test, up, down, device , threshold = 0.0, n_class=10)

        all_trunk_outputs = dTNet.trunk_net(inputs.clone())
        all_gate_outputs = dTNet.gate_nets[0](inputs)
        all_branch_outputs = dTNet.branch_nets[0](inputs)
        for n in range(inputs.shape[0]):
            target = targets[n:n + 1]
            input_img = inputs[n:n + 1]
            
            with torch.no_grad():
                outputs = all_perturb_branch[n]
                predicted = outputs.max(dim=0)[1]
                adv_branch = (predicted == target).squeeze()  

                outputs = all_perturb_trunk[n]
                predicted = outputs.max(dim=0)[1]
                adv_trunk = (predicted == target).squeeze()                    

                #choose = certify(dTNet.gate_nets[0], input_img, torch.ones_like(target.view(1,-1)).int(), args.eps_test, up, down, device , threshold=args.threshold, n_class =2).float().mean()
                gate_out = all_gate_outputs[n]
                choose = ((gate_out[1] - gate_out[0]).cpu() > args.threshold) & all_choose[n].cpu().bool() 

                if choose == 1:
                    #certified_acc = certify(dTNet.branch_nets[0], input_img, target, args.eps_test, up, down, device , threshold = 0.0, n_class=10).float().mean()
                    certified_acc = all_certified[n]
                    if certified_acc == 1:
                        tot_ver_corr +=1 
                    outputs = all_branch_outputs[n]
 
                    tot_corr += (outputs.max(dim=0)[1] == target).float().mean()
                    
                    outputs = all_perturb_gate_1[n]
                    predicted = outputs.max(dim=0)[1]
                    adv_gate = (predicted == 1).squeeze()
                    if adv_gate == 0:
                        tot_adv_corr += adv_trunk & adv_branch
                    else:
                        tot_adv_corr += adv_branch
                else:
                    outputs = all_trunk_outputs[n]
                    tot_corr += (outputs.max(dim=0)[1] == target).float().mean()   
                    
                    outputs = all_perturb_gate_0[n]
                    predicted = outputs.max(dim=0)[1]
                    adv_gate = (predicted == 0).squeeze()
                    if adv_gate == 0:
                        tot_adv_corr += adv_trunk & adv_branch
                    else:
                        tot_adv_corr += adv_trunk     
    print('nat_corr:%.4f, adv_corr:%.4f, ver_corr:%.4f' %(tot_corr/tot_samples, tot_adv_corr/tot_samples, tot_ver_corr/tot_samples))


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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, weight):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)


        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.weight is not None:
            true_dist *= self.weight.unsqueeze(0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# The hinge loss function is a combination of max_hinge_loss and average_hinge_loss.
def hinge(mix=0.50):
    def loss_fun(outputs, targets):
        return mix * outputs.max(dim=1)[0].clamp(min=0).mean() + (1 - mix) * outputs.clamp(min=0).mean()
    return loss_fun

class Loss():
    def __init__(self, loss, kappa ):
        self.loss = loss
        self.kappa = kappa
    def __call__(self, *args):
        margin_output = args[0] - torch.gather(args[0], dim=1, index=args[-1].view(-1, 1))
        if len(args) == 2:
            return self.loss(margin_output, args[-1])
        target  = args[-1].view(-1,1)
        return  (1-self.kappa) * ((args[1][:,1]-args[1][:,0]).clamp(min=0).view(1,-1)*(1-target)).float().mean() + self.kappa * ((-args[1][:,1]+args[1][:,0]).clamp(min=0).view(1,-1)*(target)).float().mean()
        
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

        #if ratio >= 1:
        p_norm = float('inf')
        #else:
        #    p_norm = args.p_start * math.exp(speed * ratio)
        set_p_norm(model, p_norm)

        if epoch2 > 0:
            ratio = min(max(num_batches(epoch - epoch0, minibatch) / num_batches(epoch2), 0), 1)
        else:
            ratio = 1.0
        set_eps(model, args.eps_train)
        loss.kappa = args.kappa

    return schedule

def train(net, loss_fun, epoch, trainloader, optimizer, schedule, logger, train_logger, device, print_freq, attacker, threshold):
    if logger is not None:
        logger.print('Epoch %d training start' % (epoch))
    net.train()
    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    train_loader_len = len(trainloader)
    #count_vars(net)
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

        outputs, worse_outputs = net(inputs, targets=targets.long())
        loss = loss_fun(outputs, worse_outputs, targets.long())

        with torch.no_grad():
            losses.update(loss.data.item(), targets.size(0))
            accs.update(cal_acc(outputs.data, targets, threshold=threshold, n_class =2 ).float().mean().item(), targets.size(0))
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
def test(net, loss_fun, epoch, testloader, logger, test_logger, device, print_freq, threshold):
    net.eval()
    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    test_loader_len = len(testloader)
    tot_1 = 0
    tot_0 = 0
    acc_1 = 0
    acc_0 = 0
    tot_samples = 0
    for batch_idx, data in enumerate(testloader):
        if len(data)==2:
            inputs, targets = data
            sample_weights = torch.ones(size=targets.size(),dtype=torch.float)
        else:
            inputs, targets, sample_weights = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)

        loss = loss_fun(outputs,targets.long())

        losses.update(loss.mean().item(), targets.size(0))
        accs.update(cal_acc(outputs, targets, threshold=threshold, n_class =2 ).float().mean().item(), targets.size(0))
        tot_samples +=targets.size()[0]
        tot_1 += (outputs.max(dim=1)[1] == 1).float().sum().item()
        tot_0 += (outputs.max(dim=1)[1] == 0).float().sum().item()
        acc_1 += ((outputs.max(dim=1)[1]==1) & (targets == 1)).float().sum().item()
        acc_0 += ((outputs.max(dim=1)[1]==0) & (targets == 0)).float().sum().item()
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
    print("percent_1:%.4f" %(tot_1/tot_samples))
    print("percent_0:%.4f" %(tot_0/tot_samples))
    print("acc_1:%.4f" %(acc_1/(tot_samples)))
    print("acc_0:%.4f" %(acc_0/(tot_samples)))
    return loss, acc





def get_gated_data_loaders(args, device, net, train_loader, test_loader, threshold=0):

    print("Get Branch_net's certified labels on Training set:")
    gate_train_loader, class_list = get_gated_data(args, device, net, train_loader, is_train=True, threshold=threshold)

    print("Get Branch_net's certified labels on Test set:")
    gate_test_loader, _ = get_gated_data(args, device, net, test_loader, is_train=False, threshold=threshold)

    return gate_train_loader, gate_test_loader

def get_gated_data(args, device, net, data_loader,is_train=True, threshold=0):
    save_p = get_p_norm(net)
    set_p_norm(net, float('inf'))
    net.eval()
    class_list = None

    data_set_tmp = MyVisionDataset.from_idx(data_loader.dataset, np.ones_like(data_loader.dataset.targets).astype(bool))
    data_loader_tmp = torch.utils.data.DataLoader(data_set_tmp, batch_size=data_loader.batch_size,
                                                  shuffle=False, num_workers=data_loader.num_workers, pin_memory=True, drop_last=False)


    mean, std = get_statistics(args.dataset.upper())        

    up = torch.FloatTensor((1 - mean) / std).view(-1, 1, 1).to(device)
    down = torch.FloatTensor((0 - mean) / std).view(-1, 1, 1).to(device)


    if is_train:
        threshold_cert = certified_test(net, args.eps_test/2, up, down, data_loader, None, device, threshold=threshold, n_class =10)
    else:
        threshold_cert = certified_test(net, args.eps_test, up, down, data_loader, None, device, threshold=threshold, n_class =10)

        

    gate_target = threshold_cert.cpu()
    print("gate_percent:%.4f" %(gate_target.float().mean()))

    data_set = MyVisionDataset.from_idx_and_targets(data_loader.dataset, torch.ones_like(gate_target).numpy(),
                                                        gate_target.float(), ["gate_easy_cert", "gate_difficult_cert"])
                                
  
    if len(data_set) == 0:
        return None, class_list
    else:
        return torch.utils.data.DataLoader(data_set, batch_size=args.train_batch if is_train else args.test_batch,
                                       shuffle=-1 if is_train else False, num_workers=data_loader.num_workers,
                                       pin_memory=True, drop_last=False), class_list
    set_p_norm(net, save_p)

def cal_acc(outputs, targets, threshold = 0.0, n_class = 2):
    if n_class != 2:
        return (outputs.max(dim=1)[1] == targets)
    else:
        predicted = (outputs[:,1] - outputs[:,0]) > threshold 
        return (predicted == targets)

def certify(net, inputs, targets ,eps, up, down, device, threshold=0.0, n_class = 2):
    save_p = get_p_norm(net)
    save_eps = get_eps(net)
    set_eps(net, eps)
    set_p_norm(net, float('inf'))
    net.eval()


    targets = torch.squeeze(targets, dim = 0)
    lower = torch.max(inputs - eps, down)
    upper = torch.min(inputs + eps, up)
    outputs=net(inputs, lower=lower, upper=upper, targets=targets.long())[1]
    labels=targets


    if n_class != 2:
        res = (outputs.max(dim=1)[1] == labels)
    else:
    
        predicted = (outputs[:,1] - outputs[:,0]) > threshold 
        res = (predicted==labels)

    set_p_norm(net, save_p)
    set_eps(net, save_eps)

    return res

def certified_test(net, eps, up, down, testloader, logger, device, threshold=0.0, n_class = 2):
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
    if n_class != 2:
        res = (outputs.max(dim=1)[1] == labels)
    else:
        predicted = (outputs[:,1] - outputs[:,0]) > threshold 
        res = (predicted==labels)

    if logger is not None:
        logger.print(' certified acc ' + f'{res.float().mean().item():.4f}')

    set_p_norm(net, save_p)
    set_eps(net, save_eps)
    return res

def gen_adv_examples(model, attacker, test_loader, device, logger, fast=False , threshold=0.0, n_class = 2):
    model.eval()
    correct = 0
    tot_num = 0
    size = len(test_loader)
    
    mean = [0.4914, 0.4822, 0.4465]
    std = 0.2009
    ## Use models like Resnet, need to un_normalize.
    def normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
        return t
    def un_normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] * std) + mean[0]
        t[:, 1, :, :] = (t[:, 1, :, :] * std) + mean[1]
        t[:, 2, :, :] = (t[:, 2, :, :] * std) + mean[2]
        return t
    
    for batch_idx, data in enumerate(test_loader):
        if len(data)==2:
            inputs, targets = data
            sample_weights = torch.ones(size=targets.size(),dtype=torch.float)
        else:
            inputs, targets, sample_weights = data
        inputs = inputs.to(device)
        targets = targets.to(device,dtype=torch.long)
        result = torch.ones(targets.size(0), dtype=torch.bool, device=targets.device)
        #inputs = un_normalize(inputs)
        
        for i in range(1):
            perturb = attacker.find(inputs, targets)
            with torch.no_grad():
                outputs = model(perturb)
                if n_class ==2:
                    predicted = (outputs[:,1] - outputs[:,0]) > threshold 
                else:
                    predicted = torch.max(outputs.data, 1)[1]
                result &= (predicted == targets)
        correct += result.float().sum().item()
        tot_num += inputs.size(0)
        if fast and batch_idx * 10 >= size: break

    acc = correct / tot_num

    if logger is not None:
        logger.print('adversarial attack acc ' + f'{acc:.4f}')
    return acc


def main():
    args = get_args()
    run(args=args)


if __name__ == '__main__':
    main()
