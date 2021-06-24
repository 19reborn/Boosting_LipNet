import torch
import torch.nn.functional as F
import re
from src.utils import get_net, get_trunk_net, load_net_state
from src.zonotope import HybridZonotope, clamp_image
from src.relaxed_networks import CombinedNetwork
from model.model import Model, set_eps, get_eps
from model.norm_dist import set_p_norm, get_p_norm
import os

class MyDeepTrunkNet(torch.nn.Module):
    def __init__(self, device, args, dataset, trunk_net, input_size, input_channel, n_class, n_branches, gate_type,
                 branch_net_names, gate_net_names, evalFn, lossFn):
        super(MyDeepTrunkNet, self).__init__()
        self.dataset = dataset
        self.input_size = input_size
        self.input_channel = input_channel
        self.n_class = n_class
        self.gate_type = gate_type
        self.n_branches = n_branches
        self.trunk_net = trunk_net
        self.evalFn = evalFn
        self.lossFn = lossFn

        assert gate_type in ["entropy", "net"], f"Unknown gate mode: {gate_type:s}"

        self.exit_ids = [-1] + list(range(n_branches))

        self.threshold = {exit_idx: args.gate_threshold for exit_idx in self.exit_ids[1:]}
        self.gate_nets = {}
        self.branch_nets = {}

        if len(branch_net_names) != n_branches:
            print("Number of branches does not match branch net names")
            branch_net_names = n_branches * branch_net_names[0:1]

        if gate_net_names is None:
            gate_net_names = branch_net_names
        elif len(gate_net_names) != n_branches:
            print("Number of branches does not match gate net names")
            gate_net_names = n_branches * gate_net_names[0:1]

        if args.load_branch_model is not None and len(args.load_branch_model) != n_branches:
            args.load_branch_model = n_branches * args.load_branch_model[0:1]
        if args.load_gate_model is not None and len(args.load_gate_model) != n_branches:
            args.load_gate_model = n_branches * args.load_gate_model[0:1]

        parallel = False

        for i, branch_net_name in zip(range(n_branches), branch_net_names):
            exit_idx = self.exit_ids[i+1]
            self.branch_nets[exit_idx] = get_net(device, dataset, branch_net_name, input_size, input_channel, n_class,
                                                 load_model=None if args.load_branch_model is None else args.load_branch_model[i],
                                                  net_dim=None)
            if args.load_branch_model is not None:
                args.load_branch_model = args.load_branch_model[0]

                assert os.path.isfile(args.load_branch_model)
                checkpoint = torch.load(args.load_branch_model, map_location=lambda storage, loc: storage.cuda(0))
                state_dict = checkpoint['state_dict']
                if next(iter(state_dict))[0:7] == 'module.' and not parallel:
                    new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                    state_dict = new_state_dict
                elif next(iter(state_dict))[0:7] != 'module.' and parallel:
                    new_state_dict = OrderedDict([('module.' + k, v) for k, v in state_dict.items()])
                    state_dict = new_state_dict
                self.branch_nets[exit_idx].load_state_dict(state_dict)
                #optimizer.load_state_dict(checkpoint['optimizer'])
                print("branch_net => loaded '{}'".format(args.load_branch_model))

            if gate_type == "net":
                self.gate_nets[exit_idx] = get_net(device, dataset, gate_net_names[i], input_size, input_channel, 2,
                                                   load_model=None if args.load_gate_model is None else args.load_gate_model[i],
                                                   net_dim=None)
                if args.load_gate_model is not None:
                    args.load_gate_model = args.load_gate_model[0]

                    assert os.path.isfile(args.load_gate_model)
                    checkpoint = torch.load(args.load_gate_model, map_location=lambda storage, loc: storage.cuda(0))
                    state_dict = checkpoint['state_dict']
                    if next(iter(state_dict))[0:7] == 'module.' and not parallel:
                        new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                        state_dict = new_state_dict
                    elif next(iter(state_dict))[0:7] != 'module.' and parallel:
                        new_state_dict = OrderedDict([('module.' + k, v) for k, v in state_dict.items()])
                        state_dict = new_state_dict

                    state_dict = {k: v for k, v in state_dict.items() if v.size() == self.gate_nets[exit_idx].state_dict()[k].size() and "feature" in k}
                    #print(self.gate_nets[exit_idx].state_dict[predictor.fc1.bias])
                    #optimizer.load_state_dict(checkpoint['optimizer'])
                    print("gate_net => loaded '{}'".format(args.load_gate_model))
                    for p_name, params in self.gate_nets[exit_idx].named_parameters():
                        if "feature" in p_name:
                            params.requires_grad = False

            else:
                self.gate_nets[exit_idx] = SeqNet(Sequential(*[*self.branch_nets[exit_idx].blocks, Entropy(n_class, low_mem=True, neg=True)]))
                self.gate_nets[exit_idx].determine_dims(torch.randn((2, input_channel, input_size, input_size), dtype=torch.float).to(device))

            self.add_module("gateNet_{}".format(exit_idx), self.gate_nets[exit_idx])
            self.add_module("branchNet_{}".format(exit_idx), self.branch_nets[exit_idx])

        ## load_trunk_model
            if args.load_trunk_model is not None:
                load_net_state(self.trunk_net, args.load_trunk_model)
                print("trunk_net => loaded '{}'".format(args.load_trunk_model))

            self.trunk_cnet = CombinedNetwork.get_cnet(self.trunk_net, device, lossFn, evalFn, n_rand_proj=50, no_r_net=True, input_channels=self.input_channel)

    @staticmethod
    def get_deepTrunk_net(args, device, lossFn, evalFn, input_size, input_channel, n_class):
        if args.trunk_nets is not None:
            trunk_net = get_trunk_net(device, args.dataset, args.trunk_nets[0], input_size, input_channel, n_class,
                                                 load_model=args.load_trunk_model,
                                                  net_dim=None)
        else:
            trunk_net = None
        specNet = MyDeepTrunkNet(device, args, args.dataset, trunk_net, input_size, input_channel, n_class,
                                 args.n_branches, args.gate_type, args.branch_nets, args.gate_nets, evalFn, lossFn)

        return specNet

    def get_adv_loss(self, inputs, targets, adv_attack_test, mode="mixed"):
        curr_head, A_0 = clamp_image(inputs, adv_attack_test.eps)
        x_out, x_gate, selected, exit_ids = self(inputs)
        with torch.enable_grad():
            adv_input = inputs.clone()
            adv_loss = torch.zeros_like(targets).float()
            adv_found = torch.zeros_like(targets).bool()
            adv_errors = torch.FloatTensor(curr_head.shape).unsqueeze(1). \
                to(curr_head.device).uniform_(-1, 1).requires_grad_(True)
            select_mask_adv = torch.zeros_like(torch.stack(selected, dim=1))
            select_out_max = -1000*torch.ones_like(x_gate)
            select_out_min = 1000 * torch.ones_like(x_gate)
            for i, exit_idx in enumerate(self.exit_ids[1:] + [-1]):
                
                branch_net = self.trunk_net if exit_idx == -1 else self.branch_nets[exit_idx]

                gate_nets = self.gate_nets
                gate_target = torch.zeros((len(targets), self.n_branches), dtype=torch.float, device=targets.device)
                if exit_idx != -1:
                    gate_target[:, i] = 1

                for it in range(adv_attack_test.n_steps+1):
                    curr_errors = A_0.unsqueeze(1) * adv_errors
                    adv_latent = curr_head + curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:]).sum(dim=1)

                    if it == adv_attack_test.n_steps:
                        adv_outs, adv_gate_outs, exit_mask, _ = self.forward(adv_latent)
                        select_out_max = adv_gate_outs.max(select_out_max)
                        select_out_min = adv_gate_outs.min(select_out_min)

                        select_mask_adv = select_mask_adv.__or__(torch.stack(exit_mask, dim=1))
                        if branch_net is None:
                            break
                        adv_branch = adv_outs[:,i]
                        ce_loss = self.lossFn(adv_branch, targets)
                        break

                    if mode == "reachability" or (branch_net is None):
                        ce_loss = 0
                    else:
                        adv_branch = branch_net(adv_latent)
                        ce_loss = self.lossFn(adv_branch, targets)
                    if not mode == "classification":
                        for j, gate_net in enumerate(gate_nets):

                            gate_net = self.gate_nets[gate_net]
                            adv_gate = gate_net(adv_latent)
                            ce_loss_gate = self.lossFn(adv_gate, gate_target[:,j].long())
                            weight = torch.nn.functional.softplus(
                                     2 * (adv_gate.view(-1) - gate_net.threshold_min)
                                     * (1 - 2 * gate_target[:, j])
                                     + 0.5).detach()
                            ce_loss -= ce_loss_gate * weight

                    self.zero_grad()
                    ce_loss.sum(dim=0).backward()
                    adv_errors.data = torch.clamp(adv_errors.data + adv_attack_test.step_size * adv_errors.grad.sign(), -1, 1)
                    adv_errors.grad.zero_()
                adv_input = torch.where(((adv_loss < ce_loss).__and__(exit_mask[i])).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1), adv_latent, adv_input)
                adv_found = (adv_loss < ce_loss).__and__(exit_mask[i]).__or__(adv_found)
                adv_loss = torch.where((adv_loss < ce_loss).__and__(exit_mask[i]), ce_loss, adv_loss)
            adv_input = torch.where(adv_found.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1), adv_input, adv_latent)
            adv_latent = adv_input.detach()
            adv_outs, _, exit_mask, adv_branches = self.forward(adv_latent)
            if self.trunk_net is None:
                exit_mask[-2] = exit_mask[-2].__or__(exit_mask[-1])
                exit_mask = exit_mask[:-1]
            adv_loss = self.lossFn(adv_outs[torch.stack(exit_mask, dim=1)], targets.long())
            adv_ok = targets.eq(self.evalFn(adv_outs[torch.stack(exit_mask,dim=1)])).detach()
        if mode == "reachability":
            return select_mask_adv, select_out_min, select_out_max
        return adv_loss, adv_ok, adv_outs, adv_latent, adv_branches

    def forward(self, x):
        x_out = []
        x_gate = []
        selected = []
        remaining = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        exit_ids = torch.zeros(x.size(0), dtype=torch.int, device=x.device)


        for exit_idx in self.exit_ids[1:]:

            x_out += [self.branch_nets[exit_idx](x=x.clone())]
            x_gate += [self.gate_nets[exit_idx](x=x.clone())]
            
            selected += [(F.softmax(x_gate[-1],1)[:,1] >= self.threshold[exit_idx]).__and__(remaining)]
            #selected += [(x_gate[-1].squeeze(dim=1) >= self.threshold[exit_idx]).__and__(remaining)]

            exit_ids[selected[-1]] = exit_idx
            remaining[selected[-1]] = False

        exit_ids[remaining] = -1
        selected += [remaining]

        if self.trunk_net is not None:
            x_out += [self.trunk_net.forward(x=x)]

        x_out = torch.stack(x_out, dim=1)
        x_gate = torch.stack(x_gate, dim=1)

        return x_out, x_gate, selected, exit_ids
