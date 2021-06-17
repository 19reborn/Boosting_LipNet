import torch
import re
from src.utils import get_net, init_slopes
from src.zonotope import HybridZonotope, clamp_image
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
                                                  net_dim=args.cert_net_dim)
            if args.load_branch_model is not None:
                args.load_branch_model = args.load_branch_model[0]
                print(args.load_branch_model)
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
                print("=> loaded '{}'".format(args.load_branch_model))

            if gate_type == "net":
                self.gate_nets[exit_idx] = get_net(device, dataset, gate_net_names[i], input_size, input_channel, 1,
                                                   load_model=None if args.load_gate_model is None else args.load_gate_model[i],
                                                   net_dim=args.cert_net_dim)
                if args.load_gate_model is not None:
                    assert os.path.isfile(args.load_gate_model)
                    checkpoint = torch.load(args.load_branch_model, map_location=lambda storage, loc: storage.cuda(0))
                    state_dict = checkpoint['state_dict']
                    if next(iter(state_dict))[0:7] == 'module.' and not parallel:
                        new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                        state_dict = new_state_dict
                    elif next(iter(state_dict))[0:7] != 'module.' and parallel:
                        new_state_dict = OrderedDict([('module.' + k, v) for k, v in state_dict.items()])
                        state_dict = new_state_dict
                    self.gate_nets[exit_idx].load_state_dict(state_dict)
                    #optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded '{}'".format(args.load_branch_model))
            else:
                self.gate_nets[exit_idx] = SeqNet(Sequential(*[*self.branch_nets[exit_idx].blocks, Entropy(n_class, low_mem=True, neg=True)]))
                self.gate_nets[exit_idx].determine_dims(torch.randn((2, input_channel, input_size, input_size), dtype=torch.float).to(device))
                init_slopes(self.gate_nets[exit_idx], device, trainable=False)

            self.add_module("gateNet_{}".format(exit_idx), self.gate_nets[exit_idx])
            self.add_module("branchNet_{}".format(exit_idx), self.branch_nets[exit_idx])

    '''
        ## 整体load
        if args.load_model is not None:
            old_state = self.state_dict()
            load_state = torch.load(args.load_model)

            old_state.update({k:v.view(old_state[k].shape) for k,v in load_state.items() if
                              k in old_state and (
                              (k.startswith("trunk") and args.load_trunk_model is None)
                              or (k.startswith("gate") and args.load_gate_model is None)
                              or (k.startswith("branch") and args.load_branch_model is None))})
            missing_keys, extra_keys = self.load_state_dict(old_state, strict=False)
            assert len([x for x in missing_keys if "gateNet" in x or "branchNet" in x]) == 0
            print("Whole model loaded from %s" % args.load_model)

            ## Trunk and branch nets have to be loaded after the whole model
            if args.load_trunk_model is not None:
                checkpoint = torch.load(args.load_trunk_model, map_location=lambda storage, loc: storage.cuda(gpu))
                state_dict = checkpoint['state_dict']
                if next(iter(state_dict))[0:7] == 'module.' and not parallel:
                    new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                    state_dict = new_state_dict
                elif next(iter(state_dict))[0:7] != 'module.' and parallel:
                    new_state_dict = OrderedDict([('module.' + k, v) for k, v in state_dict.items()])
                    state_dict = new_state_dict
                model.load_state_dict(state_dict)
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded '{}'".format(args.checkpoint))
                if parallel:
                    torch.distributed.barrier()

        if (args.load_model is not None or args.load_gate_model is not None) and args.gate_feature_extraction is not None:
            for i, net in enumerate(self.gate_nets.values()):
                extraction_layer = [ii for ii in range(len(net.blocks)) if isinstance(net.blocks[ii],Linear)]
                extraction_layer = extraction_layer[-min(len(extraction_layer),args.gate_feature_extraction)]
                net.freeze(extraction_layer-1)
        '''

    @staticmethod
    def get_deepTrunk_net(args, device, lossFn, evalFn, input_size, input_channel, n_class, trunk_net=None):
        if trunk_net is None:
            if args.net != "None":
                trunk_net = get_net(device, args.dataset, args.net, input_size, input_channel, n_class,
                                                 load_model=args.load_trunk_model)
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

                gate_nets = [self.gate_cnets[j] for j in range(self.n_branches) if j <= i]
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
                            adv_gate = gate_net.net(adv_latent)
                            ce_loss_gate = gate_net.lossFn(adv_gate, gate_target[:,j])
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
            adv_loss = self.lossFn(adv_outs[torch.stack(exit_mask, dim=1)], targets)
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

            x_out += [self.branch_nets[exit_idx].forward(x=x.clone())]
            x_gate += [self.gate_nets[exit_idx].forward(x=x.clone())]

            selected += [(x_gate[-1].squeeze(dim=1) >= self.threshold[exit_idx]).__and__(remaining)]

            exit_ids[selected[-1]] = exit_idx
            remaining[selected[-1]] = False

        exit_ids[remaining] = -1
        selected += [remaining]

        if self.trunk_net is not None:
            x_out += [self.trunk_net.forward(x=x)]

        x_out = torch.stack(x_out, dim=1)
        x_gate = torch.stack(x_gate, dim=1)

        return x_out, x_gate, selected, exit_ids
