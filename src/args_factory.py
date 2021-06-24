import argparse
import re

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args():
    parser = argparse.ArgumentParser(description='ACE compositional architectures for boosting certified robustness.')
    
    # Basic arguments
    parser.add_argument('--dataset', default='cifar10', help='dataset to use')
    #parser.add_argument('--net', required=True, type=str, help='network to use')
    parser.add_argument('--train-batch', default=100, type=int, help='batch size for training')
    parser.add_argument('--test-batch', default=100, type=int, help='batch size for testing')
    parser.add_argument('--epochs', default='0,50,50,150,200', type=str) # epoch1-epoch3: inc eps; epoch2-epoch4: inc p
    parser.add_argument('--test-freq', default=50, type=int, help='frequency of testing')
    parser.add_argument('--num-workers', default=16, type=int, help="Number of workers for data loaders")
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N', help='print frequency')
    #model-loading
    parser.add_argument('--load-model', default=None, type=str, help='model to load')

    # Optimizer and learning rate scheduling
    parser.add_argument('--p-start', default=8.0, type=float)
    parser.add_argument('--p-end', default=1000.0, type=float)
    parser.add_argument('--kappa', default=1.0, type=float)

    parser.add_argument('--eps-train', default=None, type=float)
    parser.add_argument('--eps-test', default=None, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
    parser.add_argument('--epsilon', default=1e-10, type=float)
    parser.add_argument('--wd', default=0.0, type=float)


    parser.add_argument('--visualize', action='store_true')

    # Metadata
    parser.add_argument('--experiment-name', default='dev', type=str, help='name of the experiment')
    parser.add_argument('--exp-id', default=1, type=int, help='name of the experiment')
    parser.add_argument('--no-cuda', action='store_true', help='whether to use only cpu')
    parser.add_argument('--root-dir', required=False, default='./', type=str, help='directory to store the data')
    parser.add_argument('--result-dir', default='./result/', type=str)
    ## SpecNet - Architecture
    parser.add_argument('--n-branches', default=1, type=int, help="Number of branches")
    parser.add_argument('--trunk-nets', default=None, type=str, nargs="+", help="Specify trunk net architectures")
    parser.add_argument('--branch-nets', default="LipNet", type=str, nargs="+", help="Specify branch net architectures")
    parser.add_argument('--gate-nets', default=None, type=str, nargs="+", help="Specify gate net architectures")
    parser.add_argument('--loss', default='cross_entropy', type=str) #cross_entropy, hinge
    ## SpecNet-Loading
    parser.add_argument('--load-branch-model', default=None, type=str, nargs="+", help="Model to load on branches. 'True' will load same model as on trunk ")
    parser.add_argument('--load-gate-model', default=None, type=str, nargs="+", help="Model to load on branches. 'True' will load same model as on trunk ")
    parser.add_argument('--load-trunk-model', default=None, type=str, help="model to load on trunk")

    ## SpecNet - Gating Objective
    parser.add_argument('--gate-type', default="net", choices=["net", "entropy"], help="Chose whether gating should be based on entropy or a network")
    parser.add_argument('--gate-threshold', default=0.5, type=float, help="Threshold for gate network selection. Entropy is negative.")
    parser.add_argument('--retrain-branch', default=False, type=boolean_string, help="Retrain branch with weighted dataset after gate training, only if gate_on_trunk is False")

    parser.add_argument('--mode', default="train", type=str)
    args = parser.parse_args()


    if args.load_branch_model == "True":
        args.load_branch_model = args.load_trunk_model

    if args.gate_type != "net":
        args.gate_feature_extraction = None
        print("No selection network was loaded, but feature extraction activated!")

    return args
