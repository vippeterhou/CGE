from CGETrans import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'citeseer')
parser.add_argument('--embedding_size', type = int, default = 50)
parser.add_argument('--valida_rate', type = float, default = 0.5)
parser.add_argument('--window_size', type = int, default = 3)
parser.add_argument('--path_length', type = int, default = 10)
parser.add_argument('--super_pairs', type = int, default = 20)
parser.add_argument('--step_learning_rate', type = float, default = 0.1)
parser.add_argument('--sup_learning_rate', type = float, default = 1e-3)
parser.add_argument('--sup_batch_size', type = int, default = 200)
parser.add_argument('--unsup_batch_size', type = int, default = 20)
parser.add_argument('--sup_iters', type = int, default = 2000)
parser.add_argument('--unsup_iters', type = int, default = 10000)
args = parser.parse_args()

for arg in vars(args):
    print arg, getattr(args, arg)
print "Please check the args..."

m = CGETransModel(args)
m.loadDataset()
m.buildModel()
m.supervisedTrain(args.sup_iters)
m.unsupervisedTrain(args.unsup_iters)
