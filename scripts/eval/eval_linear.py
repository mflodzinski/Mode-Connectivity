"""Evaluate linear interpolation between two endpoints.
Based on eval_curve.py but for simple linear interpolation.
"""
import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F
import sys

# Add external repo to path
sys.path.insert(0, 'external/dnn-mode-connectivity')
import data
import models
import utils

parser = argparse.ArgumentParser(description='Linear interpolation evaluation')
parser.add_argument('--dir', type=str, required=True, metavar='DIR',
                    help='directory to save results')
parser.add_argument('--init_start', type=str, required=True, metavar='PATH',
                    help='checkpoint for start point')
parser.add_argument('--init_end', type=str, required=True, metavar='PATH',
                    help='checkpoint for end point')
parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points (default: 61)')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET')
parser.add_argument('--use_test', action='store_true')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH')
parser.add_argument('--batch_size', type=int, default=128, metavar='N')
parser.add_argument('--num_workers', type=int, default=4, metavar='N')
parser.add_argument('--model', type=str, required=True, metavar='MODEL')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

loaders, num_classes = data.loaders(
    args.dataset, args.data_path, args.batch_size, args.num_workers,
    args.transform, args.use_test, shuffle_train=False
)

# Load model architecture
architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
model = model.to(device)

# Load endpoints
checkpoint_start = torch.load(args.init_start, map_location=device)
checkpoint_end = torch.load(args.init_end, map_location=device)

weights_start = checkpoint_start['model_state']
weights_end = checkpoint_end['model_state']

T = args.num_points
ts = np.linspace(0.0, 1.0, T)
tr_loss, tr_nll, tr_acc, tr_err = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
te_loss, te_nll, te_acc, te_err = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
l2_norm = np.zeros(T)

columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

print('Evaluating linear interpolation...')
for i, t_value in enumerate(ts):
    # Linear interpolation: w(t) = (1-t)*w_start + t*w_end
    weights = {}
    for key in weights_start.keys():
        weights[key] = (1.0 - t_value) * weights_start[key] + t_value * weights_end[key]

    model.load_state_dict(weights)
    model.eval()

    utils.update_bn(loaders['train'], model)
    tr_res = utils.test(loaders['train'], model, F.cross_entropy)
    te_res = utils.test(loaders['test'], model, F.cross_entropy)

    tr_loss[i] = tr_res['loss']
    tr_nll[i] = tr_res['nll']
    tr_acc[i] = tr_res['accuracy']
    tr_err[i] = 100.0 - tr_acc[i]
    te_loss[i] = te_res['loss']
    te_nll[i] = te_res['nll']
    te_acc[i] = te_res['accuracy']
    te_err[i] = 100.0 - te_acc[i]

    # Compute L2 norm of weights at this point
    l2 = 0.0
    for param in model.parameters():
        l2 += torch.sum(param ** 2).item()
    l2_norm[i] = np.sqrt(l2)

    values = [t_value, tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i]]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

# Save results
np.savez(
    os.path.join(args.dir, 'linear.npz'),
    ts=ts,
    l2_norm=l2_norm,
    tr_loss=tr_loss,
    tr_nll=tr_nll,
    tr_acc=tr_acc,
    tr_err=tr_err,
    te_loss=te_loss,
    te_nll=te_nll,
    te_acc=te_acc,
    te_err=te_err,
)

print(f"\nResults saved to: {os.path.join(args.dir, 'linear.npz')}")
print(f"Max test error: {np.max(te_err):.2f}% at t={ts[np.argmax(te_err)]:.2f}")
