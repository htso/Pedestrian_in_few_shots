import argparse
import os
import time
import sys

from motdata import MOTDataset
from motmodel import Statistician
from motplot import save_test_grid
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

# NOTE : The pedestrian snapshot data in MOT16PERSON/PERSON1/VIDEO
# have been shaped into image of size
#
#                (160, 96, 3)
#
# So, height=160, width=96, channel=3
#


# command line args
parser = argparse.ArgumentParser(description='Neural Statistician Experiment on MOT16 Dataset')

parser.add_argument('--data-dir', required=True, type=str, default=None,
                    help='location of formatted Omniglot data')
parser.add_argument('--output-dir', required=True, type=str, default=None,
                    help='output directory for checkpoints and figures')

parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs for training (default: 300)')

parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--clip-gradients', type=bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')

parser.add_argument('--batch-size', type=int, default=2,
                    help='batch size (of datasets) for training (default: 64)')
parser.add_argument('--K', type=int, default=5,
                    help='the K in K-shot learning, or number of sample images per dataset (default: 5)')

parser.add_argument('--c-dim', type=int, default=512,
                    help='dimension of c variables (default: 512)')
parser.add_argument('--n-hidden-statistic', type=int, default=1,
                    help='number of hidden layers in statistic network modules '
                         '(default: 1)')
parser.add_argument('--hidden-dim-statistic', type=int, default=1000,
                    help='dimension of hidden layers in statistic network (default: 1000)')
parser.add_argument('--n-stochastic', type=int, default=1,
                    help='number of z variables in hierarchy (default: 1)')
parser.add_argument('--z-dim', type=int, default=16,
                    help='dimension of z variables (default: 16)')
parser.add_argument('--n-hidden', type=int, default=1,
                    help='number of hidden layers in modules outside statistic network '
                         '(default: 1)')
parser.add_argument('--hidden-dim', type=int, default=1000,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 1000)')

parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all trainable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--viz-interval', type=int, default=-1,
                    help='number of epochs between visualizing context space '
                         '(default: -1 (only visualize last epoch))')
parser.add_argument('--save_interval', type=int, default=-1,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
args = parser.parse_args()
assert (args.data_dir is not None) and (args.output_dir is not None)
os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

# experiment start time
time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")


def run(model, optimizer, loaders, datasets, width, height):
    train_dataset, test_dataset = datasets
    train_loader, test_loader = loaders   

    # tr_batch = next(iter(train_loader))
    # tt_batch = next(iter(test_loader))

    viz_interval = args.epochs if args.viz_interval == -1 else args.viz_interval
    save_interval = args.epochs if args.save_interval == -1 else args.save_interval

    # initial weighting for two term loss
    alpha = 1
    # main training loop
    tbar = tqdm(range(args.epochs))
    for epoch in tbar:

        # train step (iterate once over training data)
        model.train()
        running_vlb = 0
        for batch in train_loader:
            inputs = Variable(batch.cuda())
            #print('run inputs : ', inputs.shape)
            vlb = model.step(inputs, alpha, optimizer, clip_gradients=args.clip_gradients)
            running_vlb += vlb

        # update running lower bound
        running_vlb /= (len(train_dataset) // args.batch_size)
        s = "VLB: {:.3f}".format(running_vlb)
        tbar.set_description(s)

        # reduce weight
        alpha *= 0.5

        # evaluate on test set by sampling conditioned on contexts
        model.eval()
        if (epoch + 1) % viz_interval == 0:
            print(' ===== test batch ======')
            test_batch = next(iter(test_loader))
            filename = 'testbatch-' + time_stamp + '-grid-{}.png'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, 'figures/' + filename)
            inputs = Variable(test_batch.cuda())
            samples = model.sample_conditioned(inputs)
            save_test_grid(inputs, samples, save_path, n=10, width=width, height=height)
            
            print(' ===== train batch ======')
            tr_iter = iter(train_loader)
            try:
                train_batch = next(tr_iter)
            except StopIteration:
                tr_iter = iter(train_loader)
                train_batch = next(tr_iter)

            filename = 'trainbatch-' + time_stamp + '-grid-{}.png'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, 'figures/' + filename)
            inputs = Variable(train_batch.cuda())
            samples = model.sample_conditioned(inputs)
            save_test_grid(inputs, samples, save_path, n=10, width=width, height=height)

        # checkpoint model at intervals
        if (epoch + 1) % save_interval == 0:
            filename = time_stamp + '-{}.m'.format(epoch + 1)
            print('output_dir : ', args.output_dir)
            #save_path = os.path.join(args.output_dir, '/checkpoints/' + filename)
            #print('save_path : ', save_path)
            save_path = args.output_dir + '/checkpoints/' + filename
            model.save(optimizer, save_path)

def main():

    print('data_dir : ', args.data_dir)
    print('batch_size = ', args.batch_size)
    print('K = ', args.K)

    train_val_test_split = {'train':0.6, 'valid':0.3, 'test':0.1}
 
    # create datasets
    train_dataset = MOTDataset(data_dir=args.data_dir, train_val_test_split=train_val_test_split, split='train', K=args.K, verbose=False)
    test_dataset = MOTDataset(data_dir=args.data_dir, train_val_test_split=train_val_test_split, split='valid',  K=args.K, verbose=False)
    datasets = (train_dataset, test_dataset)

    height = train_dataset.height
    width = train_dataset.width
    channels = train_dataset.channels
    print('height :', height)
    print('wdith :', width)
    print('channels :', channels)

    # create loaders
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
    loaders = (train_loader, test_loader)
    
    # print(' ==== first train batch ========')
    # b1 = next(iter(train_loader))
    # print(' ==== second train batch ========')
    # b2 = next(iter(train_loader))
    # print(' ==== first valid batch ========')
    # b3 = next(iter(test_loader))
    # print(' ==== second valid batch ========')
    # b4 = next(iter(test_loader))
    # #b2 = next(iter(train_loader))
    #sys.exit()

    #print('b1 ', b1.shape)
    #path = '/mnt/WanChai/Dropbox/Tensorflow-Mostly/FewShotLearning/edwards-neural-statistician_ht/mot16/ShowMeBatch.png'
    #save_test_grid(b1, b2, path, n=10, width=width, height=height)
    #sys.exit()

    # create model
    shrunk_height = int(float(height) / 2**4)
    shrunk_width = int(float(width) / 2**4)
    n_features = 256 * shrunk_height * shrunk_width  # output shape of convolutional encoder
    # 160 / 2^4 = 10, 4x down sampling in 4 Conv layers
    # 96 / 2^4 = 6
    # To see how this comes about, see the code in class SharedConvolutionalEncoder(nn.Module)
    
    model_kwargs = {
        'batch_size': args.batch_size,
        'K': args.K,
        'n_features': n_features,
        'c_dim': args.c_dim,
        'n_hidden_statistic': args.n_hidden_statistic,
        'hidden_dim_statistic': args.hidden_dim_statistic,
        'n_stochastic': args.n_stochastic,
        'z_dim': args.z_dim,
        'n_hidden': args.n_hidden,
        'hidden_dim': args.hidden_dim,
        'nonlinearity': F.elu,
        'print_vars': args.print_vars,
        'width': width,
        'height': height
    }
    model = Statistician(**model_kwargs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    run(model, optimizer, loaders, datasets, width=width, height=height)


if __name__ == '__main__':
    main()
