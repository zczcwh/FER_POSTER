import torch
from torch._C import default_generator
import torchvision
import numpy as np
import random
import data_preprocessing.data_loader as dl
import argparse
from models.resnet import resnet56, resnet18
from models.resnet_gradaug import resnet56 as resnet56_gradaug
from models.resnet_gradaug import resnet18 as resnet18_gradaug
from models.resnet_feddepth import resnet56 as resnet56_feddepth
from models.resnet_feddepth import resnet18 as resnet18_feddepth
from models.resnet_ours import resnet56 as resnet56_ours
from models.resnet_ours import resnet18 as resnet18_ours
from torch.multiprocessing import Pool, Process, set_start_method, Queue
import logging
import os
from collections import defaultdict
import time

# methods
import methods.fedavg as fedavg
import methods.gradaug as gradaug
import methods.fedprox as fedprox
import methods.moon as moon
import methods.feddepth as feddepth
import methods.mixup as mixup
import methods.ours as ours
import data_preprocessing.custom_multiprocess as cm

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--method', type=str, default='fedavg', metavar='N',
                        help='federated learning method')

    # parser.add_argument('--model', type=str, default='resnet56', metavar='N',
    #                     help='neural network used in training')

    parser.add_argument('--data_dir', type=str, default='data/cifar100',
                        help='data directory: data/cifar100, data/cifar10, or a personal dataset')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--imbalanced_sampler', action = 'store_true',
                        help='use imlalanced sampler for training')                    

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=20, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we shoud use')

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='test pretrained model')

    parser.add_argument('--mu', type=float, default=0.001, metavar='MU',
                        help='mu (default: 0.001)')
    parser.add_argument('--width', type=float, default=0.7, metavar='WI',
                        help='minimum width for mutual training')
    parser.add_argument('--mult', type=float, default=1.0, metavar='MT',
                        help='multiplier for mutual training')
    parser.add_argument('--num_subnets', type=int, default=3,
                        help='how many subnets for mutual training')
    parser.add_argument('--localbn', action='store_true', default=False,
                        help='Keep local BNs each round')
    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')
    parser.add_argument('--thread_number', type=int, default=16, metavar='NN',
                        help='number of threads in a distributed cluster')
    parser.add_argument('--client_sample', type=float, default=1.0, metavar='MT',
                        help='Fraction of clients to sample')
    parser.add_argument('--resolution_type', type=int, default=0,
                        help='Specifies the resolution list used')
    parser.add_argument('--stoch_depth', default=0.5, type=float,
                    help='stochastic depth probability')
    parser.add_argument('--beta', default=0.0, type=float,
                    help='hyperparameter beta for mixup')
    args = parser.parse_args()

    return args

# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Helper Functions
def init_process(q, Client):
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])

def run_clients(received_info):
    try:
        # client, received_info = client_args
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info('exiting')
        return None

def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        if args.client_sample<1.0:
            num_clients = int(args.client_number*args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number==0 and num_clients>0:
            clients_per_thread = int(num_clients/args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t+clients_per_thread)]
                mapping_dict[c].append(idxs)
        else:
            logging.info("############ WARNING: Sampled client number not divisible by number of threads ##############")
            break
    return mapping_dict

if __name__ == "__main__":
    try:
     set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed()
    # get arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
 
    # get data
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict,\
         class_num = dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size, args.imbalanced_sampler)

    mapping_dict = allocate_clients_to_threads(args)
    #init method and model type
    if args.method=='fedavg':
        Server = fedavg.Server
        Client = fedavg.Client
        Model = resnet56 if 'cifar' in args.data_dir else resnet18
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method=='gradaug':
        Server = gradaug.Server
        Client = gradaug.Client
        Model = resnet56_gradaug if 'cifar' in args.data_dir else resnet18_gradaug
        width_range = [args.width, 1.0]
        resolution_dict = {'cifar': [[32, 28, 24, 20], [32, 28, 24], [32, 24, 16], [32, 28, 24, 20, 16], [32]], 'imagenet': [[224, 192, 160, 128], [224, 192, 160]]}
        resolutions = resolution_dict['cifar'][args.resolution_type] if 'cifar' in args.data_dir else resolution_dict['imagenet'][args.resolution_type]
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 
                            'width_range': width_range, 'resolutions': resolutions} for i in range(args.thread_number)]
    elif args.method=='fedprox':
        Server = fedprox.Server
        Client = fedprox.Client
        Model = resnet56 if 'cifar' in args.data_dir else resnet18
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method=='moon':
        Server = moon.Server
        Client = moon.Client
        Model = resnet56 if 'cifar' in args.data_dir else resnet18
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method=='feddepth':
        Server = feddepth.Server
        Client = feddepth.Client
        Model = resnet56_feddepth if 'cifar' in args.data_dir else resnet18_feddepth
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method=='mixup':
        Server = mixup.Server
        Client = mixup.Client
        Model = resnet56 if 'cifar' in args.data_dir else resnet18
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method=='ours':
        Server = ours.Server
        Client = ours.Client
        Model = resnet56_ours if 'cifar' in args.data_dir else resnet18_ours
        width_range = [args.width, 1.0]
        resolution_dict = {'cifar': [[32], [32, 28], [32, 28, 24], [32, 28, 24, 20]], 'imagenet': [[224]]}
        resolutions = resolution_dict['cifar'][args.resolution_type] if 'cifar' in args.data_dir else resolution_dict['imagenet'][args.resolution_type]
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 
                            'width_range': width_range, 'resolutions': resolutions} for i in range(args.thread_number)]

    else:
        raise ValueError('Invalid --method chosen! Please choose from availible methods.')

    if args.pretrained:
        print('Pretrained')
        server_dict['save_path'] = "best.pt"
        server = Server(server_dict, args)
        server.model.load_state_dict(torch.load('logs/2022-03-21 20:56:10.626330_fedavg_e20_c16/server.pt'))
        acc = server.test()
    else:
        #init nodes
        client_info = Queue()
        # clients = {}
        for i in range(args.thread_number):
            client_info.put((client_dict[i], args))
            # clients[i] = Client(client_dict[i], args)

        # Start server and get initial outputs
        pool = cm.MyPool(args.thread_number, init_process, (client_info, Client))
        # init server
        server_dict['save_path'] = './logs/{}_{}_e{}_c{}'.format(
                        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()), args.method, args.epochs, args.client_number)
        if not os.path.exists(server_dict['save_path']):
            os.makedirs(server_dict['save_path'])
        server = Server(server_dict, args)
        server_outputs = server.start()
        # Start Federated Training
        time.sleep(150*(args.client_number/16)) #  Allow time for threads to start up
        for r in range(args.comm_round):
            logging.info('************** Round: {} ***************'.format(r))
            round_start = time.time()
            client_outputs = pool.map(run_clients, server_outputs)
            client_outputs = [c for sublist in client_outputs for c in sublist]
            server_outputs = server.run(client_outputs)
            round_end = time.time()
            logging.info('Round {} Time: {}s'.format(r, round_end-round_start))
        pool.close()
        pool.join()
