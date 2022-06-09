import torch
import wandb
import logging
from data_preprocessing.sam import SAM
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
from utils import *

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(img_size=224, num_classes=self.num_classes, type=args.poster_type).to(self.device)

        self.CE_criterion = torch.nn.CrossEntropyLoss()
        self.lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

        params = self.model.parameters()
        if args.poster_optimizer == 'adamw':
            # base_optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
            base_optimizer = torch.optim.AdamW
        elif args.poster_optimizer == 'adam':
            # base_optimizer = torch.optim.Adam(params, args.lr, weight_decay=1e-4)
            base_optimizer = torch.optim.Adam
        elif args.poster_optimizer == 'sgd':
            # base_optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-4)
            base_optimizer = torch.optim.SGD
        else:
            raise ValueError("Optimizer not supported.")
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False,)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer.base_optimizer, gamma=0.98)
    def train(self):
        # train the local model
        for i in range(1, self.args.epochs + 1):
            train_loss = 0.0
            correct_sum = 0
            iter_cnt = 0
            self.model.train()
            epoch_loss = []
            for batch_i, (imgs, targets) in enumerate(self.train_dataloader):
                batch_loss = []
                iter_cnt += 1
                self.optimizer.zero_grad()
                imgs = imgs.cuda()
                outputs, features = self.model(imgs)
                targets = targets.cuda()

                CE_loss = self.CE_criterion(outputs, targets)
                lsce_loss = self.lsce_criterion(outputs, targets)
                loss = 2 * lsce_loss + CE_loss
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                outputs, features = self.model(imgs)
                CE_loss = self.CE_criterion(outputs, targets)
                lsce_loss = self.lsce_criterion(outputs, targets)

                loss = 2 * lsce_loss + CE_loss
                loss.backward() # make sure to do a full forward pass
                self.optimizer.second_step(zero_grad=True)

                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            i, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))

            self.scheduler.step()
        weights = self.model.cpu().state_dict()
        return weights

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        pre_labels = []
        gt_labels = []
        test_size = len(self.test_dataloader)
        with torch.no_grad():
            bingo_cnt = 0

            for batch_i, (imgs, targets) in enumerate(self.test_dataloader):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                outputs, features = self.model(imgs)
                _, predicts = torch.max(outputs, 1)
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()


            acc = bingo_cnt.float() / float(test_size)
            acc = np.around(acc.numpy(), 4)
            logging.info("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
        return acc

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(img_size=224, num_classes=self.num_classes, type=args.poster_type).to(self.device)
    def test(self):
        self.model.to(self.device)
        self.model.eval()

        pre_labels = []
        gt_labels = []
        test_size = len(self.test_data)
        with torch.no_grad():
            bingo_cnt = 0
            for batch_i, (imgs, targets) in enumerate(self.test_data):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                outputs, features = self.model(imgs)
                
                _, predicts = torch.max(outputs, 1)
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()


            acc = bingo_cnt.float() / float(test_size)
            acc = np.around(acc.numpy(), 4)
            logging.info("************* Server Acc = {:.2f} **************".format(acc))
        return acc
