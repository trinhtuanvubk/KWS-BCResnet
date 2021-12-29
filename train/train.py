from torch.utils.tensorboard.writer import SummaryWriter
# I got wrong here 
import numpy as np
import torch
import tqdm
import os
import config
from layers.melspectrogram import MelSpectrogram
import utils
import models
from data_loader import data_loaders
from config.config import Config
from train.args import get_args

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class Trainer:

    def __init__(self, args,config):
        # save args
        self.args = args
        self.log_folder = utils.get_log_folder(args,config)
        self.checkpoint_path = utils.get_checkpoint_path(args,config)
        # clear cache
        self.clear_cache()
        # get loader
        self.train_loader, self.dev_loader, self.test_loader = data_loaders.GSC_loader(config)
        # get model
        self.model = utils.get_model(args,config)
        # get criterion
        self.criterion = utils.get_criterion(args)
        # get optimizer
        self.optimizer = utils.get_optimizer(self.model, args)
        # get scheduler
        self.scheduler = utils.get_scheduler(self.optimizer, args)
        # get writer
        self.writer = SummaryWriter(self.log_folder)
        # get iteration
        self.iteration = 0
    

    def train_step(self, batch, batch_idx):
        x, lens, y = batch
        x = x.to(self.args.device)
        lens = lens.to(self.args.device)
        y = y.to(self.args.device)
        # reset grad
        self.optimizer.zero_grad()
        # compute forward
        feats = MelSpectrogram(self.args).transform(x, lens)
        output = self.model(feats, y)
        # compute loss
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def validation_step(self, batch, batch_idx, stage=None):
        with torch.no_grad():
            x, lens, y = batch
            x = x.to(self.args.device)
            lens = lens.to(self.args.device)
            y = y.to(self.args.device)
            # inference
            feats = MelSpectrogram(self.args).transform(x, lens)
            output = self.model(feats, y)
            accuracy = torch.mean((torch.argmax(output, dim=1) == y).float())
        return accuracy

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

    def load_checkpoint(self):
        self.epoch = 0
        self.accuracy = 0.
        path = self.checkpoint_path
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            self.accuracy = checkpoint['accuracy']
            print(f'Best accuracy: {self.accuracy}')

    def write_dev_metric_to_tensorboard(self, epoch, accuracy, stage='train'):
        accuracy = float(accuracy)
        # display
        print(f'Evaluate epoch {epoch} - {stage}_accuracy: {accuracy}')
        with open(f'{os.path.join(self.log_folder, "train_log.txt")}', 'a') as fin:
            fin.write(f'Evaluate epoch {epoch} - {stage}_accuracy: {accuracy}\n')
        # write to tensorboard
        metrics = {f'{stage}_accuracy': accuracy}
        self.writer.add_scalars('accuracy', metrics, epoch)

    def write_train_metric_to_tensorboard(self, loss):
        metrics = {'loss': float(loss)}
        self.writer.add_scalars('loss', metrics, self.iteration)

    def limit_train_batch_hook(self, batch_idx):
        if self.args.limit_train_batch > 0:
            if batch_idx > self.args.limit_train_batch:
                return True
        return False

    def limit_val_batch_hook(self, batch_idx):
        if self.args.limit_val_batch > 0:
            if batch_idx > self.args.limit_val_batch:
                return True
        return False

    def clear_cache(self):
        if self.args.clear_cache:
            os.system(f'rm -rf {self.checkpoint_path} {self.log_folder}')

    def _fit(self):
        # load checkpoint
        self.load_checkpoint()

        # training
        for epoch in range(self.epoch, self.args.num_epoch):
            ##########################################################################################
            # evalute on test
            self.model.eval()
            with tqdm.tqdm(self.test_loader, unit='it') as pbar:
                pbar.set_description(f'Evaluate epoch {epoch}')
                test_accuracy = []
                for batch_idx, batch in enumerate(pbar):
                    # validate
                    accuracy = self.validation_step(batch, batch_idx, stage='test')
                    test_accuracy.append(float(accuracy))
                    pbar.set_postfix(accuracy=float(accuracy))

                    # limit train batch hook
                    if self.limit_val_batch_hook(batch_idx):
                        break

                # print epoch summary
                self.write_dev_metric_to_tensorboard(epoch, np.mean(test_accuracy), stage='test')

            # evalute on dev + augment
            if self.args.no_evaluate:
                self.model.eval()
                with tqdm.tqdm(self.dev_loader, unit='it') as pbar:
                    pbar.set_description(f'Evaluate epoch {epoch}')
                    val_accuracy = []
                    for batch_idx, batch in enumerate(pbar):
                        # validate
                        accuracy = self.validation_step(batch, batch_idx)
                        val_accuracy.append(float(accuracy))
                        pbar.set_postfix(accuracy=float(accuracy))

                        # limit train batch hook
                        if self.limit_val_batch_hook(batch_idx):
                            break

                # print epoch summary
                self.write_dev_metric_to_tensorboard(epoch, np.mean(val_accuracy), stage='val')

            # evaluate on train + augment
            if self.args.no_evaluate:
                self.model.eval()
                with tqdm.tqdm(self.train_loader, unit='it') as pbar:
                    pbar.set_description(f'Evaluate epoch {epoch}')
                    train_accuracy = []
                    for batch_idx, batch in enumerate(pbar):
                        # validate
                        accuracy = self.validation_step(batch, batch_idx)
                        train_accuracy.append(float(accuracy))
                        pbar.set_postfix(accuracy=float(accuracy))

                        # limit train batch hook
                        if self.limit_val_batch_hook(batch_idx):
                            break

                # print epoch summary
                self.write_dev_metric_to_tensorboard(epoch, np.mean(train_accuracy), stage='train')

            ##########################################################################################
            self.model.train()
            with tqdm.tqdm(self.train_loader, unit='it') as pbar:
                pbar.set_description(f'Epoch {epoch}')
                for batch_idx, batch in enumerate(pbar):

                    # perform training step
                    loss = self.train_step(batch, batch_idx)
                    pbar.set_postfix(loss=float(loss))

                    # log
                    self.epoch = epoch
                    self.iteration += 1
                    if self.iteration % self.args.log_iter == 0:
                        self.write_train_metric_to_tensorboard(loss)

                    # limit train batch hook
                    if self.limit_train_batch_hook(batch_idx):
                        break
            
            # save checkpoint
            if self.accuracy < np.mean(test_accuracy):
                self.accuracy = np.mean(test_accuracy)
                self.save_checkpoint()
            
            # update lr via scheduler plateau
            if self.scheduler != None:
                self.scheduler.step(self.accuracy)

    def fit(self):
        models.print_summary(self.model)
        try:
            self._fit()
        except KeyboardInterrupt: 
            pass

    def save_checkpoint(self):
        # save checkpoint
        torch.save({
            'accuracy': self.accuracy,
            'iteration': self.iteration,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            }, self.checkpoint_path)
        print('[+] checkpoint saved')

def train(args,config):
    trainer = Trainer(args,config)
    trainer.fit()


if __name__=='__main__':

    args = get_args()
    config = Config()
    train(args=args,config=config)
