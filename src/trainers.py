from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
import monai

import settings
import training_utils
import transforms
import visualization
from datasets import SegmentationDataset
from pytorch_models.monai_segmentation_models import SegResNetModel, SegResNetVAEModel


class SegmentationTrainer:

    def __init__(self, model_name, model_path, model_parameters, training_parameters):

        self.model_name = model_name
        self.model_path = model_path
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters

    def get_model(self):

        model = None

        if self.model_name == 'segresnet':
            model = SegResNetModel(**self.model_parameters)
        elif self.model_name == 'segresnetvae':
            model = SegResNetVAEModel(**self.model_parameters)

        return model

    def train_fn(self, train_loader, model, criterion, optimizer, scheduler, device):

        print('\n')
        model.train()
        progress_bar = tqdm(train_loader)
        losses = []

        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            average_loss = np.mean(losses)
            progress_bar.set_description(f'train_loss: {average_loss:.6f} - lr: {scheduler.get_last_lr()[0]:.8f}')

        train_loss = np.mean(losses)
        return train_loss

    def val_fn(self, val_loader, model, criterion, device):

        model.eval()
        progress_bar = tqdm(val_loader)
        losses = []

        with torch.no_grad():
            for images, masks in progress_bar:
                images, masks = images.to(device), masks.to(device)
                output = model(images)
                loss = criterion(output, masks.detach())

                losses.append(loss.item())
                average_loss = np.mean(losses)
                progress_bar.set_description(f'val_loss: {average_loss:.6f}')

        val_loss = np.mean(losses)
        return val_loss

    def train_and_validate(self, df_train):

        print(f'\n{"-" * 30}\nRunning {self.model_name} for Training\n{"-" * 30}\n')

        for fold in sorted(df_train['fold'].unique()):

            print(f'\nFold {fold}\n{"-" * 6}')

            trn_idx, val_idx = df_train.loc[df_train['fold'] != fold].index, df_train.loc[df_train['fold'] == fold].index
            train_dataset = SegmentationDataset(
                images=df_train.loc[trn_idx, 'id'].reset_index(drop=True),
                masks='npy',
                transforms=None
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_parameters['data_loader']['batch_size'],
                sampler=RandomSampler(train_dataset),
                pin_memory=True,
                drop_last=False,
                num_workers=self.training_parameters['data_loader']['num_workers'],
            )
            val_dataset = SegmentationDataset(
                images=df_train.loc[val_idx, 'id'].reset_index(drop=True),
                masks='npy',
                transforms=None
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_parameters['data_loader']['batch_size'],
                sampler=SequentialSampler(val_dataset),
                pin_memory=True,
                drop_last=False,
                num_workers=self.training_parameters['data_loader']['num_workers'],
            )

            training_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model = self.get_model()
            model = model.to(device)

            criterion = getattr(monai.losses, self.training_parameters['loss'])(**self.training_parameters['loss_parameters'])
            optimizer = getattr(optim, self.training_parameters['optimizer'])(model.parameters(), **self.training_parameters['optimizer_parameters'])
            scheduler = getattr(optim.lr_scheduler, self.training_parameters['lr_scheduler'])(optimizer, **self.training_parameters['lr_scheduler_parameters'])

            early_stopping = False
            summary = {
                'train_loss': [],
                'val_loss': []
            }

            for epoch in range(1, self.training_parameters['epochs'] + 1):

                if early_stopping:
                    break

                train_loss = self.train_fn(train_loader, model, criterion, optimizer, scheduler, device)
                val_loss = self.val_fn(val_loader, model, criterion, device)
                print(f'Epoch {epoch} - Training Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f}')

                best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
                if val_loss < best_val_loss:
                    model_path = f'{settings.MODELS_PATH}/{self.model_name}/{self.model_name}_fold{fold}.pt'
                    torch.save(model.state_dict(), model_path)
                    print(f'Saving model to {model_path} (validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f})')

                summary['train_loss'].append(train_loss)
                summary['val_loss'].append(val_loss)

                best_iteration = np.argmin(summary['val_loss']) + 1
                if len(summary['val_loss']) - best_iteration >= self.training_parameters['early_stopping_patience']:
                    print(f'Early stopping (validation loss didn\'t increase for {self.training_parameters["early_stopping_patience"]} epochs/steps)')
                    print(f'Best validation loss is {np.min(summary["val_loss"]):.6f}')
                    visualization.visualize_learning_curve(
                        training_losses=summary['train_loss'],
                        validation_losses=summary['val_loss'],
                        title=f'{self.model_name} - Fold {fold} Learning Curve',
                        path=f'{settings.MODELS_PATH}/{self.model_name}/{self.model_name}_fold{fold}_learning_curve.png'
                    )
                    early_stopping = True
