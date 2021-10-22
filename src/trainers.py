from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

import settings
import training_utils
import visualization
import transforms
from datasets import InstanceSegmentationDataset
import pytorch_models


class InstanceSegmentationTrainer:

    def __init__(self, model, model_parameters, training_parameters, transform_parameters):

        self.model = model
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.transform_parameters = transform_parameters

    def train_fn(self, train_loader, model, optimizer, device, scheduler=None):

        """
        Train given model on given data loader

        Parameters
        ----------
        train_loader (torch.utils.data.DataLoader): Training set data loader
        model (torch.nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Location of the model and inputs
        scheduler (torch.optim.LRScheduler or None): Learning rate scheduler

        Returns
        -------
        train_loss (float): Average training loss after model is fully trained on training set data loader
        """

        print('\n')
        model.train()
        progress_bar = tqdm(train_loader)
        classifier_losses = []
        box_reg_losses = []
        mask_losses = []
        objectness_losses = []
        rpn_box_reg_losses = []
        total_losses = []

        for images, targets in progress_bar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            classifier_losses.append(loss_dict['loss_classifier'].item())
            box_reg_losses.append(loss_dict['loss_box_reg'].item())
            mask_losses.append(loss_dict['loss_mask'].item())
            objectness_losses.append(loss_dict['loss_objectness'].item())
            rpn_box_reg_losses.append(loss_dict['loss_rpn_box_reg'].item())
            total_loss_value = total_loss.item()
            total_losses.append(total_loss_value)

            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            average_classifier_loss = np.mean(classifier_losses)
            average_box_reg_loss = np.mean(box_reg_losses)
            average_mask_loss = np.mean(mask_losses)
            average_objectness_loss = np.mean(objectness_losses)
            average_rpn_box_reg_loss = np.mean(rpn_box_reg_losses)
            average_total_loss = np.mean(total_losses)
            progress_bar.set_description(f'train_total_loss: {average_total_loss:.4f} (train_classifier_loss: {average_classifier_loss:.4f} - train_box_reg_loss: {average_box_reg_loss:.4f} - train_mask_loss: {average_mask_loss:.4f} - train_objectness_loss: {average_objectness_loss:.4f} - train_rpn_box_reg_loss: {average_rpn_box_reg_loss:.4f})')

        train_loss = np.mean(total_losses)
        return train_loss

    def val_fn(self, val_loader, model, device):

        """
        Validate given model on given data loader

        Parameters
        ----------
        val_loader (torch.utils.data.DataLoader): Validation set data loader
        model (torch.nn.Module): Model to validate
        device (torch.device): Location of the model and inputs

        Returns
        -------
        val_loss (float): Average validation loss after model is fully validated on validation set data loader
        """

        progress_bar = tqdm(val_loader)
        classifier_losses = []
        box_reg_losses = []
        mask_losses = []
        objectness_losses = []
        rpn_box_reg_losses = []
        total_losses = []

        with torch.no_grad():
            for images, targets in progress_bar:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                classifier_losses.append(loss_dict['loss_classifier'].item())
                box_reg_losses.append(loss_dict['loss_box_reg'].item())
                mask_losses.append(loss_dict['loss_mask'].item())
                objectness_losses.append(loss_dict['loss_objectness'].item())
                rpn_box_reg_losses.append(loss_dict['loss_rpn_box_reg'].item())
                total_losses.append(total_loss.item())

                average_classifier_loss = np.mean(classifier_losses)
                average_box_reg_loss = np.mean(box_reg_losses)
                average_mask_loss = np.mean(mask_losses)
                average_objectness_loss = np.mean(objectness_losses)
                average_rpn_box_reg_loss = np.mean(rpn_box_reg_losses)
                average_total_loss = np.mean(total_losses)
                progress_bar.set_description(f'val_total_loss: {average_total_loss:.4f} (val_classifier_loss: {average_classifier_loss:.4f} - val_box_reg_loss: {average_box_reg_loss:.4f} - val_mask_loss: {average_mask_loss:.4f} - val_objectness_loss: {average_objectness_loss:.4f} - val_rpn_box_reg_loss: {average_rpn_box_reg_loss:.4f})')

        val_loss = np.mean(total_losses)
        return val_loss

    def train_and_validate(self, df):

        print(f'\n{"-" * 30}\nRunning {self.model} for Training\n{"-" * 30}\n')
        instance_segmentation_transforms = transforms.get_transforms(**self.transform_parameters)

        for fold in sorted(df['fold'].unique()):

            print(f'\nFold {fold}\n{"-" * 6}')

            trn_idx, val_idx = df.loc[df['fold'] != fold].index, df.loc[df['fold'] == fold].index
            train_dataset = InstanceSegmentationDataset(
                images=df.loc[trn_idx, 'id'].values,
                masks=df.loc[trn_idx, 'annotation'].values,
                transforms=instance_segmentation_transforms['train']
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_parameters['data_loader']['training_batch_size'],
                sampler=RandomSampler(train_dataset),
                pin_memory=True,
                drop_last=False,
                collate_fn=training_utils.collate_fn,
                num_workers=self.training_parameters['data_loader']['num_workers'],
            )
            val_dataset = InstanceSegmentationDataset(
                images=df.loc[val_idx, 'id'].values,
                masks=df.loc[val_idx, 'annotation'].values,
                transforms=instance_segmentation_transforms['test']
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_parameters['data_loader']['test_batch_size'],
                sampler=SequentialSampler(val_dataset),
                pin_memory=True,
                drop_last=False,
                collate_fn=training_utils.collate_fn,
                num_workers=self.training_parameters['data_loader']['num_workers'],
            )

            training_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device(self.training_parameters['device'])
            model = getattr(pytorch_models, self.model)(**self.model_parameters)
            model = model.to(device)
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

                if self.training_parameters['lr_scheduler'] == 'ReduceLROnPlateau':
                    train_loss = self.train_fn(train_loader, model, optimizer, device, scheduler=None)
                    val_loss = self.val_fn(val_loader, model, device)
                    scheduler.step(val_loss)
                else:
                    train_loss = self.train_fn(train_loader, model, optimizer, device, scheduler)
                    val_loss = self.val_fn(val_loader, model, device)

                print(f'Epoch {epoch} - Training Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f}')

                best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
                if val_loss < best_val_loss:
                    model_path = f'{settings.MODELS_PATH}/{self.model}/{self.model}_fold{fold}.pt'
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
                        title=f'{self.model} - Fold {fold} Learning Curve',
                        path=f'{settings.MODELS_PATH}/{self.model}/{self.model}_fold{fold}_learning_curve.png'
                    )
                    early_stopping = True
