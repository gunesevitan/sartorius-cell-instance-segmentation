from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from sklearn.metrics import accuracy_score

import settings
import training_utils
import inference_utils
import visualization
import transforms
import datasets
import pytorch_models


class InstanceSegmentationTrainer:

    def __init__(self, model, model_path, model_parameters, training_parameters, transform_parameters, post_processing_parameters):

        self.model = model
        self.model_path = model_path
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.transform_parameters = transform_parameters
        self.post_processing_parameters = post_processing_parameters

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

        """
        Train and validate on train_images in a cross-validation loop

        Parameters
        ----------
        df [pandas.DataFrame of shape (606, 3)]: Dataframe of filenames, annotations and labels
        """

        print(f'\n{"-" * 30}\nRunning {self.model} for Training\n{"-" * 30}\n')
        instance_segmentation_transforms = transforms.get_instance_segmentation_transforms(**self.transform_parameters)

        for fold in sorted(df['fold'].unique()):

            print(f'\nFold {fold}\n{"-" * 6}')

            trn_idx, val_idx = df.loc[df['fold'] != fold].index, df.loc[df['fold'] == fold].index
            train_dataset = datasets.InstanceSegmentationDataset(
                images=df.loc[trn_idx, 'id'].values,
                masks=df.loc[trn_idx, 'annotation'].values,
                labels=df.loc[trn_idx, 'label'].values,
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
            val_dataset = datasets.InstanceSegmentationDataset(
                images=df.loc[val_idx, 'id'].values,
                masks=df.loc[val_idx, 'annotation'].values,
                labels=df.loc[val_idx, 'label'].values,
                transforms=instance_segmentation_transforms['val']
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
            # Using in-domain pretrained weights for backbone
            # Backbone is trained for classification task and reached perfect accuracy
            if self.model_parameters['in_domain_pretrained_backbone']:
                pretrained_fpn_state_dict = torch.load(f'{settings.MODELS_PATH}/resnet50/resnet50_perfect_classifier.pt')
                # Match key names and filter out head weights and biases
                pretrained_fpn_state_dict = {
                    f'fpn.backbone.body.{k.replace("backbone.", "")}': v
                    for k, v in pretrained_fpn_state_dict.items()
                    if ('num_batches_tracked' not in k) and ('head' not in k)
                }
                model_state_dict = model.state_dict()
                model_state_dict.update(pretrained_fpn_state_dict)
                model.load_state_dict(model_state_dict)

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
                    model_path = f'{settings.MODELS_PATH}/{self.model_path}/{self.model_path}_fold{fold}.pt'
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
                        path=f'{settings.MODELS_PATH}/{self.model_path}/{self.model_path}_fold{fold}_learning_curve.png'
                    )
                    early_stopping = True

    def inference(self, df):

        """
        Evaluate on train_images in a cross-validation loop

        Parameters
        ----------
        df [pandas.DataFrame of shape (606, 3)]: Dataframe of filenames, annotations and labels
        """

        print(f'\n{"-" * 30}\nRunning {self.model} for Inference\n{"-" * 30}\n')
        instance_segmentation_transforms = transforms.get_instance_segmentation_transforms(**self.transform_parameters)
        df[f'{self.model_path}_predictions_average_precision'] = 0

        for fold in sorted(df['fold'].unique()):

            print(f'\nFold {fold}\n{"-" * 6}')

            _, val_idx = df.loc[df['fold'] != fold].index, df.loc[df['fold'] == fold].index
            val_dataset = datasets.InstanceSegmentationDataset(
                images=df.loc[val_idx, 'id'].values,
                masks=df.loc[val_idx, 'annotation'].values,
                labels=df.loc[val_idx, 'label'].values,
                transforms=instance_segmentation_transforms['val']
            )

            training_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model = getattr(pytorch_models, self.model)(**self.model_parameters)
            model = model.to(device)
            model_path = f'{settings.MODELS_PATH}/{self.model_path}/{self.model_path}_fold{fold}.pt'
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

            for idx in tqdm(range(len(val_dataset))):

                if self.post_processing_parameters['verbose']:
                    print(f'Evaluating {val_dataset.images[idx]}\n{"-" * 15}')

                with torch.no_grad():
                    image = val_dataset[idx][0]
                    prediction = inference_utils.predict_single_image(
                        image=image,
                        model=model,
                        device=device,
                        nms_iou_thresholds=self.post_processing_parameters['nms_iou_thresholds'],
                        score_thresholds=self.post_processing_parameters['score_thresholds'],
                        verbose=self.post_processing_parameters['verbose']
                    )
                    ground_truth_masks = val_dataset[idx][1]['masks'].numpy()
                    prediction_masks = prediction['masks'].reshape(-1, image.shape[1], image.shape[2])
                    # Select label_threshold based on the most predicted label and convert probabilities to labels
                    most_predicted_label = settings.LABEL_MAPPING[prediction['most_predicted_label']]
                    prediction_masks = np.uint8(prediction_masks > self.post_processing_parameters['label_thresholds'][most_predicted_label])
                    average_precision = inference_utils.get_average_precision(
                        ground_truth_masks=ground_truth_masks,
                        prediction_masks=prediction_masks,
                        thresholds=self.post_processing_parameters['average_precision_thresholds'],
                        verbose=self.post_processing_parameters['verbose']
                    )
                    df.loc[val_idx[idx], f'{self.model_path}_predictions_average_precision'] = average_precision

            fold_score = np.mean(df.loc[val_idx, f'{self.model_path}_predictions_average_precision'])
            print(f'Fold {fold} - mAP: {fold_score:.6f}')
        oof_score = np.mean(df[f'{self.model_path}_predictions_average_precision'])
        print(f'{"-" * 30}\nOOF mAP: {oof_score:.6}\n{"-" * 30}')


class ClassificationTrainer:

    def __init__(self, model, model_path, model_parameters, training_parameters, transform_parameters):

        self.model = model
        self.model_path = model_path
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.transform_parameters = transform_parameters

    def train_fn(self, train_loader, model, criterion, optimizer, scheduler, device):

        """
        Train given model on given data loader

        Parameters
        ----------
        train_loader (torch.utils.data.DataLoader): Training set data loader
        model (torch.nn.Module): Model to train
        criterion (torch.nn.modules.loss): Loss function
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
        losses = []

        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
            average_loss = np.mean(losses)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            progress_bar.set_description(f'train_loss: {average_loss:.6f} - lr: {lr:.8f}')

        train_loss = np.mean(losses)
        return train_loss

    def val_fn(self, val_loader, model, criterion, device):

        """
        Validate given model on given data loader

        Parameters
        ----------
        val_loader (torch.utils.data.DataLoader): Validation set data loader
        model (torch.nn.Module): Model to validate
        criterion (torch.nn.modules.loss): Loss function
        device (torch.device): Location of the model and inputs

        Returns
        -------
        val_accuracy (float): Validation accuracy
        val_loss (float): Average validation loss after model is fully validated on validation set data loader
        """

        model.eval()
        progress_bar = tqdm(val_loader)
        losses = []
        labels = []
        predictions = []

        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                loss = criterion(output, targets)
                losses.append(loss.item())
                average_loss = np.mean(losses)
                progress_bar.set_description(f'val_loss: {average_loss:.6f}')
                labels += targets.detach().cpu().numpy().tolist()
                predictions += F.softmax(output.detach().cpu(), dim=1).numpy().tolist()

        predictions = np.array(predictions)
        labels = np.array(labels)
        predictions = np.argmax(predictions, axis=1)
        val_accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        val_loss = np.mean(losses)
        return val_accuracy, val_loss

    def train_and_validate(self, df):

        """
        Train on train_images and validate on train_semi_supervised_images with specified configurations

        Parameters
        ----------
        df [pandas.DataFrame of shape (2578, 3)]: Dataframe of filenames and targets
        """

        print(f'\n{"-" * 30}\nRunning {self.model} for Training\n{"-" * 30}\n')

        classification_transforms = transforms.get_classification_transforms(**self.transform_parameters)
        trn_idx, val_idx = df.loc[df['fold'] == 'train'].index, df.loc[df['fold'] == 'val'].index
        train_dataset = datasets.ClassificationDataset(
            images=df.loc[trn_idx, 'id'].values,
            targets=df.loc[trn_idx, 'cell_type_target'].values,
            image_directory='train_images',
            transforms=classification_transforms['train'],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_parameters['data_loader']['training_batch_size'],
            sampler=RandomSampler(train_dataset),
            pin_memory=True,
            drop_last=False,
            num_workers=self.training_parameters['data_loader']['num_workers']
        )
        val_dataset = datasets.ClassificationDataset(
            images=df.loc[val_idx, 'id'].values,
            targets=df.loc[val_idx, 'cell_type_target'].values,
            image_directory='train_semi_supervised_images',
            transforms=classification_transforms['val'],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_parameters['data_loader']['val_batch_size'],
            sampler=SequentialSampler(val_dataset),
            pin_memory=True,
            drop_last=False,
            num_workers=self.training_parameters['data_loader']['num_workers']
        )

        training_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
        device = torch.device(self.training_parameters['device'])
        criterion = nn.CrossEntropyLoss()
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
                train_loss = self.train_fn(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, scheduler=None, device=device)
                val_accuracy, val_loss = self.val_fn(val_loader=val_loader, model=model, criterion=criterion, device=device)
                scheduler.step(val_loss)
            else:
                train_loss = self.train_fn(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device)
                val_accuracy, val_loss = self.val_fn(val_loader=val_loader, model=model, criterion=criterion, device=device)

            print(f'Epoch {epoch} - Training Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f} - Validation Accuracy: {val_accuracy:6f}')
            best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
            if val_loss < best_val_loss:
                model_path = f'{settings.MODELS_PATH}/{self.model_path}/{self.model_path}.pt'
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
                    title=f'{self.model_path} - Learning Curve',
                    path=f'{settings.MODELS_PATH}/{self.model_path}/{self.model_path}_learning_curve.png'
                )
                early_stopping = True
