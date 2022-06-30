import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from random_walk_dataset import RandomWalkEllipseDataset
from torch.utils.data import DataLoader


class MultiChannelEllipticalNN(pl.LightningModule):
    """
    CNN that takes two-channel images as input and produces 2D regression output.
    Input will be location/uncertainty across locations as the two image channels, and output the estimated length
    and width of the object in scaled coordinates (assuming that you use the corresponding data for training of course!)
    """
    def __init__(self, lr=1e-3, use_cuda=False, lr_step_size=None, lr_step=None, weight_decay=0.0):
        """
        Create a new CNN instance
        :param lr: Learning Rate used for training
        :param use_cuda: Bool: Whether to use CUDA (i.e. use GPU)
        :param lr_step_size: Int: After this many epochs, the learning rate will be scaled down. Set to None to disable.
        Can also be "cosine" (str), in which case a cosine learning rate schedule will be used instead.
        :param lr_step: After lr_step_size epochs, learning rate will be scaled down by this factor. If None, will be
        set to 1e-2.
        :param weight_decay: Weight decay, i.e. L2 normalization of weights. Defaults to 0, which means no weight decay.
        Set to small (e.g. 1e-1 to 1e-4) positive value to use.
        """
        super().__init__()

        # params
        self.lr_step_size = lr_step_size
        self.lr_step = lr_step if lr_step is not None else 1e-2
        self.use_cuda = use_cuda
        self.lr = lr
        self.weight_decay = weight_decay
        if self.use_cuda:
            self.device_to_use = torch.device('cuda')
        else:
            self.device_to_use = torch.device('cpu')

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(out_features=2)
        )
        self.to(self.device_to_use)

    def move_to_cuda(self):
        """
        Set device to use to cuda and move self to this device.
        """
        self.device_to_use = torch.device('cuda')
        self.to(self.device_to_use)

    def forward(self, x):
        """
        Forward pass through the network
        :param x: Input
        :return: Regression output
        """
        if len(x.shape) == 2:
            x = torch.tensor(x).reshape(1, 1, *x.shape)
        elif len(x.shape) == 3:
            x = torch.tensor(x).reshape(1, *x.shape)
        result = self.layers(x.float().to(self.device_to_use))
        return result

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.
        """
        x, y = batch
        y = torch.tensor(y).float().to(self.device_to_use)
        y_hat = self.layers(x.float())
        # TODO ensure y_hat and y are both torch.Size([2]) to prevent automatic broadcasting causing a warning
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.
        """
        x, y = batch
        y = torch.tensor(y).float().to(self.device_to_use)
        y_hat = self.layers(x.float())
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        """
        On validation epoch end, all outputs from validation epoch steps will be averaged and printed to console if this
        function is registered to be called in that case.
        """
        avg_loss = torch.stack(outputs).mean().cpu().detach().numpy()
        print("\nValidation epoch end loss: {:.5f}\n".format(avg_loss))

    def configure_optimizers(self):
        """
        Set up optimizers and schedulers according to parameters passed in __init__
        Using ADAM + optional Step Schedule for learning rate
        Can also use cosine schedule if self.lr_step_size = "cosine"
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_step_size is not None and not type(self.lr_step_size) == str:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_step)
            return [optimizer], [scheduler]
        elif self.lr_step_size is None:
            return optimizer
        elif "cosine" in self.lr_step_size:
            n_epochs = int(self.lr_step_size.split(" ")[-1])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer


def train_on_random_walk_based_dataset(n_scenes_train=250,
                                       n_scenes_val=25,
                                       n_steps_per_scene=20,
                                       skip_first_n_steps=0,
                                       noise_scaling_factor=1.,
                                       uniformly_vary_noise=True,
                                       verbose=True,
                                       measurement_lambda_min_max=None,
                                       n_epochs=40,
                                       base_lr=1e-3,
                                       lr_step_size=10,
                                       lr_step_gamma=1e-1,
                                       min_size=1.,
                                       max_size=10.,
                                       ds_gamma=0.95,
                                       uniformly_vary_gamma=True,
                                       weight_decay=None,
                                       max_size_generation_factor=1.0
                                       ):
    gpus = 1  # use gpu

    measurement_lambda_min_max = measurement_lambda_min_max if measurement_lambda_min_max is not None else [6, 15]

    # Create the data sets for training and validation

    # TRAINING SET
    if verbose:
        print("Creating training data set ...")
    train = RandomWalkEllipseDataset(n_scenes=n_scenes_train,
                                     n_steps_per_scene=n_steps_per_scene,
                                     skip_first_n_steps=skip_first_n_steps,
                                     image_size=300,
                                     noise_scaling_factor=noise_scaling_factor,
                                     uniformly_vary_noise=uniformly_vary_noise,
                                     measurement_lambda_min_max=measurement_lambda_min_max,
                                     min_size=min_size,
                                     max_size=max_size,
                                     gamma=ds_gamma,
                                     uniformly_vary_gamma=uniformly_vary_gamma,
                                     gaussian_blur=True,
                                     transform=None,
                                     max_size_generation_factor=max_size_generation_factor
                                     )
    if verbose:
        print("Training Data Set created, #={}".format(len(train)))

    # VALIDATION SET:
    if verbose:
        print("Creating validation data set ...")
    val = RandomWalkEllipseDataset(n_scenes=n_scenes_val,
                                   n_steps_per_scene=n_steps_per_scene,
                                   skip_first_n_steps=skip_first_n_steps,
                                   image_size=300,
                                   noise_scaling_factor=noise_scaling_factor,
                                   uniformly_vary_noise=uniformly_vary_noise,
                                   measurement_lambda_min_max=measurement_lambda_min_max,
                                   min_size=min_size,
                                   max_size=max_size,
                                   gamma=ds_gamma,
                                   uniformly_vary_gamma=uniformly_vary_gamma,
                                   gaussian_blur=True,
                                   transform=None,
                                   max_size_generation_factor=max_size_generation_factor
                                   )
    if verbose:
        print("Validation Data Set created, #={}".format(len(val)))

    if verbose:
        print("Successfully created training and validation data sets")

    # Create a new network
    network = MultiChannelEllipticalNN(use_cuda=(gpus != 0), lr=base_lr, lr_step_size=lr_step_size,
                                       lr_step=lr_step_gamma, weight_decay=weight_decay)

    # Create the pytorch lightning trainer for the network
    trainer = pl.Trainer(max_epochs=n_epochs, gpus=gpus)

    # Fit the neural network based on training and validation data
    if verbose:
        print("Launching training of NN")
    trainer.fit(network, DataLoader(train), DataLoader(val))
    print("Finished training network")


if __name__ == '__main__':
    train_on_random_walk_based_dataset(n_scenes_train=6000,
                                       n_scenes_val=300,
                                       n_steps_per_scene=5,
                                       skip_first_n_steps=3,
                                       noise_scaling_factor=10,
                                       uniformly_vary_noise=True,
                                       verbose=True,
                                       measurement_lambda_min_max=[5, 300],
                                       n_epochs=40,
                                       base_lr=1e-3,
                                       lr_step_size=10,
                                       lr_step_gamma=1e-1,
                                       min_size=0.5,
                                       max_size=10,
                                       weight_decay=1e-3,
                                       max_size_generation_factor=1.33
                                       )
