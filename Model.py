import torch
from torch import nn

# Linear layer has position information, suitable for fixed-size image

# if using convolutional:
# -> linear layer+convolutional layer or conlolutional+positional encoding

# cannot use batchnorm in the decoder

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck_size = 256  # this might be worth to play around with ...
        self.input_dim = 28*28
        self.h_dim = 400
        self.encoder = Encoder(self.input_dim, self.h_dim, bottleneck_size=self.bottleneck_size)
        self.decoder = Decoder(self.input_dim, self.h_dim, bottleneck_size=self.bottleneck_size)
        self.prior_distribution = torch.distributions.Normal(0, 1)

    def forward(self,
                target_data=None,  # during training this should be a batch of target data.
                # During inference simply leave this out to sample unconditionally.
                noise_scale_during_inference=0.2,  # might be worth to play around with this ...
                device="cpu"
                ):
        if target_data is not None:
            # run the encoder
            means, log_variance = self.encoder(target_data)

            # convert means and log_variance to sample so that z is differentiable
            z = means + log_variance * self.prior_distribution.sample(means.shape)

        else:
            z = torch.randn(self.bottleneck_size).to(device).unsqueeze(0) * noise_scale_during_inference

        # run the decoder
        reconstructions_of_targets = self.decoder(z)

        if target_data is not None:
            # calculate the losses
            predicted_distribution = torch.distributions.Normal(means, log_variance.exp())
            kl_loss = torch.distributions.kl_divergence(predicted_distribution, self.prior_distribution).mean()
            reconstruction_loss = nn.functional.mse_loss(reconstructions_of_targets, target_data, reduction="mean")
            return reconstructions_of_targets, kl_loss, reconstruction_loss

        return reconstructions_of_targets


class Encoder(nn.Module):
    def __init__(self, input_dim, h_dim, bottleneck_size):
        """
        The input to the encoder will have the shape (batch_size, 1, 28, 28)
        The output should be a batch of vectors of the bottleneck_size
        """
        super().__init__()
        self.img_to_hid = nn.Linear(input_dim, h_dim)
        self.hid_to_mu = nn.Linear(h_dim, bottleneck_size)
        self.hid_to_sigma = nn.Linear(h_dim, bottleneck_size)
        self.relu = nn.ReLU()

    def forward(self, target_data_for_compression):
        # q_phi(z|x)
        h = self.img_to_hid(target_data_for_compression)
        h = self.relu(h)
        mu, sigma = self.hid_to_mu(h), self.hid_to_sigma(h)
        
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, input_dim, h_dim, bottleneck_size):
        """
        The input of the decoder will be a batch of fixed size vectors with the bottleneck_size as dimensionality
        The output of the decoder should have the shape (batch_size, 1, 28, 28)
        """
        super().__init__()
        self.z_to_hid = nn.Linear(bottleneck_size, h_dim)
        self.hid_to_img = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, compressed_data_for_decompression):
        # p_theta(x|z)
        h = self.z_to_hid(compressed_data_for_decompression)
        h = self.relu(h)
        output = self.hid_to_img(h)
        output = torch.sigmoid(output)

        return output
