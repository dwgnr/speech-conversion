from ..melgan.mel2wav.modules import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations, product
import numpy as np
from scipy.cluster.vq import kmeans2


class VectorQuantizer(nn.Module):
    """
    This layer takes a tensor to be quantized. The channel dimension will be used as the space in which to quantize.
    All other dimensions will be flattened and will be seen as different examples to quantize.

    The output tensor will have the same shape as the input.
    As an example for a BCHW tensor of shape [16, 64, 32, 32], we will first convert it to an BHWC tensor
    of shape [16, 32, 32, 64] and then reshape it into [16384, 64] and all 16384 vectors of size 64 will be quantized
    independently. In other words, the channels are used as the space in which to quantize.
    All other dimensions will be flattened and be seen as different examples to quantize, 16384 in this case.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.kld_scale = 100.0
        self.register_buffer('data_initialized', torch.zeros(1))

        self.DEBUG = False

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        if self.DEBUG: print("VectorQuantizer: [BCHW] inputs=", inputs.shape)
        inputs = inputs.permute(0, 2, 1).contiguous()
        # 0 -> 0
        # 1 -> 2
        # 2 -> 1
        if self.DEBUG: print("VectorQuantizer: inputs after permute [BHWC] =", inputs.shape)

        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        if self.DEBUG: print("VectorQuantizer: flat_input [BxHxW, C]=", flat_input.shape)

        if self.training and self.data_initialized.item() == 0:
            print('Running K-Means') # data driven initialization for the embeddings
            rp = torch.randperm(flat_input.size(0))
            if self.DEBUG: print("KMeans: rp=", rp.shape)
            # We compute self._num_embeddings clusters
            kd = kmeans2(flat_input[rp[:20000]].data.cpu().numpy(), self._num_embeddings, minit='points')
            # kd[0] are the centroids; kd[1] are the labels
            if self.DEBUG: print("KMeans: kd=", kd[0])
            if self.DEBUG: print("KMeans: How many points are in each cluster? ", np.bincount(kd[1]))
            if self.DEBUG: print("KMeans: kd=", len(kd), kd[0].shape)

            self._embedding.weight.data.copy_(torch.from_numpy(kd[0]))
            if self.DEBUG: print("KMeans: embedding=", self._embedding.weight.data.shape)

            self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups


        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        if self.DEBUG: print("VectorQuantizer: selected encoding index at 0 =",
                             encoding_indices.shape[0], "shape of arr", encoding_indices.shape)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).cuda()
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        if self.DEBUG: print("VectorQuantizer: quantized shape, min, max=", quantized.shape, torch.min(quantized).data, torch.max(quantized).data)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        if self.DEBUG: print("VectorQuantizer: loss=", loss)
        loss *= self.kld_scale
        if self.DEBUG: print("VectorQuantizer: loss scaled=", loss)


        quantized = inputs + (quantized - inputs).detach()
        if self.DEBUG: print("VectorQuantizer: inputs + (quantized - inputs) shape, min, max=", quantized.shape,  torch.min(quantized).data, torch.max(quantized).data)

        avg_probs = torch.mean(encodings, dim=0)
        if self.DEBUG: print("VectorQuantizer: avg_probs:", avg_probs)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        if self.DEBUG: print("VectorQuantizer: perplexity:", perplexity)


        # convert quantized from BHWC -> BCHW
        # return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        quantized = quantized.permute(0, 2, 1).contiguous()
        if self.DEBUG: print("VectorQuantizer: quantized final =", quantized.shape)
        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).cuda()
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings


class VectorQuantizerSonnet(nn.Module):
    """
    Inspired from Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937,
    in https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and
    pytorch implementation of it from zalandoresearch in https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.
    Implements the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937
    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms
            (see equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizerSonnet, self).__init__()

        self.DEBUG = False

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

        self._commitment_cost = commitment_cost
        # self._device = device

    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        """
        Connects the module to some inputs.
        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.
        Returns:
            loss: Tensor containing the loss to optimize.
            quantize: Tensor containing the quantized version of the input.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            distances
        """

        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        _, time, batch_size = input_shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Compute distances between encoded audio frames and embedding vectors
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, dtype=torch.float).cuda()
        encodings.scatter_(1, encoding_indices, 1)

        # Compute distances between encoding vectors
        if not self.training and compute_distances_if_possible:
            _encoding_distances = [torch.dist(items[0], items[1], 2).cuda() for items in
                                   combinations(flat_input, r=2)]
            encoding_distances = torch.tensor(_encoding_distances).cuda().view(batch_size, -1)
        else:
            encoding_distances = None

        # Compute distances between embedding vectors
        if not self.training and compute_distances_if_possible:
            _embedding_distances = [torch.dist(items[0], items[1], 2).cuda() for items in
                                    combinations(self._embedding.weight, r=2)]
            embedding_distances = torch.tensor(_embedding_distances).cuda()
        else:
            embedding_distances = None

        # Sample nearest embedding
        if not self.training and compute_distances_if_possible:
            _frames_vs_embedding_distances = [torch.dist(items[0], items[1], 2).cuda() for items in
                                              product(flat_input, self._embedding.weight.detach())]
            frames_vs_embedding_distances = torch.tensor(_frames_vs_embedding_distances).cuda().view(
                batch_size, time, -1)
        else:
            frames_vs_embedding_distances = None

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        concatenated_quantized = self._embedding.weight[
            torch.argmin(distances, dim=1).detach().cpu()] if not self.training or record_codebook_stats else None

        # Losses
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        commitment_loss = self._commitment_cost * e_latent_loss
        vq_loss = q_latent_loss + commitment_loss

        quantized = inputs + (quantized - inputs).detach()  # Trick to prevent backpropagation of quantized
        avg_probs = torch.mean(encodings, dim=0)

        """
        The perplexity a useful value to track during training.
        It indicates how many codes are 'active' on average.
        """
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  # Exponential entropy

        # Convert quantized from BHWC -> BCHW
        return vq_loss, quantized.permute(0, 2, 1).contiguous(), \
               perplexity, encodings.view(batch_size, time, -1), \
               distances.view(batch_size, time, -1), encoding_indices, \
               {'e_latent_loss': e_latent_loss.item(), 'q_latent_loss': q_latent_loss.item(),
                'commitment_loss': commitment_loss.item(), 'vq_loss': vq_loss.item()}, \
               encoding_distances, embedding_distances, frames_vs_embedding_distances, concatenated_quantized

    @property
    def embedding(self):
        return self._embedding



class VQVAEQuantizeKarpathy(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937
    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.kld_scale = 10.0

        self.proj = nn.Conv2d(num_hiddens, embedding_dim, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, z):
        B, C, H, W = z.size()

        # project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
        z_e = self.proj(z)
        z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
        flatten = z_e.reshape(-1, self.embedding_dim)

        # DeepMind def does not do this but I find I have to... ;\
        if self.training and self.data_initialized.item() == 0:
            print('running kmeans!!') # data driven initialization for the embeddings
            rp = torch.randperm(flatten.size(0))
            kd = kmeans2(flatten[rp[:20000]].data.cpu().numpy(), self.n_embed, minit='points')
            self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, ind = (-dist).max(1)
        ind = ind.view(B, H, W)

        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind) # (B, H, W, C)
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        z_q = z_q.permute(0, 3, 1, 2) # stack encodings into channels again: (B, C, H, W)
        return z_q, diff, ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)



class Residual(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal):
        super(Residual, self).__init__()

        relu_1 = nn.ReLU(True)
        conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        if use_kaiming_normal:
            conv_1 = nn.utils.weight_norm(conv_1)
            nn.init.kaiming_normal_(conv_1.weight)

        relu_2 = nn.ReLU(True)
        conv_2 = nn.Conv1d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            bias=False
        )
        if use_kaiming_normal:
            conv_2 = nn.utils.weight_norm(conv_2)
            nn.init.kaiming_normal_(conv_2.weight)

        # All parameters same as specified in the paper
        self._block = nn.Sequential(
            relu_1,
            conv_1,
            relu_2,
            conv_2
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal):
        super(ResidualStack, self).__init__()

        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal)] * self._num_residual_layers)

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.leaky_relu(x, 0.2)
        return x


class Encoder(nn.Module):
    """
    This is the VQ-VAE Encoder as proposed in Neural Discrete Representation Learning
    with strided 1d convolutions
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, debug=False):
        super(Encoder, self).__init__()

        self.DEBUG = debug

        # 128
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        # 64
        self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=0)
        # 128
        self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens, use_kaiming_normal=False)

    def forward(self, inputs):

        if self.DEBUG: print("Encoder: inputs", inputs.shape)

        x = self._conv_1(inputs)
        x = F.relu(x)
        if self.DEBUG: print("Encoder: x_1", x.shape)

        x = self._conv_2(x)
        x = F.relu(x)
        if self.DEBUG: print("Encoder: x_2", x.shape)

        x = self._conv_3(x)
        if self.DEBUG: print("Encoder: x_3", x.shape)

        return self._residual_stack(x)


class Conv1DBuilder(object):

    @staticmethod
    def build(in_channels, out_channels, kernel_size, stride=1, padding=0, use_kaiming_normal=False):
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv


class ConvTranspose1DBuilder(object):

    @staticmethod
    def build(in_channels, out_channels, kernel_size, stride=1, padding=0, use_kaiming_normal=False):
        conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv


class ConvolutionalEncoder(nn.Module):

    def __init__(self, features_filters, num_hiddens, num_residual_layers,
                 num_residual_hiddens=512, use_kaiming_normal=False, debug=False):

        super(ConvolutionalEncoder, self).__init__()

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = Conv1DBuilder.build(
            in_channels=features_filters,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,  # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        2 convolutional layers with length 3 and
        residual connections.
        """

        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        4 feedforward ReLu layers with residual connections.
        """

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )

        self._features_filters = features_filters
        self._verbose = debug

    def forward(self, inputs):
        if self._verbose:
            print('inputs size: {}'.format(inputs.size()))

        x_conv_1 = F.relu(self._conv_1(inputs))
        if self._verbose:
            print('x_conv_1 output size: {}'.format(x_conv_1.size()))

        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        if self._verbose:
            print('_conv_2 output size: {}'.format(x.size()))

        x_conv_3 = F.relu(self._conv_3(x))
        if self._verbose:
            print('_conv_3 output size: {}'.format(x_conv_3.size()))

        x_conv_4 = F.relu(self._conv_4(x_conv_3)) + x_conv_3
        if self._verbose:
            print('_conv_4 output size: {}'.format(x_conv_4.size()))

        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4
        if self._verbose:
            print('x_conv_5 output size: {}'.format(x_conv_5.size()))

        x = self._residual_stack(x_conv_5) + x_conv_5
        if self._verbose:
            print('_residual_stack output size: {}'.format(x.size()))

        return x


class ConvolutionalEncoderRawAudio(nn.Module):

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal=False, debug=False):

        super(ConvolutionalEncoderRawAudio, self).__init__()

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = Conv1DBuilder.build(
            in_channels=1,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,  # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )


        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_31 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,  # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_32 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,  # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_33 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,  # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_34 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,  # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_35 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,  # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        2 convolutional layers with length 3 and
        residual connections.
        """

        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        4 feedforward ReLu layers with residual connections.
        """

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )

        # self._input_features_type = input_features_type
        # self._features_filters = features_filters
        # self._sampling_rate = sampling_rate
        # self._device = device
        self._debug = debug

    def forward(self, inputs):
        if self._debug:
            print('inputs size: {}'.format(inputs.size()))

        x_conv_1 = F.leaky_relu(self._conv_1(inputs))
        if self._debug:
            print('x_conv_1 output size: {}'.format(x_conv_1.size()))

        x = F.leaky_relu(self._conv_2(x_conv_1), 0.2) + x_conv_1
        if self._debug:
            print('_conv_2 output size: {}'.format(x.size()))

        x_conv_3 = F.leaky_relu(self._conv_3(x), 0.2)
        if self._debug:
            print('_conv_3 output size: {}'.format(x_conv_3.size()))

        x_conv_31 = F.leaky_relu(self._conv_31(x_conv_3), 0.2)
        if self._debug:
            print('_conv_31 output size: {}'.format(x_conv_31.size()))

        x_conv_32 = F.leaky_relu(self._conv_32(x_conv_31), 0.2)
        if self._debug:
            print('_conv_32 output size: {}'.format(x_conv_32.size()))

        x_conv_33 = F.leaky_relu(self._conv_33(x_conv_32), 0.2)
        if self._debug:
            print('_conv_33 output size: {}'.format(x_conv_33.size()))

        x_conv_34 = F.leaky_relu(self._conv_34(x_conv_33), 0.2)
        if self._debug:
            print('_conv_34 output size: {}'.format(x_conv_34.size()))

        x_conv_35 = F.leaky_relu(self._conv_35(x_conv_34), 0.2)
        if self._debug:
            print('_conv_35 output size: {}'.format(x_conv_35.size()))

        x_conv_4 = F.leaky_relu(self._conv_4(x_conv_35), 0.2) + x_conv_35
        # x_conv_4 = self._conv_4(x_conv_35) + x_conv_35
        if self._debug:
            print('_conv_4 output size: {}'.format(x_conv_4.size()))

        x_conv_5 = F.leaky_relu(self._conv_5(x_conv_4), 0.2) + x_conv_4
        # x_conv_5 = self._conv_5(x_conv_4) + x_conv_4
        if self._debug:
            print('x_conv_5 output size: {}'.format(x_conv_5.size()))

        x = self._residual_stack(x_conv_5) + x_conv_5
        if self._debug:
            print('_residual_stack output size: {}'.format(x.size()))

        return x


class DeconvolutionalDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers,
                 num_residual_hiddens, use_kaiming_normal, use_jitter, jitter_probability,
                 use_speaker_conditioning, verbose=False):

        super(DeconvolutionalDecoder, self).__init__()

        self._use_jitter = use_jitter
        self._use_speaker_conditioning = use_speaker_conditioning
        self._verbose = verbose

        if self._use_jitter:
            raise NotImplementedError()
            # self._jitter = Jitter(jitter_probability)

        in_channels = in_channels + 40 if self._use_speaker_conditioning else in_channels

        self._conv_1 = Conv1DBuilder.build(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal
        )

        self._upsample = nn.Upsample(scale_factor=2)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )

        self._conv_trans_1 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal
        )

        self._conv_trans_2 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal
        )

        self._conv_trans_3 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal
        )

    def forward(self, inputs):
        # speaker_dic, speaker_id -> required as input to forward: forward(self, inputs, speaker_dic, speaker_id)
        x = inputs
        if self._verbose:
            print('[FEATURES_DEC] input size: {}'.format(x.size()))

        if self._use_jitter and self.training:
            x = self._jitter(x)

        if self._use_speaker_conditioning:
            raise NotImplementedError()
            # speaker_embedding = GlobalConditioning.compute(speaker_dic, speaker_id, x,
            #                                                device=self._device, gin_channels=40, expand=True)
            # x = torch.cat([x, speaker_embedding], dim=1).to(self._device)

        x = self._conv_1(x)
        if self._verbose:
            print('[FEATURES_DEC] _conv_1 output size: {}'.format(x.size()))

        x = self._upsample(x)
        if self._verbose:
            print('[FEATURES_DEC] _upsample output size: {}'.format(x.size()))

        x = self._residual_stack(x)
        if self._verbose:
            print('[FEATURES_DEC] _residual_stack output size: {}'.format(x.size()))

        x = F.relu(self._conv_trans_1(x))
        if self._verbose:
            print('[FEATURES_DEC] _conv_trans_1 output size: {}'.format(x.size()))

        x = F.relu(self._conv_trans_2(x))
        if self._verbose:
            print('[FEATURES_DEC] _conv_trans_2 output size: {}'.format(x.size()))

        x = self._conv_trans_3(x)
        if self._verbose:
            print('[FEATURES_DEC] _conv_trans_3 output size: {}'.format(x.size()))

        return x




class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VQVAEGAN(nn.Module):
    """This is a VQ-VAE where the Decoder can be optionally replaced by a MelGAN Generator"""
    def __init__(self, num_hiddens=128,
                 num_embeddings=512, embedding_dim=64, commitment_cost=0.25,
                 decay=0, mel_channels=80, ngf=32, n_residual_layers=3, num_residual_hiddens=512,
                 ratios=None, use_sonnet_vq=True, use_sonnet_encoder=True,
                 use_kaiming_normal=True, train_on_raw_audio=False, use_melgan_decoder=True, debug=False):
        super(VQVAEGAN, self).__init__()

        self.DEBUG = debug
        self.use_sonnet_vq = use_sonnet_vq
        self.use_sonnet_encoder = use_sonnet_encoder

        # self.fft = Audio2Mel(n_mel_channels=mel_channels, sampling_rate=16000, mel_fmax=6000)
        if train_on_raw_audio:
            self._encoder = ConvolutionalEncoderRawAudio(num_hiddens, n_residual_layers, num_hiddens,
                                                         use_kaiming_normal=use_kaiming_normal, debug=self.DEBUG)
        else:
            if self.use_sonnet_encoder:
                self._encoder = ConvolutionalEncoder(mel_channels, num_hiddens, n_residual_layers,
                                                     num_residual_hiddens=num_residual_hiddens,
                                                     use_kaiming_normal=use_kaiming_normal,
                                                     debug=self.DEBUG)
            else:
                self._encoder = Encoder(mel_channels, num_hiddens,
                                        n_residual_layers,
                                        ngf, debug=self.DEBUG)

        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        if use_sonnet_vq:
            self._vq_vae = VectorQuantizerSonnet(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost
            )
        else:
            if decay > 0.0:
                # TODO this will not work unless you make it like VectorQuantizer
                self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                                  commitment_cost, decay)
            else:
                self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                               commitment_cost)

        if use_melgan_decoder:
            self._decoder = Generator(embedding_dim, ngf, n_residual_layers, ratios, debug=self.DEBUG)
        else:
            self._decoder = DeconvolutionalDecoder(
                in_channels=embedding_dim,
                out_channels=mel_channels,
                num_hiddens=num_hiddens,
                num_residual_layers=n_residual_layers,
                num_residual_hiddens=num_residual_hiddens,
                use_kaiming_normal=use_kaiming_normal,
                use_jitter=False,
                jitter_probability=0.0,
                use_speaker_conditioning=False,
                verbose=self.DEBUG
            )

    def forward(self, x):

        if self.DEBUG: print("1. forward: x before encoder", x.shape, torch.min(x).item(), torch.max(x).item())
        z = self._encoder(x)
        if self.DEBUG: print("2. forward: z after encoder", z.shape, torch.min(z).item(), torch.max(z).item())
        z = self._pre_vq_conv(z)
        if self.DEBUG: print("3. forward: z after _pre_vq_conv", z.shape, torch.min(z).item(), torch.max(z).item())

        if self.use_sonnet_vq:
            loss, quantized, perplexity, _, _, encoding_indices, losses, _, _, _, concatenated_quantized = \
                self._vq_vae(z, record_codebook_stats=False)
            if self.DEBUG:
                print("----- Vector Quantizer Sonnet ----")
                for k,v in losses.items():
                    print(k, v)
                print("----------------------------------")

        else:
            loss, quantized, perplexity, _ = self._vq_vae(z)

        if self.DEBUG: print("3. forward: quantized", quantized.shape, torch.min(quantized).item(), torch.max(quantized).item())

        if self.DEBUG:
            print("z", z.shape)
            print("loss", loss)
            print("quantized", quantized.shape)
            print("perplexity", perplexity)

        x_recon = self._decoder(quantized)
        if self.DEBUG: 
            print("3. forward: reconstruction", x_recon.shape, torch.min(x_recon).item(), torch.max(x_recon).item())

        return loss, x_recon, perplexity