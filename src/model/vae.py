import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth=3, mlp_dropout=0.1):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([nn.Linear(mlp_width, mlp_width)
                         for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x.view(len(x), -1))
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout,
                 gaussian=False):
        super().__init__()
        self.mlp = MLP(input_size, hidden_size, hidden_size, num_layers, dropout)
        self.latent_layer = nn.Linear(hidden_size, output_size)
        self.gaussian = gaussian
        if gaussian:
            self.variance_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.mlp(x)
        if self.gaussian:
            means = self.latent_layer(x)
            log_var = self.variance_layer(x)
            return means, log_var
        else:
            return self.latent_layer(x)
    

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super().__init__()
        self.mlp = MLP(input_size, hidden_size, hidden_size, num_layers, dropout)
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        x = self.mlp(z)
        return self.out_layer(x)
    

class Soft_quantize(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, gumbel=False):
        super().__init__()
        if gumbel:
            self.softmax = lambda x: F.gumbel_softmax(x, tau=1, hard=False)
        else:
            self.softmax = nn.Softmax(dim=-1)
        self.embeddings = nn.Linear(num_embeddings, embedding_dim, bias=False)
    
    def forward(self, x):
        return self.embeddings(self.softmax(x))


class VAE(nn.Module):
    """
    VAE that generate x from z 
    and infer z from x
    """
    def __init__(self, input_size, hidden_size=512, num_layers=3, 
                dropout=0.0, lr=1e-4, loss_type='contrastive',
                weight_decay=0.0, neg_cost=0.1, 
                num_embeddings=10, embedding_dim=64, kl_cost=0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.encoder = Encoder(input_size, hidden_size, num_embeddings, 
                               num_layers, dropout, True)
        self.soft_q = Soft_quantize(embedding_dim, num_embeddings)
        self.decoder = Decoder(embedding_dim, hidden_size, 
                               input_size, num_layers, dropout)
        self.neg_cost = neg_cost
        self.kl_cost = kl_cost

        self.loss_type = loss_type

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # sample from standard normal distribution

        return mu + eps * std

    def forward(self, x):
        
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        quantized = self.soft_q(z)
        recon_x = self.decoder(quantized)

        return recon_x, means, log_var, quantized

    def update(self, minibatch):

        batch_size = minibatch.size(0)

        recon_x, means, log_var, quantized = self.forward(minibatch)
        kld = -0.5 * torch.sum(1 + log_var - log_var.exp() - means.pow(2)) / batch_size

        if self.loss_type == 'contrastive':
            # use Gaussian noise N(0, 1) and prior N(0, 1)
            recon_loss = 0.5 * F.mse_loss(recon_x.view(batch_size, -1), 
                    minibatch.view(batch_size, -1), reduction='sum') / batch_size
            neg_loss = 0

            if self.neg_cost > 0 and len(quantized) > 2:
                q_shift_right = quantized[:-2]
                q_shift_left = quantized[2:]

                neg_loss += F.mse_loss(quantized[1:-1], q_shift_left.detach(), reduction='mean')
                neg_loss += F.mse_loss(quantized[1:-1], q_shift_right.detach(), reduction='mean')

            loss = recon_loss + self.kl_cost*kld - self.neg_cost*neg_loss
        
        elif self.loss_type == 'next-step':

            recon_loss = 0.5 * F.mse_loss(recon_x.view(batch_size, -1)[:-1], 
                    minibatch.view(batch_size, -1)[1:], reduction='sum') / (batch_size-1)
            
            loss = recon_loss + self.kl_cost*kld

        else:
            recon_loss = 0.5 * F.mse_loss(recon_x.view(batch_size, -1), 
                    minibatch.view(batch_size, -1), reduction='sum') / batch_size
            loss = recon_loss + self.kl_cost*kld

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.loss_type == 'contrastive' and neg_loss > 0:
            return {'loss': loss.item(), 'recon_loss': recon_loss.item(), 
                    'kl_loss': kld.item(), 'neg_loss': neg_loss.item()}
        else:
            return {'loss': loss.item(), 'recon_loss': recon_loss.item(), 
                    'kl_loss': kld.item()}

    def predict(self, x):
        means, log_var = self.encoder(x)
        dist = self.soft_q.softmax(means)
        idx = torch.argmax(dist, dim=-1)
        return idx
    
    def get_feature_names_out(self):
        return [f'vae_{i}' for i in range(self.num_embeddings)]



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self._embedding = nn.Embedding(self.num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def encode(self, inputs):
        #inputs shape: [B, N, C]
        inputs = inputs.reshape(inputs.shape[0], -1, self.embedding_dim)
        input_shape = inputs.shape
        inputs = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        encodings = torch.zeros(encoding_indices.shape[0], 
                                  self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        encoding_indices = encoding_indices.view(input_shape[:-1])

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)
        quantized = quantized.view(input_shape[0], -1)
        encodings = encodings.view(input_shape[0], -1, self.num_embeddings)

        return quantized, encoding_indices, encodings

    def forward(self, inputs, compute_loss=True):
        inputs = inputs.contiguous()
        quantized, encoding_indices, encodings = self.encode(inputs)  
        
        if compute_loss:
            # Loss
            e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='mean')
            q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='mean')
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        else:
            loss = None
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-5)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), perplexity, encoding_indices, encodings


class VectorQuantizerEMA(VectorQuantizer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon):
        super(VectorQuantizerEMA, self).__init__(num_embeddings, embedding_dim, commitment_cost)
        
        self.register_buffer('_ema_cluster_size', torch.zeros(self.num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        self._ema_w.data.normal_()
        
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, inputs, compute_loss=True):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.contiguous()
        
        quantized, encoding_indices, encodings = self.encode(inputs)
        encodings_shape = encodings.shape
        encodings = encodings.view(-1, self.num_embeddings)
        inputs = inputs.view(-1, self.embedding_dim)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n)
            
            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * self.decay + (1 - self.decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        encodings = encodings.view(encodings_shape)
        inputs = inputs.view(encodings_shape[0], -1)

        if compute_loss:
            # Loss
            e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='mean')
            loss = self.commitment_cost * e_latent_loss
        else:
            loss = None
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), perplexity, encoding_indices, encodings
    

class VQ_VAE(nn.Module):
    """
    VQ VAE that generate x from quantized z
    each discrete latent is selected with prob 1 and a uniform prior
    and infer z from x
    """

    def __init__(self, input_size, hidden_size=128, num_layers=3, 
                dropout=0.0, lr=1e-5, 
                weight_decay=0.0, neg_cost=0.05, n_perfix_per_step=2,
                num_embeddings=10, embedding_dim=64, 
                commitment_cost=0.25, vq_decay=0.99, epsilon=1e-5):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, 
                               n_perfix_per_step*embedding_dim, num_layers, dropout)
        self.neg_cost = neg_cost
        self.n_perfix_per_step = n_perfix_per_step

        if vq_decay > 0.0:
            self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                         commitment_cost, vq_decay, epsilon)
        else:
            self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = Decoder(n_perfix_per_step*embedding_dim, hidden_size, 
                               input_size, num_layers, dropout)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, encodings, encodings_one_hot = self.vq(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity, z, quantized, encodings

    def update(self, minibatch):
        vq_loss, x_recon, perplexity, z, quantized, encodings = self.forward(minibatch)
            
        recon_loss = F.mse_loss(x_recon, minibatch, reduction='mean')

        neg_loss = 0
        q_shift_right = torch.concat([quantized[2].unsqueeze(0), quantized[:-1]], 0)
        q_shift_left = torch.concat([quantized[1:], quantized[-3].unsqueeze(0)], 0)

        neg_loss += F.mse_loss(quantized, q_shift_left.detach(), reduction='mean')
        neg_loss += F.mse_loss(quantized, q_shift_right.detach(), reduction='mean')

        loss = recon_loss + vq_loss - self.neg_cost*neg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'recon_loss': recon_loss.item(), 
                'vq_loss': vq_loss.item(), 'neg_loss': neg_loss.item()}

    def encode(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, encodings, encodings_one_hot = self.vq(z, compute_loss=False)

        return encodings