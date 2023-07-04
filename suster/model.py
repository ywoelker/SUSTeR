from modules.context import ContextAwareMeanState
from modules.embedding import Embedding
from modules.projection import LocationBasedOutputv2, SimpleOutput
import torch
from torch import nn
import torch.nn.functional as F
from stgcn.stgcn_arch import STGCNChebGraphConv
import numpy as np



class SUSTeR(nn.Module):


    def __init__(self, n_proxy, embed_dim, n_channels, target, context_meanstate = False, norm_adj = False, factor = 1) -> None:

        super().__init__()

        self.norm_adj = norm_adj
        self.n_proxy = n_proxy
        self.embed_dim = embed_dim

        if context_meanstate:
            self.mean_state = ContextAwareMeanState(n_proxy, embed_dim, 2)
        else:
            self.mean_state = nn.Parameter(torch.randn(size = (self.n_proxy, self.embed_dim)) * .5, requires_grad= True)

      
        self.embedding = Embedding(embed_dim, n_channels, n_proxy)
        self.factor = factor
        if factor is not None:
            self.stgcn = STGCNChebGraphConv(3, 3, [[embed_dim], [64//factor, 16//factor, 64//factor], [64//factor, 16//factor, 64//factor], [128//factor, 128//factor], [embed_dim]], 12, n_proxy, 'glu', 'cheb_graph_conv', True, .5)


        if np.isscalar(target):
            self.output = SimpleOutput(n_proxy, embed_dim, target)
        else:
            self.output = LocationBasedOutputv2(n_proxy, embed_dim, target)

        

    def forward(self, observations, target_positions):
        """
        Processes a sequence of observation timesteps into a full proxy network.
        From this network the target values are extracted.

        # Parameters
        - observartions [batch, time, sensors, channels]
        - target_positions [n, 4] 
        """
        nB, nT, nS, nC = observations.shape

        # 1. Get the mean state from the context information.
        if isinstance(self.mean_state, nn.Module):
            mean_state = self.mean_state(target_positions[:, 0, :2])
        else:
            mean_state = self.mean_state
            mean_state = torch.tile(mean_state, dims = (nB, 1, 1))
        

        # 2. Create an graph for each of the provided timesteps
        delta_proxies = [mean_state] 
        p_densities = []

        for timestep in range(nT):
            
            obs_t = observations[:, timestep] # [batch, sensors, channels]

            # 2.1 Embed each observation into an embedding for each of the proxies
            embeddings, proxy_density = self.embedding(obs_t, delta_proxies[-1].detach()) # [batch, sensors, proxy * embed]

            p_densities.append(proxy_density)

            # 2.2 Sum over all the emebeddings which count to real observations
            mask_observations = torch.any(obs_t != 0, dim = -1).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1,-1, self.n_proxy, self.embed_dim) # [batch, sensors, 1, 1]
            embeddings_mul = torch.mul(embeddings , mask_observations.int()) # Set all the non existing observation to zero

            batch_n_obs = torch.sum(mask_observations[:, :, 0, 0], dim = 1) # b, s, p, e
            batch_mask = batch_n_obs.squeeze() > 0 # b, p, e
            delta_proxy_embedding = torch.zeros(size= (nB, self.n_proxy, self.embed_dim), device = observations.device)

            delta_proxy_embedding[batch_mask] = torch.sum(embeddings_mul[batch_mask], dim = 1)
            delta_proxies.append(delta_proxy_embedding)


        delta_proxies = torch.stack(delta_proxies, dim = 1)
        graphs = torch.cumsum(delta_proxies, dim = 1)

        # 3. Calculate the laplace matrix for the graph sequence from the last hidden state.
        if not self.norm_adj:        
            adj_mx = torch.softmax(torch.relu(torch.einsum('bpe,ble->bpl', graphs[:, -1], graphs[:,-1])), dim = -1)
        else:
            norm_graph = graphs[:, -1]/ (torch.norm(graphs[:, -1], p = 2, dim = -1, keepdim= True) + 1e-6)
            adj_mx = torch.softmax(torch.relu(torch.einsum('bpe,ble->bpl',norm_graph, norm_graph)), dim = -1)

        # 4. Use the wrapped STGNN for the forward propagation of information through the hidden spatio temporal graph.
        if self.factor is not None:
            full_state = self.stgcn(graphs[:, 1:], adj_mx) # [b, 1, p, e]
        else:
            full_state = torch.mean(graphs[:, 1:], dim = 1)


        # 5. Project the full state onto the spatial domain
        out = self.output(full_state.reshape(nB, -1), target_positions)


        return out.reshape(nB, 1, -1, 1), self.output(mean_state.reshape(nB, -1), target_positions).reshape(nB, 1, -1, 1)
