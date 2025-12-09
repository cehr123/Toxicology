from typing import Optional, Tuple
import math
import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torchmdnet.models.utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)
from torchmdnet.models.feature_embedding import FeatureEmbedding

class TorchMD_ET(nn.Module):
    r"""The TorchMD equivariant Transformer architecture.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom.
            (default: :obj:`32`)
    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        num_atom_types=5,
        max_num_neighbors=32,
        layernorm_on_vec=None,
        use_total_charge=False,
        use_energy_feature=False,
        use_smiles=False,
        use_atom_props=False,
        use_edge_attention=True,
    ):
        super(TorchMD_ET, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.num_atom_types = num_atom_types
        self.layernorm_on_vec = layernorm_on_vec
        self.use_total_charge = use_total_charge
        self.use_energy_feature = use_energy_feature
        self.use_smiles = use_smiles
        self.use_atom_props = use_atom_props
        self.use_edge_attention = use_edge_attention
        
        act_class = act_class_mapping[activation]
        
        num_atom_props = 10
        
        self.charge_embedding = FeatureEmbedding(
            1, hidden_channels, bias=False, max_z=self.max_z
        ) if self.use_total_charge else None
        
        self.energy_embedding = nn.Linear(1, self.hidden_channels,
        ) if self.use_energy_feature else None
        
        self.smiles_embedding = FeatureEmbedding(
            128, hidden_channels, bias=False, max_z=self.max_z
        ) if self.use_smiles else None
        
        self.embedding = nn.Embedding(self.max_z, hidden_channels) if not self.use_atom_props else nn.Linear(num_atom_props, hidden_channels)
            
        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z, self.num_atom_types,
            )#.jittable()
            if neighbor_embedding
            else None
        )

        ''' added edge layers that we will alternate with attention layers'''
        self.attention_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        for _ in range(num_layers):
            num_angle_bins = 6
            edge_layer = EdgeEmbedding(
                hidden_channels,
                num_angle_bins=num_angle_bins,
                activation = nn.SiLU,
                use_attention=self.use_edge_attention,
            )
            self.edge_layers.append(edge_layer)

            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
                edge_emb_dim=edge_layer.output_dim,
            )
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)
        if self.layernorm_on_vec:
            if self.layernorm_on_vec == "whitened":
                self.out_norm_vec = EquivariantLayerNorm(hidden_channels)
            else:
                raise ValueError(f"{self.layernorm_on_vec} not recognized.")
            
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        y: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
        smiles: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)

        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"
        
        x = self.embedding(z)

        if self.use_total_charge:
            assert (Q is not None), "Use total charge True, but no Q found!"
            x = x + self.charge_embedding(Q[batch], z=z)
        if self.use_energy_feature:
            assert (y is not None), "Use energy as feature True, but no y found!"
            #num_atoms = batch.bincount()
            #y = y / num_atoms.unsqueeze(1)
            x = x + self.energy_embedding(y[batch])
        if self.use_smiles:
            assert (smiles is not None), "Use smiles as feature True, but no smiles found!"
            x = x + self.smiles_embedding(smiles[batch], z=z)
        
        edge_attr = self.distance_expansion(edge_weight)
        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)

        for i, attn in enumerate(self.attention_layers):
            ''' Alternate Edge embeddings with Node embeddings '''
            edge_emb = self.edge_layers[i](x, pos, edge_index)

            dx, dvec = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec, edge_emb)
            x = x + dx
            vec = vec + dvec
        x = self.out_norm(x)
        if self.layernorm_on_vec:
            vec = self.out_norm_vec(vec)
        
        return x, vec, z, pos, batch, edge_index, edge_weight

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class EdgeEmbedding(MessagePassing):
    def  __init__(
        self,
        hidden_channels,
        num_angle_bins,
        activation,
        use_attention=True,
    ):
        super(EdgeEmbedding, self).__init__(aggr="add", node_dim=0)
        self.hidden_channels = hidden_channels
        self.num_angle_bins = num_angle_bins
        self.act = activation()
        self.use_attention = use_attention

        # create the inital edge embeding from the two node emebdings 
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, hidden_channels)
        )

        # create the angle bins for the edge embedding
        self.register_buffer(
            "angle_bins",
            torch.linspace(-1, 1, num_angle_bins + 1),
            persistent=False,
        )

        # self.edge_proj = nn.Linear(hidden_channels * num_angle_bins, hidden_channels)
        
        # final dimensionality matches concatenation over all buckets
        self.output_dim = hidden_channels * num_angle_bins

        self.query_proj = nn.Linear(hidden_channels, hidden_channels)
        self.key_proj = nn.Linear(hidden_channels, hidden_channels)
        self.value_proj = nn.Linear(hidden_channels, hidden_channels)
        self.attn_scale = 1.0 / math.sqrt(hidden_channels)
        self.neg_inf = -1e9

    # x all the node embeddings, all the node's positions, connections between nodes
    def forward(self, x, pos, edge_index):
        # get the two nodes that we're learning the edge emebedding for
        node_i, node_j = edge_index
        
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return x.new_zeros(x.size(0), self.output_dim)
        
        # initial edge representation
        e = self.edge_mlp(torch.cat([x[node_i], x[node_j]], dim=-1))
        # by subtracting the positions of each node we get a vector that points from one node to the next
        edge_vec = pos[node_j] - pos[node_i]
        # normalize so it is a unit vector (1e-8 is preventing us from dividing by 0)
        edge_vec = edge_vec / (edge_vec.norm(dim=1, keepdim=True) + 1e-8)

        # equation variables
        q = self.query_proj(e)
        k = self.key_proj(e)
        v = self.value_proj(e)


        # this is some torchgeometric magic, should call message
        out = self.propagate(edge_index, x=x, edge_vec=edge_vec, e=e, q=q, k=k, v=v, size=None)
        
        return out

    def message(self, e_j, edge_vec_i, edge_vec_j, index, ptr, size_i, q_i, k_j, v_j):
        # compute the cosin of the angle between the two atoms, this will be used to divide into the N buckets
        cos_theta = (edge_vec_i * edge_vec_j).sum(dim=-1).clamp(-1.0, 1.0)
        # figure out what bin it belongs in based on the angle
        bin_ids = torch.bucketize(
            cos_theta, self.angle_bins.to(cos_theta.device)
        ) - 1
        bin_ids = torch.clamp(bin_ids, 0, self.num_angle_bins - 1)
        # put the value only in the correct bucket
        one_hot = torch.nn.functional.one_hot(
            bin_ids, num_classes=self.num_angle_bins
        ).float()

        # e_expanded = e_j.unsqueeze(2) * one_hot.unsqueeze(1)
        # return e_expanded

        values = e_j.unsqueeze(2) * one_hot.unsqueeze(1)
        if self.use_attention:
            attn_logits = (q_i * k_j).sum(dim=-1) * self.attn_scale
            logits = one_hot * attn_logits.unsqueeze(1) + (1 - one_hot) * self.neg_inf
        else:
            logits = one_hot
        return values, logits

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # sum all values in inputs that correspond to each index
        # m = scatter_sum(inputs, index, dim=0) 
        # concatenate those to form a final embeding

        values, logits = inputs
        if dim_size is None:
            dim_size = int(index.max().item() + 1) if index.numel() > 0 else 0

        if dim_size == 0:
            return values.new_zeros(0, self.output_dim)

        if not self.use_attention:
            summed = scatter(values, index, dim=0, dim_size=dim_size)
        else:
            num_edges = values.size(0)
            num_bins = self.num_angle_bins
            hidden = self.hidden_channels

            values_flat = values.permute(0, 2, 1).reshape(-1, hidden)
            logits_flat = logits.reshape(-1)

            index_expanded = index.unsqueeze(1).expand(-1, num_bins).reshape(-1)
            bucket_ids = torch.arange(num_bins, device=index.device).view(1, -1).expand(num_edges, -1).reshape(-1)
            group_index = index_expanded * num_bins + bucket_ids

            valid_mask = logits_flat > self.neg_inf / 2
            if valid_mask.any():
                logits_valid = logits_flat[valid_mask]
                values_valid = values_flat[valid_mask]
                group_valid = group_index[valid_mask]

                attn = softmax(
                    logits_valid, group_valid, num_nodes=dim_size * num_bins
                )
                weighted = values_valid * attn.unsqueeze(-1)
                summed = scatter(
                    weighted, group_valid, dim=0, dim_size=dim_size * num_bins
                )
                summed = summed.view(dim_size, num_bins, hidden).permute(0, 2, 1)
            else:
                summed = values.new_zeros(dim_size, hidden, num_bins)

        #e_out = summed.flatten(start_dim=1)
        #e_out = self.edge_proj(e_out)
        e_out = summed.reshape(dim_size, -1)
        return e_out



class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
        edge_emb_dim=None,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.edge_emb_dim = edge_emb_dim or hidden_channels
        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.reset_parameters()

        ''' small MLP to combine value and edge embedding for node representation learning '''
        fusion_in_dim = (
            3 * hidden_channels // num_heads + num_rbf + self.edge_emb_dim
        )
        fusion_out_dim = 3 * hidden_channels // num_heads

        self.edge_fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_out_dim),
            activation(),
            nn.Linear(fusion_out_dim, fusion_out_dim),
        )

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)


    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, edge_emb):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor, f_ij: Tensor, edge_emb: Tensor)
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            f_ij=f_ij,
            edge_emb=edge_emb,
            size=None,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij, f_ij, edge_emb):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        ''' Combine the value embedding with the edge embedding, one copy of the edge embeding for each head''' 
        edge_emb_expanded = edge_emb.unsqueeze(1).expand(-1, self.num_heads, -1)
        f_ij_expanded = f_ij.unsqueeze(1).expand(-1, self.num_heads, -1)

        # Concatenate per-head features
        fused = torch.cat([v_j, f_ij_expanded, edge_emb_expanded], dim=-1)
        # fused: [E, num_heads, 226]

        # Flatten across heads before MLP
        fused = fused.reshape(-1, fused.shape[-1])  # [E*num_heads, 226]

        # Apply MLP
        v_j = self.edge_fusion(fused)               # [E*num_heads, 48]

        # Restore per-head structure
        v_j = v_j.reshape(-1, self.num_heads, 3 * self.head_dim)


        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

class EquivariantLayerNorm(nn.Module):
    r"""Rotationally-equivariant Vector Layer Normalization
    Expects inputs with shape (N, n, d), where N is batch size, n is vector dimension, d is width/number of vectors.
    """
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None) # Without bias term to preserve equivariance!

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        """Compute the inverse square root of a positive definite matrix.

        Based on https://github.com/pytorch/pytorch/issues/25481
        """
        _, s, v = matrix.svd()
        good = (
            s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        )
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(
            -2, -1
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float64) # Need double precision for accurate inversion.
        input = self.mean_center(input)
        # We use different diagonal elements in case input matrix is approximately zero,
        # in which case all singular values are equal which is problematic for backprop.
        # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html
        reg_matrix = (
            torch.diag(torch.tensor([1.0, 2.0, 3.0]))
            .unsqueeze(0)
            .to(input.device)
            .type(input.dtype)
        )
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(
            self.weight.dtype
        ) * self.weight.reshape(1, 1, self.normalized_shape[0])

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, "
            "elementwise_linear={elementwise_linear}".format(**self.__dict__)
        )