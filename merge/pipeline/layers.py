from typing import Tuple, Dict, Optional
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.activations import GELU, GEGLU, ApproximateGELU
from diffusers.utils import deprecate
from diffusers.models.normalization import PixArtAlphaCombinedTimestepSizeEmbeddings

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        sigmoid_output: bool = False,
        affine_func=nn.Linear,
        act_fn="relu"
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        if act_fn == "gelu_tanh":
            self.act = nn.GELU(approximate="tanh")
        elif act_fn == "gelu":
            self.act = nn.GELU()
        elif act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "relu":
            self.act = nn.ReLU()
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class ResidualMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_mlp,
        num_layer_per_mlp,
        sigmoid_output: bool = False,
        affine_func=nn.Linear,
        act_fn="relu"
    ):
        super().__init__()
        self.num_mlp = num_mlp
        self.in2hidden_dim = affine_func(input_dim, hidden_dim)
        self.hidden2out_dim = affine_func(hidden_dim, output_dim)
        self.mlp_list = nn.ModuleList(
            MLP(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                num_layer_per_mlp,
                affine_func=affine_func,
                act_fn=act_fn
            ) for _ in range(num_mlp)
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor):
        x = self.in2hidden_dim(x)
        for mlp in self.mlp_list:
            out = mlp(x)
            x = x + out
        out = self.hidden2out_dim(x)
        return out


class _FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))
            self.final_dropout = True
        else:
            self.final_dropout = False
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        hidden_states = self.net[0](hidden_states)
        hidden_states = self.net[1](hidden_states)
        hidden_states = self.net[2](hidden_states)

        if self.final_dropout:
            hidden_states = self.net[3](hidden_states)
        return hidden_states

class _AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False, timestep_dim: int = None):
        super().__init__()

        if timestep_dim is None:
            timestep_dim = embedding_dim

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * timestep_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep