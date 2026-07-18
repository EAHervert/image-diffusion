
import torch
import torch.nn.functional as func
import torch.nn as nn


class PositionwiseFFN(nn.Module):
    """
    FFN(x) = GELU(x W_1 + b_1) W_2 + b_2, applied per-position.
    """
    def __init__(self, d_model=384, d_ff=1536, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # W_1, b_1
        self.linear2 = nn.Linear(d_ff, d_model)   # W_2, b_2
        self.dropout = dropout

    def forward(self, x):
        # x: (B, n, d_model). nn.Linear broadcasts over batch/position, the same (W_1, b_1) is applied.
        out = func.gelu(self.linear1(x))

        if self.dropout > 0.0:
            out = func.dropout(out, p=self.dropout, training=self.training)

        return self.linear2(out)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=384, num_heads=6, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # One big projection per role; reshape into h heads at forward time.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        # (B, n, d_model) -> (B, n, h, d_k) -> (B, h, n, d_k).
        B, n, _ = x.shape

        return x.reshape(B, n, self.num_heads, self.d_k).transpose(1, 2)

    @staticmethod
    def merge_heads(x):
        # (B, h, n, d_k) -> (B, n, d_model).
        B, h, n, d_k = x.shape

        return x.transpose(1, 2).reshape(B, n, h * d_k)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Q, K, V: (Q)uery, (K)ey, and (V)alue matrices with last dimensions (n, d_k)
        Returns: output (..., n_q, d_v)
        """

        scores = torch.matmul(Q * (Q.size(-1) ** -0.5), K.transpose(-2, -1))
        attn = func.softmax(scores, dim=-1)

        if self.dropout > 0.0:
            attn = func.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, V)

        return out # softmax(Q K^T / sqrt(d_k)) V

    def forward(self, X):
        """
        Same tensor input to get Q, K, V
        X: (B, n, d_model) input embedding.
        Returns: (B, n, d_model) — self-attention preserves shape. 
        """        

        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        out = self.scaled_dot_product_attention(Q, K, V)
        out = self.merge_heads(out)

        return self.W_o(out)


class DiTBlock(nn.Module):
    """
    Pre-norm DiT block with AdaLN-Zero modulation (for class-aware training): 
    AdaLN -> self-attention -> residual -> AdaLN -> FFN -> residual.
    AdaLN uses conditioning vector c derived from class and time step in noise generation.
    (B, n, d_model) x (B, d_model) -> (B, n, d_model)
    """
    def __init__(self, d_model=384, num_heads=6, d_ff=1536, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, self.dropout)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # One MLP -> 6 modulation vectors: (shift, scale, gate) x 2 sublayers.
        self.modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )

        # Zero-init - all 6 modulation vectors are 0 at step 0.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, X, c):
        """
        X: (B, n, d_model), c: (B, d_model)
        returns (B, n, d_model)
        """

        shift_1, scale_1, gate_1, shift_2, scale_2, gate_2 = self._get_modulation_scalars(c)

        X = X + gate_1 * self.attn(self._adaln(X, norm=self.norm1, shift=shift_1, scale=scale_1))
        X = X + gate_2 * self.ffn(self._adaln(X, norm=self.norm2, shift=shift_2, scale=scale_2))

        return X

    def _get_modulation_scalars(self, c):
        """
        c: (B, d_model) -> shift_1, scale_1, gate_1, shift_2, scale_2, gate_2 all (B, 1, d_model)
        """

        modulation = self.modulation(c).unsqueeze(1)

        return modulation.chunk(6, dim=-1)  # shift_1, scale_1, gate_1, shift_2, scale_2, gate_2

    @staticmethod
    def _adaln(x, norm, shift, scale):
        """adaLN: normalize x, then apply scale and shift derived from c."""

        return norm(x) * (1 + scale) + shift


class FinalLayer(nn.Module):
    """
    Final adaLN + linear projection to patch-pixel space.
    Zero-inits both the modulation Linear and the head Linear so that
    at init the model outputs the zero tensor (predicts zero velocity/noise).
    """
    def __init__(self, d_model, patch_size, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        self.head = nn.Linear(d_model, patch_size * patch_size * in_channels, bias=True)

        # AdaLN-Zero: modulation returns zero at init -> identity modulation
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

        # Zero-init the head so the model outputs zero at init
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, X, c):
        shift, scale = self.modulation(c).unsqueeze(1).chunk(2, dim=-1)
        X = DiTBlock._adaln(X, self.norm, shift, scale)
        return self.head(X)


class Embedding(nn.Module):
    def __init__(self, image_size=(128, 128), num_classes=10,
                 in_channels=3, d_model=384, patch_size=8, 
                 t_scale=1000.0, timestep_freq_dim=256):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.grid_h = self.image_size[0] // self.patch_size
        self.grid_w = self.image_size[1] // self.patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.t_scale = t_scale
        self.timestep_freq_dim = timestep_freq_dim

        # Learned MLP mapping sinusoidal timestep features (timestep_freq_dim,) -> (d_model,).
        self.timestep_mlp = nn.Sequential(
            nn.Linear(timestep_freq_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            )

        self.class_embedding_map = nn.Embedding(self.num_classes, self.d_model)

        # Convolution used to map image tensor to token
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels, out_channels=self.d_model,
            kernel_size=self.patch_size, stride=self.patch_size, bias=True,
        )

        # Fixed 2D sinusoidal positional embedding, shape (1, num_patches, d_model).
        pos_embed = self._build_2d_sincos_pos_embed(
            d_model=d_model,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
        )

        # Attach to module lifecycle but not adjusted during training.
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))

    def forward(self, x, t, y):
        return self.positional_embed(x), self.conditioning_embed(t, y)

    def timestep_embed(self, t):
        """
        Timestep t -> conditioning-vector-space features.

        Args:
            t: (B,) float tensor of timesteps in [0, 1].
        Returns:
            (B, d_model) tensor, ready to be fused additively with the class embedding.
        """

        t_scaled = t * self.t_scale
        t_sinusoid = self.encode_1d(t_scaled, self.timestep_freq_dim)  # (B, timestep_freq_dim)

        return self.timestep_mlp(t_sinusoid)

    def class_embed(self, classes):
        """
        classes -> embedded classes.

        Args:
            classes: (B,) long tensor of class indices in [0, num_classes).
        Returns:
            (B, d_model) tensor, ready to be fused with the time embedding.
        """

        return self.class_embedding_map(classes)

    def conditioning_embed(self, t, classes):
        """
        Fuse timestep and class label into a single conditioning vector c.

        Args:
            t: (B,) float tensor of timesteps in [0, 1].
            classes: (B,) long tensor of class indices in [0, num_classes).
        Returns:
            (B, d_model) conditioning vector.
        """

        return self.timestep_embed(t) + self.class_embed(classes)

    def positional_embed(self, x):
        """
        Image tensor x: (B, C, H, W) to tokens (B, num_patches, d_model)
        gh, gw = H // patch_size, W // patch_size, n = gh * gw
        """

        tokens = self.patch_embed(x)  # (B, C, H, W) -> (B, d_model, gh, gw)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, d_model, gh, gw) -> (B, d_model, gh * gw) -> (B, n, d_model)
        tokens = tokens + self.pos_embed  # pos_embed (1, n, d_model) broadcasted over B
        return tokens

    @staticmethod
    def _build_2d_sincos_pos_embed(d_model, grid_h, grid_w):
        """
        Fixed 2D sinusoidal positional embedding for a grid_h x grid_w patch grid.

        Args:
            d_model, grid_h, grid_w
        Returns:
            (grid_h * grid_w, d_model) tensor
        """
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        d_half = d_model // 2   # dims used per spatial axis

        # Coordinate grid for the patch grid.
        rows = torch.arange(grid_h, dtype=torch.float32)
        cols = torch.arange(grid_w, dtype=torch.float32)
        emb_rows = Embedding.encode_1d(rows, d_half)  # (grid_h, d_half) - one vector per row idx
        emb_cols = Embedding.encode_1d(cols, d_half)  # (grid_w, d_half) - one vector per col idx

        # Place them on a 2D grid via broadcasting - pos_embed_2d[i, j, :] = concat(emb_rows[i], emb_cols[j])
        emb_rows_2d = emb_rows.unsqueeze(1).expand(grid_h, grid_w, d_half)
        emb_cols_2d = emb_cols.unsqueeze(0).expand(grid_h, grid_w, d_half)
        pos_embed_2d = torch.cat([emb_rows_2d, emb_cols_2d], dim=-1)  # (gh, gw, d_model)

        # Flatten the 2D grid to a token sequence.
        return pos_embed_2d.reshape(grid_h * grid_w, d_model)  # (n, d_model)

    @staticmethod
    def encode_1d(pos, d):
        # pos: (n,); d: even -> returns (n, d) sinusoidal frequencies
        omega = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        angles = pos.unsqueeze(1) * omega.unsqueeze(0)   # (n, 1) * (1, d/2) -> (n, d/2)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def unpatchify(self, X):
        """
        Geometric inverse of the spatial patchify layout.

        Args:
            X: (B, n, P*P*C) where n = grid_h * grid_w, and each token holds
        Returns:
            (B, self.in_channels, H, W) where H = grid_h * P, W = grid_w * P.
        """

        gh, gw = self.grid_h, self.grid_w
        X = X.reshape(X.shape[0], gh, gw, self.patch_size, self.patch_size, self.in_channels)  # (B, gh, gw, ph, pw, rgb_channels)
        X = X.permute(0, 5, 1, 3, 2, 4)  # (B, gh, gw, ph, pw, rgb_channels) -> (B, rgb_channels, gh, ph, gw, pw)

        return X.reshape(X.shape[0], self.in_channels, gh * self.patch_size, gw * self.patch_size)   # -> (B, rgb_channels, gh * ph, gw * pw)


class DiT(nn.Module):
    """
    DiT-S/8 encoder-only diffusion transformer for class-conditional image generation.
    Predicts a velocity field (flow-matching) or noise (DDPM) at every pixel,
    given a noisy image x, timestep t, and class label y.
    """
    def __init__(
        self,
        image_size=(128, 128),
        in_channels=3,
        d_model=384,
        depth=12,
        num_heads=6,
        d_ff=1536,
        patch_size=8,
        num_classes=10,
        dropout=0.0,
        t_scale=1000.0,
        timestep_freq_dim=256,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_size = patch_size

        # Mappings from (x, t, y) -> (tokens, c) and from output back to x (image)
        self.embeddings = Embedding(
            image_size=image_size,
            num_classes=num_classes,
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            t_scale=t_scale,
            timestep_freq_dim=timestep_freq_dim,
        )

        # Transformer stack: depth number DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(depth)
        ])

        # Final adaLN (shift + scale only, this is the last projection)
        # AdaLN-Zero taken care of in the FinalLayer class
        self.final_layer = FinalLayer(d_model, patch_size, in_channels)

    def forward(self, x, t, y):
        X, c = self.embeddings(x, t, y)  # Transform input image and class labels to the image tokens and class conditioning (with time)

        # Pass tokens and class conditioning through the encoder Transformer (DiT)
        for block in self.blocks:
            X = block(X, c)
        
        # Apply final layer on to the output of the DiT layers
        X = self.final_layer(X, c)

        return self.embeddings.unpatchify(X)  # Transform output to image shape using unpatchify
