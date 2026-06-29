"""
Magnitude-Direction (MD) Decoupled Linear layer.

The idea (from arXiv:2606.25971):
    Every weight matrix W can be split into
        - a DIRECTION  (what the feature does)   -> kept on a fixed-norm sphere
        - a MAGNITUDE  (how loud the feature is) -> small per-row / per-col "gains"

We write the effective weight as:

        W[i,j] = g_row[i] * D[i,j] * g_col[j]

where:
        D      is a (out, in) matrix normalized to a fixed Frobenius norm,
               so only its ORIENTATION can change, never its scale.
        g_row  is a (out,)  vector of per-output gains  (one number per row)
        g_col  is a (in,)   vector of per-input  gains  (one number per column)

The model still sees a single fused weight tensor W, but we can control
direction and magnitude separately. In particular we can:

    - freeze the direction  (D.requires_grad = False)
    - keep training the gains (g_row, g_col)

which is the basis of the continual-learning recipe:
    "Frozen directions, living gains".
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # --- DIRECTION ---------------------------------------------------
        # Raw direction parameter. We keep it on a fixed-norm sphere by
        # dividing by its (detached) norm in _W(), so its scale is meaningless
        # and only its orientation can be learned. We initialize it so that,
        # after normalization, its entries are ~ N(0, 1).
        self.direction = nn.Parameter(torch.randn(out_features, in_features))
        target_norm = math.sqrt(out_features * in_features)  # makes entries ~N(0,1)
        self.register_buffer("dir_norm", torch.tensor(float(target_norm)))

        # --- MAGNITUDE (gains) -------------------------------------------
        # Start them so that the effective W is ~ Kaiming initialized:
        #   W[i,j] = (1/sqrt(in)) * D[i,j] * 1   with D[i,j] ~ N(0,1)
        #         => W[i,j] ~ N(0, 1/in)   (standard for a Linear layer)
        self.g_row = nn.Parameter(torch.full((out_features,), 1.0 / math.sqrt(in_features)))
        self.g_col = nn.Parameter(torch.ones(in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # --- per-task gain storage (for continual learning) --------------
        # We store one (g_row, g_col, bias) triple per task so we can switch
        # back to an old task's full per-task state at test time without
        # retraining. The LIVE parameters (self.g_row, self.g_col, self.bias)
        # always represent the CURRENT task. When we switch tasks we first
        # save the live params back into the dict, then load the requested
        # task's params into the live tensors. If the dict is empty (naive /
        # joint training, single shared state) set_task() is a no-op.
        self.task_params: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.current_task = 0

    # ------------------------------------------------------------------ #
    #  Build the effective weight matrix W from direction + gains.        #
    # ------------------------------------------------------------------ #
    def effective_weight(self) -> torch.Tensor:
        # Keep D on the sphere: normalize to self.dir_norm, detaching the norm
        # so no gradient flows into the (meaningless) scale of `direction`.
        d = self.direction
        Dn = d * (self.dir_norm / d.norm().detach())
        # W[i,j] = g_row[i] * Dn[i,j] * g_col[j]
        W = self.g_row.unsqueeze(1) * Dn * self.g_col.unsqueeze(0)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    # ------------------------------------------------------------------ #
    #  Continual-learning controls.                                       #
    # ------------------------------------------------------------------ #
    def freeze_direction(self) -> None:
        """Stop the direction from changing. Gains (and bias) stay trainable."""
        self.direction.requires_grad_(False)

    def unfreeze_direction(self) -> None:
        self.direction.requires_grad_(True)

    def new_task(self) -> None:
        """
        Call this when moving to a new task.

        1. Save the CURRENT task's (live) gains + bias into storage.
        2. Reset the live gains to the default initialization and zero the
           bias, so the new task starts from a clean 'adapter' on top of the
           (frozen) direction.
        """
        b = self.bias.detach().clone() if self.bias is not None else None
        self.task_params[self.current_task] = (self.g_row.detach().clone(),
                                                self.g_col.detach().clone(), b)
        self.current_task += 1
        with torch.no_grad():
            self.g_row.fill_(1.0 / math.sqrt(self.in_features))
            self.g_col.fill_(1.0)
            if self.bias is not None:
                self.bias.zero_()

    def set_task(self, t: int) -> None:
        """
        Load the params (gains + bias) for task `t` into the live parameters
        (for evaluation). The currently-live params are saved back to storage
        first, so nothing is lost. If no per-task state has ever been stored
        (naive / joint training, single shared state), this is a no-op.
        """
        if len(self.task_params) == 0:
            return  # single shared state; nothing to switch
        if t == self.current_task:
            return  # already live
        # commit the currently-live params back to their slot
        b = self.bias.detach().clone() if self.bias is not None else None
        self.task_params[self.current_task] = (self.g_row.detach().clone(),
                                                self.g_col.detach().clone(), b)
        gr, gc, gb = self.task_params[t]
        with torch.no_grad():
            self.g_row.copy_(gr)
            self.g_col.copy_(gc)
            if self.bias is not None and gb is not None:
                self.bias.copy_(gb)
        self.current_task = t


class MDMLP(nn.Module):
    """A plain MLP where every Linear layer is an MDLinear layer.

    Architecture: input -> hidden -> hidden -> ... -> output, with ReLU.
    No CNN, no convolutions -- just a multi-layer perceptron.
    """

    def __init__(self, sizes: list[int], hidden_act: type[nn.Module] = nn.ReLU):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(MDLinear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # no activation after the final layer
                layers.append(hidden_act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # --- group controls over all MDLinear layers ----------------------- #
    def freeze_direction(self) -> None:
        for m in self.net:
            if isinstance(m, MDLinear):
                m.freeze_direction()

    def unfreeze_direction(self) -> None:
        for m in self.net:
            if isinstance(m, MDLinear):
                m.unfreeze_direction()

    def new_task(self) -> None:
        for m in self.net:
            if isinstance(m, MDLinear):
                m.new_task()

    def set_task(self, t: int) -> None:
        for m in self.net:
            if isinstance(m, MDLinear):
                m.set_task(t)

    def trainable_gain_params(self) -> list[nn.Parameter]:
        """Return only the gain (+ bias) parameters, skipping frozen directions."""
        params = []
        for m in self.net:
            if isinstance(m, MDLinear):
                params.append(m.g_row)
                params.append(m.g_col)
                if m.bias is not None:
                    params.append(m.bias)
        return params

    def per_layer_direction_grad_flat(self) -> list[torch.Tensor]:
        """Each MDLinear direction's .grad flattened to a 1D tensor (detached),
        in order. Used to compute cosine similarity between the direction
        gradient on a NEW task and on an OLD-task memory (A-GEM-style conflict
        measure). Call right after a backward() with directions unfrozen.
        Returns detached tensors on the same device as the params."""
        out = []
        for m in self.net:
            if isinstance(m, MDLinear):
                g = m.direction.grad
                out.append(g.detach().flatten().clone() if g is not None
                           else torch.zeros(m.direction.numel(), device=m.direction.device))
        return out
