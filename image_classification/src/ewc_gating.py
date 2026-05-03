# -----------------------------
# 1) Model with task gating
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskGatedCNN(nn.Module):
    def __init__(
            self,
            n_channels=1,
            num_classes=2,
            num_tasks=5,
            gate_frac=0.2,
            hidden_dim=1024,
            bias_conv=True,
            bias_dense=True,
            use_gating=True,
            use2dense=True
    ):
        super().__init__()
        assert 0.0 < gate_frac <= 1.0
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.gate_frac = gate_frac
        self.use_gating = use_gating
        self.use2dense = use2dense

        # 1. Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=bias_conv),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=bias_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # 14 -> 7

            nn.Flatten(),
        )

        self.feat_dim = 64 * 7 * 7  # 3136

        # 2. Determine Pre-Output Dimension and Setup Hidden Layer
        if self.use2dense:
            self.fc1 = nn.Linear(self.feat_dim, hidden_dim, bias=bias_dense)
            self.act = nn.ReLU(inplace=True)
            self.gate_dim = hidden_dim
        else:
            self.fc1 = nn.Identity()
            self.act = nn.Identity()
            self.gate_dim = self.feat_dim

        # 3. Output Layer
        self.fc2 = nn.Linear(self.gate_dim, num_classes, bias=bias_dense)

        # 4. Gating Masks (Always applied to `gate_dim`)
        self.register_buffer(
            "gate_masks",
            self._make_binary_masks(num_tasks, self.gate_dim, gate_frac)
        )

    def set_gating(self, on: bool = True):
        self.use_gating = on

    @staticmethod
    def _make_binary_masks(n_masks: int, dim: int, frac: float) -> torch.Tensor:
        k = max(1, int(dim * frac))
        masks = torch.zeros(n_masks, dim)
        for i in range(n_masks):
            idx = torch.randperm(dim)[:k]
            masks[i, idx] = 1.0
        return masks

    def forward(self, x: torch.Tensor, gate_ids: torch.Tensor | None = None):
        """
        gate_ids: LongTensor [B] with values in {0..num_tasks-1}
        """
        f = self.features(x)  # [B, 3136]

        # h is the representation strictly before the final output layer
        if self.use2dense:
            h = self.act(self.fc1(f))  # [B, hidden_dim]
        else:
            h = f  # [B, 3136]

        # Apply per-sample gating in the pre-output space
        if self.use_gating and gate_ids is not None:
            m = self.gate_masks[gate_ids]  # [B, gate_dim]
            h = h * m

        logits = self.fc2(h)  # [B, num_classes]
        return logits

    @torch.no_grad()
    def hidden(self, x: torch.Tensor):
        # Expose pre-gated representation for inference/analysis
        f = self.features(x)
        if self.use2dense:
            return self.act(self.fc1(f))
        return f

class ClassGatedSmallCNN(nn.Module):
    def __init__(
        self,
        n_channels=1,
        num_classes=10,
        gate_frac=0.2,
        hidden_dim=1024,      # NEW bottleneck dim
        bias_conv=True,
        bias_dense=True,
        use_gating: bool = True,
        use2dense: bool = True
    ):
        super().__init__()
        assert 0.0 < gate_frac <= 1.0
        self.num_classes = num_classes
        self.gate_frac = gate_frac
        self.use_gating = use_gating
        self.use2dense = use2dense

        # 1. Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, out_channels=32, kernel_size=3, padding=1, bias=bias_conv),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 28 -> 14

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=bias_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),           # 14 -> 7

            nn.Flatten(),
        )

        self.feat_dim = 64 * 7 * 7  # 3136
        self.hidden_dim = hidden_dim

        # 2. Determine Pre-Output Dimension and Setup Hidden Layer
        if self.use2dense:
            self.fc1 = nn.Linear(self.feat_dim, hidden_dim, bias=bias_dense)
            self.act = nn.ReLU(inplace=True)
            self.gate_dim = hidden_dim
        else:
            self.fc1 = nn.Identity()
            self.act = nn.Identity()
            self.gate_dim = self.feat_dim

        # 3. Output layer
        self.fc2 = nn.Linear(self.gate_dim, num_classes, bias=bias_dense)

        # 4. Gating masks now live in pre-output space: [num_classes, gate_dim]
        self.register_buffer(
            name="gate_masks",
            tensor=self._make_binary_masks(num_classes, self.gate_dim, gate_frac)
        )

    def set_gating(self, on: bool = True):
        self.use_gating = on

    @staticmethod
    def _make_binary_masks(n_masks: int, dim: int, frac: float) -> torch.Tensor:
        k = max(1, int(dim * frac))
        masks = torch.zeros(n_masks, dim)
        for i in range(n_masks):
            idx = torch.randperm(dim)[:k]
            masks[i, idx] = 1.0
        return masks

    def forward(self, x: torch.Tensor, gate_ids: torch.Tensor | None = None):
        f = self.features(x)              # [B, 3136]

        if self.use2dense:
            h = self.act(self.fc1(f))     # [B, hidden_dim]
        else:
            h = f                         # [B, 3136]

        if self.use_gating and gate_ids is not None:
            m = self.gate_masks[gate_ids] # [B, gate_dim]
            h = h * m

        logits = self.fc2(h)              # [B, num_classes]
        return logits

    @torch.no_grad()
    def hidden(self, x: torch.Tensor):
        f = self.features(x)
        if self.use2dense:
            return self.act(self.fc1(f))
        return f

# test phase
@torch.no_grad()
def infer_logits_optionA(model, xb):
    if not getattr(model, "use_gating", False):
        logits = model(xb, gate_ids=None)
        return logits, None

    h = model.hidden(xb)
    logits_all = []
    scores = []

    n_gates = getattr(model, "num_tasks", getattr(model, "num_classes", 1))

    for g in range(n_gates):
        hg = h * model.gate_masks[g].view(1, -1)  # [B, hidden_dim]
        lg = model.fc2(hg)  # [B, output_dim]

        logits_all.append(lg)

        scores.append(lg.softmax(dim=-1).max(dim=-1).values)  # [B]

    scores = torch.stack(scores, dim=0)  # [n_gates, B]

    g_hat = scores.argmax(dim=0)

    logits_all = torch.stack(logits_all, dim=0)  # [n_gates, B, output_dim]

    logits = logits_all[g_hat, torch.arange(xb.size(0), device=xb.device), :]  # [B, output_dim]

    return logits, g_hat

@torch.no_grad()
def infer_pred_optionA_selfclass(model, x: torch.Tensor):
    if not getattr(model, "use_gating", False):
        logits = model(x, gate_ids=None)
        return logits.argmax(dim=-1)

    h = model.hidden(x)  # [B, hidden_dim]
    scores = []

    for c in range(model.num_classes):
        hc = h * model.gate_masks[c].view(1, -1)   # [B, hidden_dim]
        lc = model.fc2(hc)                         # [B, C]
        scores.append(lc[:, c])                    # [B] self-class logit

    scores = torch.stack(scores, dim=0)            # [C, B]
    c_hat = scores.argmax(dim=0)                   # [B]
    return c_hat

@torch.no_grad()
def infer_logits_optionB(model, x: torch.Tensor, reduce="logsumexp", tau=1.0):
    if not getattr(model, "use_gating", False):
        logits = model(x, gate_ids=None)
        return logits, logits.argmax(dim=-1)

    h = model.hidden(x)  # [B, hidden_dim]
    logits_all = []

    for c in range(model.num_classes):
        hc = h * model.gate_masks[c].view(1, -1)   # [B, hidden_dim]
        logits_all.append(model.fc2(hc))           # [B, C]

    logits_all = torch.stack(logits_all, dim=0)    # [C, B, C]

    if reduce == "max":
        logits = logits_all.max(dim=0).values      # [B, C]
    elif reduce == "logsumexp":
        logits = torch.logsumexp(logits_all / tau, dim=0) * tau
    else:
        raise ValueError("reduce must be 'max' or 'logsumexp'")

    pred = logits.argmax(dim=-1)
    return logits, pred

@torch.no_grad()
def infer_logits_entropy_gate(model, x: torch.Tensor, tau: float = 1.0):
    if not getattr(model, "use_gating", False):
        logits = model(x, gate_ids=None)
        return logits, None

    h = model.hidden(x)  # [B, hidden_dim]
    logits_all = []
    entropies = []

    for g in range(model.num_classes):
        hg = h * model.gate_masks[g].view(1, -1)   # [B, hidden_dim]
        zg = model.fc2(hg)                         # [B, C]
        logits_all.append(zg)

        logp = torch.log_softmax(zg / tau, dim=-1)  # [B, C]
        p = logp.exp()
        Hg = -(p * logp).sum(dim=-1)                # [B]
        entropies.append(Hg)

    entropies = torch.stack(entropies, dim=0)       # [G, B]
    g_hat = entropies.argmin(dim=0)                 # [B]  (min entropy)

    logits_all = torch.stack(logits_all, dim=0)     # [G, B, C]
    logits = logits_all[g_hat, torch.arange(x.size(0), device=x.device), :]  # [B, C]
    return logits, g_hat

def snapshot_params(model):
    """Copy current parameters (theta*)."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def _num_classes_from_model_output(model, xb, yb=None):
    if getattr(model, "use_gating", False) and (yb is not None):
        logits = model(xb, gate_ids=yb)
    else:
        logits = model(xb)
    return logits.shape[1]

def ewc_penalty(model, fisher, theta_star):
    loss = torch.zeros((), device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if (not p.requires_grad) or (n not in fisher):
            continue
        loss = loss + (fisher[n] * (p - theta_star[n]).pow(2)).sum()
    return 0.5 * loss

def mse_loss_sum(logits, y_onehot):
    return (0.5 * (logits - y_onehot) ** 2).sum(dim=1).sum() # mean()

def estimate_fisher_diag(
    model, X, y_target, device,
    criterion="crossentropy",   # or "mse"
    fisher_n_samples=None,      # set to int to subsample
    fisher_eps: float = 0.0,
):
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}

    n_total = X.shape[0]
    if fisher_n_samples is not None and fisher_n_samples < n_total:
        idx = torch.randperm(n_total)[:fisher_n_samples]
        X = X[idx]
        y_target = y_target[idx]

    n_used = X.shape[0]
    batch_size = 1
    
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_target = torch.as_tensor(y_target, dtype=torch.long, device=device)

    for i in range(0, n_used, batch_size):
        xb = X[i:i+batch_size]
        yb = y_target[i:i+batch_size]

        # forward (oracle gate if gating model)
        if getattr(model, "use_gating", False):
            logits = model(xb, gate_ids=yb)
        else:
            logits = model(xb)

        if criterion == "crossentropy":
            loss = F.cross_entropy(logits, yb, reduction="mean")
        elif criterion == "mse":
            yb_oh = F.one_hot(yb, num_classes=logits.shape[1]).to(dtype=logits.dtype)
            loss = mse_loss_sum(logits, yb_oh)
        else:
            raise ValueError("criterion must be 'crossentropy' or 'mse'")

        model.zero_grad(set_to_none=True)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is None or (n not in fisher):
                continue
            fisher[n] += (p.grad.detach() ** 2)
    # Average across samples (empirical expectation of squared gradient)
    for n in fisher:
        fisher[n] /= float(n_used)
        if fisher_eps > 0.0:
            fisher[n] = fisher[n] + fisher_eps
    return fisher