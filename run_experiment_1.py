"""
SAND on continual-learning benchmarks — Experiment 1.

The decisive go/no-go test for SAND (Surprise-Adaptive Neural Directions).

SAND is built on magnitude-direction decoupling: each weight matrix is
factorized into a fixed-norm DIRECTION on a hypersphere plus learnable per-row /
per-column magnitude GAINS. The direction is the fragile, shared 'what the
feature does' part (moving it causes forgetting); the gains are the cheap,
per-task 'how loud' part (moving them barely causes forgetting). So the recipe
is: protect the direction, adapt the gains freely per task.

The open question SAND answers: *how much* should the direction be allowed to
drift for a new task? Too little drift underfits new tasks; too much destroys
old ones. SAND sets the direction's learning rate PER TASK using a gradient
CONFLICT probe (A-GEM / GPM / PCGrad style): it measures the cosine similarity
between the direction gradient on the new task and on a tiny replay memory of
old tasks. Aligned gradients -> drifting is safe -> drift freely. Opposite
gradients -> drifting would break old tasks -> protect. (Need-based probes --
accuracy, raw gradient, loss-drop -- were tried and fail to read task novelty;
only a conflict-based probe works.)

This script compares:
  - naive            : keep training everything (catastrophic-forgetting baseline)
  - fixed frac=...   : constant direction-LR fraction for every task (ablation fan,
                        no probe) -- the 'is the probe necessary?' baseline
  - SAND             : our method (cosine-conflict-adaptive per-task direction LR)
  - joint            : upper bound, all data at once (the ceiling)

Task streams (--stream):
  - varied  : MIXED drift-need. Task 0 builds a broad 5-way backbone {0..4};
              later tasks alternate NOVEL pairs (digits the backbone has never
              seen) and FAMILIAR pairs (digits already in the backbone). This is
              the setting per-task adaptivity is designed for -- a single fixed
              fraction must compromise between novel and familiar tasks. DEFAULT.
  - split   : disjoint digit pairs, all equally novel.
  - permuted: pixel permutations, all equally novel.

Fairness:
  - Equal compute: SAND's cosine probe is a few backward passes with no kept
    optimizer steps, so the direction trains for the FULL epochs == fixed
    fractions.
  - The fair bar is avg/median fixed fraction (a blindly-picked constant);
    'best fixed' is reported as a hindsight-tuned CEILING, not the bar.
  - Multi-seed (--seeds 0,1,2) gives mean±std and a significance-aware verdict.
"""

import argparse
import sys
import time
import statistics as st
from dataclasses import dataclass, field

from tqdm.auto import tqdm

# True when stdout is a real terminal (so tqdm's animated bar works nicely).
# When piped/redirected (e.g. `| tee run.log`) we fall back to clean periodic
# text lines so the saved log stays readable and easy to paste.
_IS_TTY = sys.stdout.isatty()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from md_linear import MDMLP


def make_loader(x, y, batch_size, shuffle, num_workers=4, pin_memory=True):
    """Consistent DataLoader settings for good GPU feeding."""
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=(num_workers > 0))


# --------------------------------------------------------------------------- #
#  Data                                                                       #
# --------------------------------------------------------------------------- #
def load_mnist(data_dir: str = "./data") -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor, torch.Tensor]:
    """Download MNIST and return (x_train, y_train, x_test, y_test) tensors."""
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Lambda(lambda t: t.flatten())])  # (1,28,28)->(784,)
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    x_tr = torch.stack([train[i][0] for i in range(len(train))])
    y_tr = torch.tensor([train[i][1] for i in range(len(train))])
    x_te = torch.stack([test[i][0] for i in range(len(test))])
    y_te = torch.tensor([test[i][1] for i in range(len(test))])
    return x_tr, y_tr, x_te, y_te


@dataclass
class Task:
    """One task in a continual-learning stream.

    classes=None  -> standard 10-way classification (permuted-MNIST).
    classes=[a,b] -> 2-way classification over digit classes a,b (split-MNIST).
    shared_head   -> if True, every task uses the SAME first len(classes) output
                     neurons (so later tasks overwrite earlier tasks' heads ->
                     output interference -> forgetting returns). If False, each
                     task uses its own disjoint output neurons (no head
                     interference -> little forgetting, as in sliced split-MNIST).
    """
    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor       # ORIGINAL digit labels (not remapped)
    test_loader: DataLoader
    classes: list[int] | None
    shared_head: bool = False


def make_permutation(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)


def apply_perm(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return x[:, perm]


def build_stream(x_tr, y_tr, x_te, y_te, args) -> tuple[list[Task], torch.Tensor, torch.Tensor]:
    """Build the task stream + the joint-training data (all tasks at once).

    Returns (tasks, joint_x, joint_y) where joint_* are for the joint upper
    bound: for split it's the full MNIST (all 10 digits, 10-way); for permuted
    it's every permutation concatenated.
    """
    if args.stream == "permuted":
        perms = [torch.arange(784)] + [make_permutation(784, seed=100 + t)
                                       for t in range(1, args.n_tasks)]
        tasks = []
        for t, perm in enumerate(perms):
            tasks.append(Task(
                name=f"perm-{t}",
                train_x=apply_perm(x_tr, perm),
                train_y=y_tr,
                test_loader=make_loader(apply_perm(x_te, perm), y_te, 512, shuffle=False),
                classes=None,
                shared_head=False,
            ))
        # joint: all permutations concatenated, 10-way
        joint_x = torch.cat([apply_perm(x_tr, p) for p in perms], 0)
        joint_y = torch.cat([y_tr for _ in perms], 0)
        return tasks, joint_x, joint_y

    elif args.stream == "split":
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)][:args.n_tasks]
        if len(pairs) < args.n_tasks:
            raise SystemExit(f"split stream has at most 5 tasks (10 digits / 2), "
                             f"got n_tasks={args.n_tasks}")
        tasks = []
        for (a, b) in pairs:
            mtr = (y_tr == a) | (y_tr == b)
            mte = (y_te == a) | (y_te == b)
            tasks.append(Task(
                name=f"split{{{a},{b}}}",
                train_x=x_tr[mtr],
                train_y=y_tr[mtr],
                test_loader=make_loader(x_te[mte], y_te[mte], 512, shuffle=False),
                classes=[a, b],
                shared_head=args.shared_head,
            ))
        # joint: full MNIST, 10-way (all digits at once)
        return tasks, x_tr, y_tr

    elif args.stream == "varied":
        # MIXED-drift-need stream: task 0 builds a broad backbone over half the
        # digits (5-way); later tasks alternate NOVEL pairs (digits the backbone
        # has never seen -> direction wants to move a lot -> high surprise) and
        # FAMILIAR pairs (digits already in the backbone -> direction is happy ->
        # low surprise). This is the setting per-task adaptivity is designed for:
        # a single fixed direction-LR fraction must compromise between the novel
        # and familiar tasks, while SAND can drift a lot on the novel ones and
        # barely on the familiar ones.
        # task specs: (classes, kind). kind is just for naming.
        spec = [
            ([0, 1, 2, 3, 4], "broad"),   # task 0: 5-way backbone
            ([5, 6],        "novel"),     # task 1: brand-new digits
            ([0, 1],        "familiar"),  # task 2: already in backbone
            ([7, 8],        "novel"),     # task 3: brand-new
            ([3, 4],        "familiar"),  # task 4: already in backbone
        ][:args.n_tasks]
        if len(spec) < args.n_tasks:
            raise SystemExit(f"varied stream has at most 5 tasks, got n_tasks={args.n_tasks}")
        tasks = []
        for (cls, kind) in spec:
            mtr = torch.zeros(len(y_tr), dtype=torch.bool)
            mte = torch.zeros(len(y_te), dtype=torch.bool)
            for c in cls:
                mtr |= (y_tr == c); mte |= (y_te == c)
            tasks.append(Task(
                name=f"varied[{kind}{cls}]",
                train_x=x_tr[mtr], train_y=y_tr[mtr],
                test_loader=make_loader(x_te[mte], y_te[mte], 512, shuffle=False),
                classes=list(cls),
                shared_head=args.shared_head,
            ))
        # joint: full MNIST, 10-way
        return tasks, x_tr, y_tr

    else:
        raise SystemExit(f"unknown --stream {args.stream!r} "
                         f"(use 'split', 'permuted', or 'varied')")


# --------------------------------------------------------------------------- #
#  Model / training helpers                                                   #
# --------------------------------------------------------------------------- #
def make_model(device, hidden: int = 256, n_outputs: int = 10) -> MDMLP:
    # Plain MLP, no CNN. Output size is n_outputs; split tasks slice/remap.
    # For a shared 2-output head (forgetting-prone split), n_outputs=2.
    return MDMLP([784, hidden, hidden, n_outputs]).to(device)


def task_loss(logits: torch.Tensor, yb: torch.Tensor, classes: list[int] | None,
              shared_head: bool = False):
    """Cross-entropy for a task.
    classes=None        -> n_outputs-way over all logits.
    classes=[a,b,...], shared_head=False -> slice logits[:, classes], remap to 0..k-1
                          (each task owns disjoint outputs -> no head interference).
    classes=[a,b,...], shared_head=True  -> use logits[:, :k] (SAME first k outputs
                          for every task -> head overwritten -> forgetting).
    """
    if classes is None:
        return F.cross_entropy(logits, yb)
    yb_remap = torch.empty_like(yb)
    for i, c in enumerate(classes):
        yb_remap[yb == c] = i
    out_idx = list(range(len(classes))) if shared_head else classes
    return F.cross_entropy(logits[:, out_idx], yb_remap)


def train_one_task(model, loader, device, lr, epochs, classes,
                   freeze_direction=False, gain_lr=None, log_prefix="      ",
                   shared_head: bool = False, per_layer_dir_lrs: list[float] | None = None,
                   epoch_hook=None):
    """Train `model` for one task.

    - freeze_direction=True: only gains (+ bias) are optimized; direction frozen.
    - gain_lr is not None: separate LRs -- `gain_lr` for gains/bias, `lr` for the
      direction params (the paper's "separate learning rates" knob).
    - per_layer_dir_lrs is not None: each direction layer gets its OWN lr from the
      list (one per MDLinear, in order). This is SAND's per-layer drift: a layer
      whose gradient probe said 'move a lot' gets a high lr, a layer that's happy
      gets a low lr. gains still use gain_lr.
    - epoch_hook: optional callable(epoch_idx, train_loss) called AFTER each epoch,
      used by the learning-curve recording to capture per-epoch accuracy.
    Otherwise everything is optimized at `lr`.
    """
    gain_params = model.trainable_gain_params()
    gain_ids = {id(p) for p in gain_params}
    dir_params = [p for p in model.parameters()
                  if p.requires_grad and id(p) not in gain_ids]

    if freeze_direction:
        opt = torch.optim.Adam(gain_params, lr=lr)
        mode = "gains-only (direction FROZEN)"
    elif per_layer_dir_lrs is not None and len(dir_params) > 0:
        # one param group per direction layer, each with its own lr
        groups = [{"params": [p], "lr": plr}
                  for p, plr in zip(dir_params, per_layer_dir_lrs) if plr > 0]
        groups.append({"params": gain_params, "lr": gain_lr if gain_lr is not None else lr})
        opt = torch.optim.Adam(groups)
        mode = (f"per-layer dir lr=[{','.join(f'{x:.1e}' for x in per_layer_dir_lrs)}] "
                f"+ gains lr={gain_lr:.2e}")
    elif gain_lr is not None and len(dir_params) > 0:
        opt = torch.optim.Adam([
            {"params": dir_params, "lr": lr},
            {"params": gain_params, "lr": gain_lr},
        ])
        mode = f"direction lr={lr:.2e} + gains lr={gain_lr:.2e}"
    else:
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        mode = f"full (direction + gains) lr={lr:.2e}"
    n_params = sum(p.numel() for group in opt.param_groups for p in group["params"])
    print(f"{log_prefix}train: {mode} | trainable params: {n_params:,} | epochs={epochs}",
          flush=True)

    n_batches = len(loader)
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        running, nbatches = 0.0, 0
        if _IS_TTY:
            pbar = tqdm(loader, desc=f"{log_prefix}epoch {ep+1}/{epochs}",
                        leave=False, dynamic_ncols=True,
                        mininterval=0.5, smoothing=0.1)
        else:
            pbar = loader
        log_every = max(1, n_batches // 10)
        for xb, yb in pbar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = task_loss(logits, yb, classes, shared_head)
            loss.backward()
            opt.step()
            running += loss.item(); nbatches += 1
            if _IS_TTY:
                pbar.set_postfix(loss=f"{running/nbatches:.3f}")
            elif nbatches % log_every == 0 and nbatches < n_batches:
                dt = time.time() - t0
                its = nbatches / dt if dt > 0 else 0.0
                print(f"{log_prefix}epoch {ep+1}/{epochs}  [{nbatches}/{n_batches}]  "
                      f"loss={running/nbatches:.3f}  {its:.0f} it/s", flush=True)
        if _IS_TTY:
            pbar.close()
        print(f"{log_prefix}epoch {ep+1}/{epochs} done  loss={running/nbatches:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if epoch_hook is not None:
            epoch_hook(ep, running / nbatches)
    return opt


@torch.no_grad()
def evaluate(model, loader, device, classes: list[int] | None,
              shared_head: bool = False) -> float:
    """Accuracy on a loader using the model's CURRENT gains. If classes is given,
    predict among those classes only (and map back to original digit labels)."""
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if classes is None:
            pred = logits.argmax(1)
        else:
            out_idx = list(range(len(classes))) if shared_head else classes
            pred_local = logits[:, out_idx].argmax(1)
            pred = torch.tensor([classes[p.item()] for p in pred_local], device=yb.device)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total


@torch.no_grad()
def evaluate_all_tasks(model, tasks: list[Task], device) -> list[float]:
    """Accuracy on every task, switching in each task's gains before eval."""
    accs = []
    for t, task in enumerate(tasks):
        model.set_task(t)
        accs.append(evaluate(model, task.test_loader, device, task.classes,
                              task.shared_head))
    return accs


# --------------------------------------------------------------------------- #
#  Conditions                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    name: str
    color: str = "#9467bd"
    # acc_matrix[t] = list of accuracies on tasks 0..t after training task t
    acc_matrix: list[list[float]] = field(default_factory=list)
    # per-task surprises (only filled by SAND), for reporting/plotting
    surprises: list[float] = field(default_factory=list)
    # total training epochs spent per task (for compute-fairness auditing)
    total_epochs_per_task: int = 0
    # per-epoch learning curves for the 'learning over time' plot:
    #   curves[task_idx] = list of (epoch_idx, new_task_acc, old_tasks_acc)
    # where new_task_acc is accuracy on the task being trained and
    # old_tasks_acc is the mean accuracy on all previously-learned tasks,
    # both evaluated AFTER that epoch. (task 0 has no old tasks -> old=NaN.)
    curves: list[list[tuple[int, float, float]]] = field(default_factory=list)

    def final_avg_accuracy(self) -> float:
        last = self.acc_matrix[-1]
        return sum(last) / len(last)

    def avg_forgetting(self) -> float:
        if len(self.acc_matrix) <= 1:
            return 0.0
        final = self.acc_matrix[-1]
        forgets = []
        for t in range(len(final) - 1):
            peak = max(row[t] for row in self.acc_matrix if t < len(row))
            forgets.append(max(0.0, peak - final[t]))
        return sum(forgets) / len(forgets)


def _make_curve_hook(model, new_task, old_tasks, res, t: int, device):
    """Build an epoch_hook that records, after each training epoch of task t:
    (epoch_idx, new_task_acc, old_tasks_mean_acc) into res.curves[t].
    new_task_acc = accuracy on the task being trained; old_tasks_mean_acc =
    mean accuracy on all previously-learned tasks (NaN if none). This drives
    the 'learning over time' plot: how fast each method learns the new task
    vs how much it forgets the old ones, per epoch."""
    res.curves.append([])
    new_loader = make_loader(new_task.test_loader.dataset.tensors[0],
                             new_task.test_loader.dataset.tensors[1], 512, shuffle=False)
    old_acc_loaders = [(ot.test_loader.dataset.tensors[0],
                        ot.test_loader.dataset.tensors[1], ot.classes, ot.shared_head)
                       for ot in old_tasks]
    def hook(ep, train_loss):
        new_acc = evaluate(model, new_loader, device, new_task.classes, new_task.shared_head)
        if old_acc_loaders:
            # snapshot the new-task gains, switch to each old task's gains, eval,
            # then restore -- so 'old tasks' accuracy uses each old task's stored
            # gains (as in evaluate_all_tasks), not the current task's gains.
            saved = t  # current task index is t
            olds = []
            for oi, (ox, oy, ocl, osh) in enumerate(old_acc_loaders):
                model.set_task(oi)
                ol = make_loader(ox, oy, 512, shuffle=False)
                olds.append(evaluate(model, ol, device, ocl, osh))
            model.set_task(saved)
            old_acc = float(np_mean(olds)) if olds else float("nan")
        else:
            old_acc = float("nan")
        res.curves[t].append((ep, new_acc, old_acc))
    return hook


def np_mean(xs):
    import numpy as _np
    return float(_np.mean(xs))


def run_naive(tasks, device, args) -> Result:
    """Baseline: keep training everything on every task."""
    model = make_model(device, hidden=args.hidden, n_outputs=args.n_outputs)
    res = Result("naive", color="#d62728", total_epochs_per_task=args.epochs)
    for t, task in enumerate(tasks):
        loader = make_loader(task.train_x, task.train_y, args.batch_size, shuffle=True)
        print(f"   --- naive: task {t}/{len(tasks)-1} ({task.name}) ---", flush=True)
        hook = _make_curve_hook(model, task, tasks[:t], res, t, device)
        train_one_task(model, loader, device, args.lr, args.epochs, task.classes,
                       log_prefix="      ", shared_head=task.shared_head, epoch_hook=hook)
        accs = evaluate_all_tasks(model, tasks[:t + 1], device)
        res.acc_matrix.append(accs)
        print(f"   [naive] after task {t}: " +
              ", ".join(f"T{s}={a:.3f}" for s, a in enumerate(accs)), flush=True)
    return res


def run_fixed_frac(tasks, device, args, frac: float) -> Result:
    """Ablation baseline: CONSTANT direction-LR fraction for every task (no probe).

    frac=0  -> direction fully frozen (gains only) every task.
    frac>0  -> direction LR = lr*frac, gains LR = lr, every task.

    Per-task gains are stored like SAND, so old-task accuracy is recoverable up
    to whatever the direction drifted to. This is the 'is the probe necessary?'
    baseline: if the best fixed frac beats SAND, the surprise mechanism is
    unnecessary.
    """
    model = make_model(device, hidden=args.hidden, n_outputs=args.n_outputs)
    res = Result(f"fixed frac={frac}", color="#1f77b4", total_epochs_per_task=args.epochs)
    for t, task in enumerate(tasks):
        loader = make_loader(task.train_x, task.train_y, args.batch_size, shuffle=True)
        print(f"   --- fixed frac={frac}: task {t}/{len(tasks)-1} ({task.name}) ---", flush=True)
        hook = _make_curve_hook(model, task, tasks[:t], res, t, device)
        if t == 0:
            train_one_task(model, loader, device, args.lr, args.epochs, task.classes,
                           log_prefix="      ", shared_head=task.shared_head, epoch_hook=hook)
        else:
            model.new_task()
            if frac == 0:
                model.freeze_direction()
                train_one_task(model, loader, device, args.lr, args.epochs, task.classes,
                               freeze_direction=True, log_prefix="      ",
                               shared_head=task.shared_head, epoch_hook=hook)
            else:
                train_one_task(model, loader, device, args.lr * frac, args.epochs,
                               task.classes, freeze_direction=False, gain_lr=args.lr,
                               log_prefix="      ", shared_head=task.shared_head, epoch_hook=hook)
        accs = evaluate_all_tasks(model, tasks[:t + 1], device)
        res.acc_matrix.append(accs)
        print(f"   [fixed {frac}] after task {t}: " +
              ", ".join(f"T{s}={a:.3f}" for s, a in enumerate(accs)), flush=True)
    return res


def run_sand(tasks, device, args) -> Result:
    """SAND: Surprise-Adaptive Neural Directions (our method).

    Direction frozen by default; thaws per task in proportion to a gradient
    CONFLICT probe (cosine similarity of new vs. old-task direction gradients,
    A-GEM-style, using a tiny replay memory). See module docstring.
    """
    model = make_model(device, hidden=args.hidden, n_outputs=args.n_outputs)
    res = Result("SAND", color="#2ca02c", total_epochs_per_task=args.epochs)
    # Tiny replay memory of PAST tasks (A-GEM-style). Needed to measure novelty
    # *relative to what we already know* -- without a reference, 'is this task
    # surprising?' is unanswerable (our three no-memory probes all failed for
    # exactly this reason). mem_size per task is tiny (default 256), standard in
    # the continual-learning literature (A-GEM, GEM, ER).
    memory_x: list[torch.Tensor] = []
    memory_y: list[torch.Tensor] = []
    for t, task in enumerate(tasks):
        full_loader = make_loader(task.train_x, task.train_y, args.batch_size, shuffle=True)

        print(f"   --- SAND: task {t}/{len(tasks)-1} ({task.name}) ---", flush=True)
        hook = _make_curve_hook(model, task, tasks[:t], res, t, device)
        if t == 0:
            train_one_task(model, full_loader, device, args.lr, args.epochs, task.classes,
                           log_prefix="      ", shared_head=task.shared_head, epoch_hook=hook)
        else:
            model.new_task()                       # save old gains, reset live gains
            # ---- COSINE-CONFLICT PROBE (A-GEM / GPM / PCGrad style):
            #      Compare the direction gradient on the NEW task vs. on an
            #      OLD-task memory. The ANGLE between them tells us whether
            #      drifting for the new task would HURT old tasks:
            #        cos ~ +1  (aligned)   -> drifting helps old too -> safe, drift
            #        cos ~  0  (orthogonal)-> drifting doesn't touch old -> safe, drift
            #        cos ~ -1  (opposite)  -> drifting breaks old     -> PROTECT, don't drift
            #      This measures the actual stability/plasticity trade-off per
            #      task, which is what 'surprise' was always trying to capture.
            #      Compute the direction gradient on new and old, per layer, then
            #      cosine per layer -> a per-layer safety factor -> per-layer
            #      direction LR. COMPUTE-FREE (a few backward passes, no optim
            #      steps kept) so the direction trains for the FULL epochs.
            model.unfreeze_direction()
            # ---- direction gradient on OLD tasks: sum the direction gradient of
            #      EACH past task's loss (each with its own stored gains + classes),
            #      so the old gradient reflects 'keep ALL old tasks happy'. Using a
            #      single 10-way loss would need 10 outputs; per-task is cleaner and
            #      matches how the model was trained.
            model.zero_grad(set_to_none=True)
            for ot in range(t):
                old_task = tasks[ot]
                # sample a few batches from this old task's stored memory
                ox = memory_x[ot].to(device); oy = memory_y[ot].to(device)
                p = torch.randperm(len(ox))[:args.probe_batches * args.batch_size]
                ox, oy = ox[p], oy[p]
                model.set_task(ot)  # that old task's gains
                for i in range(0, len(ox), args.batch_size):
                    xb = ox[i:i+args.batch_size]; yb = oy[i:i+args.batch_size]
                    logits = model(xb)
                    loss = task_loss(logits, yb, old_task.classes, old_task.shared_head)
                    loss.backward()
            grad_old = model.per_layer_direction_grad_flat()
            # ---- direction gradient on NEW task (with new task's reset gains) ----
            model.set_task(t)
            piter = iter(full_loader)
            model.zero_grad(set_to_none=True)
            for _ in range(args.probe_batches):
                xb, yb = next(piter)
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = task_loss(logits, yb, task.classes, task.shared_head)
                loss.backward()
            grad_new = model.per_layer_direction_grad_flat()
            model.zero_grad(set_to_none=True)
            # per-layer cosine similarity and safety factor
            cosines, safeties, per_layer_dir_lrs = [], [], []
            for go, gn in zip(grad_old, grad_new):
                dot = float(torch.dot(go, gn).item())
                no = float(go.norm().item()); nn = float(gn.norm().item())
                cos = dot / (no * nn + 1e-8)
                cosines.append(cos)
                # safety in [0,1]: 1 when aligned (cos=+1), 0.5 when orthogonal,
                # 0 when opposite (cos=-1). Drift freely when safe, protect when not.
                safety = (cos + 1.0) / 2.0
                safety = max(0.0, min(1.0, safety ** args.surprise_temp))
                safeties.append(safety)
                per_layer_dir_lrs.append(args.lr * args.max_frac * safety)
            avg_surprise = 1.0 - (sum(safeties) / len(safeties))  # higher = more conflict
            res.surprises.append(avg_surprise)
            print(f"      [probe-cos] per-layer cos(new,old) = "
                  f"[{', '.join(f'{c:+.2f}' for c in cosines)}]", flush=True)
            print(f"      [probe-cos] per-layer safety(drift) = "
                  f"[{', '.join(f'{s:.2f}' for s in safeties)}]  "
                  f"(avg_surprise={avg_surprise:.3f})", flush=True)
            print(f"      [probe-cos] per-layer direction LR  = "
                  f"[{', '.join(f'{x:.2e}' for x in per_layer_dir_lrs)}]  "
                  f"(gains stay at LR={args.lr:.2e})", flush=True)
            # ---- REAL training: per-layer direction LRs, FULL budget (probe was free) ----
            train_one_task(model, full_loader, device, args.lr, args.epochs, task.classes,
                           freeze_direction=False, gain_lr=args.lr, log_prefix="      ",
                           shared_head=task.shared_head,
                           per_layer_dir_lrs=per_layer_dir_lrs, epoch_hook=hook)
        accs = evaluate_all_tasks(model, tasks[:t + 1], device)
        res.acc_matrix.append(accs)
        print(f"   [SAND] after task {t}: " +
              ", ".join(f"T{s}={a:.3f}" for s, a in enumerate(accs)), flush=True)
        # ---- add this task to the replay memory (tiny, per task) ----
        n = min(args.mem_size, len(task.train_x))
        perm = torch.randperm(len(task.train_x))[:n]
        memory_x.append(task.train_x[perm].clone())
        memory_y.append(task.train_y[perm].clone())
    return res


def run_joint(tasks, joint_x, joint_y, device, args) -> Result:
    """Upper bound: train once on ALL tasks' data mixed together.

    The joint model is a normal 10-way digit classifier (10 outputs), trained
    on all data at once -- 'if you had everything simultaneously'. It is NOT
    shared-head: it has all 10 outputs and we slice the 2 each task cares about
    at eval time. So we override n_outputs=10 and evaluate with
    shared_head=False regardless of the per-task setting.
    """
    model = make_model(device, hidden=args.hidden, n_outputs=10)
    res = Result("joint", color="#7f7f7f", total_epochs_per_task=args.epochs * len(tasks))
    loader = make_loader(joint_x, joint_y, args.batch_size, shuffle=True)
    print(f"   --- joint: training once on all data ({len(joint_x)} samples, 10-way) ---", flush=True)
    train_one_task(model, loader, device, args.lr, args.epochs * len(tasks), None,
                   log_prefix="      ")
    accs = []
    for t, task in enumerate(tasks):
        # joint has all 10 outputs -> slice the real class indices (non-shared)
        accs.append(evaluate(model, task.test_loader, device, task.classes,
                              shared_head=False))
    res.acc_matrix.append(accs)
    print(f"   [joint] trained once on all tasks: " +
          ", ".join(f"T{s}={a:.3f}" for s, a in enumerate(accs)), flush=True)
    return res


# --------------------------------------------------------------------------- #
#  Summary + plot                                                             #
# --------------------------------------------------------------------------- #
def print_summary(results: list[Result]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        print(f"\n{r.name}  (epochs/task={r.total_epochs_per_task})")
        for t, row in enumerate(r.acc_matrix):
            print(f"  after task {t}: " +
                  ", ".join(f"T{s}={a:.3f}" for s, a in enumerate(row)))
        if len(r.acc_matrix) > 1:
            print(f"  final average accuracy over all tasks: {r.final_avg_accuracy():.3f}")
            print(f"  average forgetting:                     {r.avg_forgetting():.3f}")
        if r.surprises:
            print(f"  per-task surprises: " +
                  ", ".join(f"t{i+1}={s:.3f}" for i, s in enumerate(r.surprises)))
    print("=" * 72)

    # explicit go/no-go line for Experiment 1
    fixed = [r for r in results if r.name.startswith("fixed")]
    sand = next((r for r in results if r.name == "SAND"), None)
    if fixed and sand and len(sand.acc_matrix) > 1:
        best = max(fixed, key=lambda r: r.final_avg_accuracy())
        facc = [r.final_avg_accuracy() for r in fixed]
        avg_acc = sum(facc) / len(facc)
        med_acc = st.median(facc)
        avg_forget = sum(r.avg_forgetting() for r in fixed) / len(fixed)
        print("\n--- Experiment 1 go/no-go ---")
        # compute-fairness audit line
        budgets = {r.name: r.total_epochs_per_task for r in results}
        print("  COMPUTE BUDGET (epochs per task): " + ", ".join(
              f"{n}={b}" for n, b in budgets.items()))
        print("  (fairness: SAND's cosine probe is COMPUTE-FREE (a few backward "
              "passes, no kept optimizer steps), so the direction trains for the "
              "FULL epochs == fixed fractions. Equal compute by construction. "
              "SAND also keeps a tiny replay memory of --mem_size examples/past-task "
              "(A-GEM-style) to measure novelty against; this is standard and "
              "reported as a memory cost, not a compute cost.)")
        print("  (fairness note: 'best fixed' is the best of N hindsight-tuned runs;")
        print("   'avg/median fixed' is the honest no-lookahead baseline SAND must beat.)")
        print(f"  avg    fixed fraction:        -> acc={avg_acc:.3f}, forget={avg_forget:.3f}  "
              f"(fair bar: a blindly-picked constant)")
        print(f"  median fixed fraction:        -> acc={med_acc:.3f}  "
              f"(robust fair bar, ignores grid choice)")
        print(f"  best   fixed fraction: {best.name} -> acc={best.final_avg_accuracy():.3f}, "
              f"forget={best.avg_forgetting():.3f}  (hindsight ceiling)")
        print(f"  SAND                 :        -> acc={sand.final_avg_accuracy():.3f}, "
              f"forget={sand.avg_forgetting():.3f}  (online, untuned, per-task)")
        sand_acc = sand.final_avg_accuracy()
        # Primary verdict vs the FAIR bar (avg fixed). Best-fixed is reported as
        # context (a ceiling SAND isn't expected to beat -- it's hindsight-tuned).
        if sand_acc > avg_acc + 0.005:
            print("  VERDICT: SAND beats the fair (avg-fixed) baseline -> adaptivity helps. "
                  "GO.")
        elif abs(sand_acc - avg_acc) <= 0.005:
            print("  VERDICT: SAND ~ fair baseline -> probe not earning its keep vs a "
                  "blind constant. INVESTIGATE (check vs best-fixed for context).")
        else:
            print("  VERDICT: fair baseline beats SAND -> a blind constant is better than "
                  "the probe. NO-GO (fix the probe, the mapping, or the benchmark).")
        # context line vs hindsight ceiling
        if sand_acc >= best.final_avg_accuracy() - 0.005:
            print("  CONTEXT: SAND also matches the hindsight-tuned best fixed fraction "
                  "(strong -- online method tying a hindsight ceiling).")
        else:
            print(f"  CONTEXT: hindsight best fixed is {best.final_avg_accuracy()-sand_acc:.3f} "
                  f"above SAND (expected -- it is tuned with full lookahead).")
        if sand.surprises and (max(sand.surprises) - min(sand.surprises)) < 0.05:
            print("  WARNING: surprise values nearly uniform across tasks -> probe is "
                  "not discriminating; try a harder/mixed stream or different measure.")
        else:
            print("  PROBE: surprise values vary across tasks -> probe IS discriminating."
                  " Good.")
    print("=" * 72)


def plot_results(results: list[Result], out_path: str, stream: str) -> None:
    """Left: ONE line per approach -- average accuracy over all tasks seen so far,
    plotted vs 'after training task t' (so each approach is a single connected
    line, one data point per task). Right: final accuracy + forgetting bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5))

    n_tasks = max(len(r.acc_matrix) for r in results)

    # ---- LEFT: one line per approach. Each data point at step t is the
    #      AVERAGE accuracy over all tasks seen up to and including t
    #      (i.e. the running mean of acc_matrix[t]). One connected line per
    #      approach -> clean comparison of how each method's overall accuracy
    #      evolves as more tasks are learned. ---- #
    for r in results:
        # running mean accuracy over tasks seen so far, per training step
        run_avg = []
        for row in r.acc_matrix:
            run_avg.append(float(np.mean(row)))
        steps = np.arange(len(run_avg))
        is_fan = r.name.startswith("fixed")
        lw = 3.0 if r.name in ("SAND", "naive", "joint") else 1.3
        alpha = 0.25 if is_fan else 0.95
        axL.plot(steps, run_avg, marker="o", ms=6,
                 color=r.color, lw=lw, alpha=alpha, label=r.name)
    axL.set_xlabel("after training task t")
    axL.set_ylabel("average accuracy over tasks seen so far")
    axL.set_title(f"Average accuracy as training progresses ({stream})\n"
                  "faded fan = fixed-fraction ablation; one line per approach")
    axL.set_ylim(0, 1.0)
    axL.set_xticks(range(n_tasks))
    axL.grid(True, alpha=0.3)
    axL.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ---- RIGHT: final accuracy + forgetting bar chart ------------------- #
    names = [r.name for r in results]
    final_acc = [r.final_avg_accuracy() if len(r.acc_matrix) > 1 else r.acc_matrix[-1][0]
                 for r in results]
    forget = [r.avg_forgetting() for r in results]
    x = np.arange(len(names))
    w = 0.38
    bars1 = axR.bar(x - w / 2, final_acc, w, label="final avg accuracy \u2191",
                    color=[r.color for r in results], alpha=0.85)
    bars2 = axR.bar(x + w / 2, forget, w, label="avg forgetting \u2193",
                    color=[r.color for r in results], alpha=0.45,
                    hatch="//", edgecolor="black", linewidth=0.6)
    axR.set_xticks(x)
    axR.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    axR.set_ylim(0, 1.0)
    axR.set_ylabel("value")
    axR.set_title("Final summary: higher accuracy + lower forgetting = better\n"
                  "(dashed line = avg fixed fraction = the fair no-lookahead bar)")
    axR.grid(True, axis="y", alpha=0.3)
    # fair-bar reference line: average accuracy of the fixed-fraction ablation.
    # Label it INLINE next to the line (no legend box) so it doesn't cover bars.
    fixed_accs = [r.final_avg_accuracy() for r in results if r.name.startswith("fixed")]
    if fixed_accs:
        avg_fixed = sum(fixed_accs) / len(fixed_accs)
        axR.axhline(avg_fixed, color="#1f77b4", ls="--", lw=1.5, alpha=0.8)
        axR.text(len(names) - 0.5, avg_fixed + 0.012,
                 f"avg fixed = {avg_fixed:.3f}", color="#1f77b4",
                 fontsize=9, ha="right", va="bottom")
    # bar value labels
    for b, v in zip(bars1, final_acc):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    for b, v in zip(bars2, forget):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    # legend at the BOTTOM, outside the bars, so it never covers the joint bar
    axR.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=9, frameon=False)

    fig.suptitle(f"SAND \u2014 Experiment 1: {stream} continual learning "
                 f"(fixed-fraction ablation vs surprise-adaptive)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    print(f"\nPlot saved to: {out_path}", flush=True)

    # ---- Learning curves over time: per-epoch new-task acc and old-tasks acc,
    #      one subplot per task (tasks 1..n-1), methods as colored lines. This
    #      directly answers 'how fast does each method learn the new task, and
    #      how much does it forget the old ones, as training progresses?' ---- #
    n_tasks = max(len(r.acc_matrix) for r in results)
    n_curve_tasks = n_tasks - 1  # task 0 has no 'old tasks' to forget
    if n_curve_tasks <= 0 or all(not r.curves for r in results):
        return
    cfig, axes = plt.subplots(2, n_curve_tasks, figsize=(4.2 * n_curve_tasks, 7),
                              squeeze=False)
    for ti in range(1, n_tasks):  # task 0 is the backbone, skip
        col = ti - 1
        ax_new, ax_old = axes[0, col], axes[1, col]
        for r in results:
            if ti >= len(r.curves) or not r.curves[ti]:
                continue
            rows = r.curves[ti]
            eps = [row[0] for row in rows]
            new = [row[1] for row in rows]
            old = [row[2] for row in rows]
            is_fan = r.name.startswith("fixed")
            lw = 2.6 if r.name in ("SAND", "naive", "joint") else 1.1
            alpha = 0.25 if is_fan else 0.95
            ax_new.plot(eps, new, marker="o", ms=4, color=r.color, lw=lw,
                        alpha=alpha, label=r.name)
            # old-tasks line only where it's not NaN
            oe = [e for e, o in zip(eps, old) if not (o != o)]  # not NaN
            oo = [o for o in old if not (o != o)]
            if oe:
                ax_old.plot(oe, oo, marker="s", ms=4, color=r.color, lw=lw,
                            alpha=alpha)
        ax_new.set_title(f"task {ti} — NEW-task accuracy", fontsize=10)
        ax_old.set_title(f"task {ti} — OLD-tasks accuracy (forgetting)", fontsize=10)
        ax_new.set_ylim(0, 1.0); ax_old.set_ylim(0, 1.0)
        ax_new.set_xlabel("epoch within task"); ax_old.set_xlabel("epoch within task")
        ax_new.grid(True, alpha=0.3); ax_old.grid(True, alpha=0.3)
        if col == 0:
            ax_new.set_ylabel("new-task acc")
            ax_old.set_ylabel("old-tasks mean acc")
    axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    cfig.suptitle(f"SAND \u2014 Learning curves over time ({stream}): "
                  f"top row = how fast each method learns each NEW task; "
                  f"bottom row = how much it forgets OLD tasks per epoch",
                  fontsize=12, fontweight="bold")
    cfig.tight_layout(rect=[0, 0, 1, 0.95])
    curves_path = out_path.replace(".png", "-curves.png")
    cfig.savefig(curves_path, dpi=130)
    print(f"Plot saved to: {curves_path}", flush=True)


# learning-time threshold for the 'time to learn' metric: first epoch at which
# new-task accuracy reaches this value.
_LEARN_THRESH = 0.80


def _per_run_metrics(run: list[Result], n_tasks: int) -> dict[str, dict[str, float]]:
    """Compute the three commenter-requested metrics for one run (one seed),
    per condition. Returns {name: {time_to_learn, learning_gained, memory_lost}}.

    - time_to_learn  : mean over tasks 1..n-1 of the first epoch (1-indexed) at
                       which new-task accuracy >= _LEARN_THRESH; if a method
                       never reaches it within the budget, count as n_epochs+1
                       (i.e. 'did not learn in time'). LOWER = faster learning.
    - learning_gained: final average accuracy across all tasks. HIGHER = more learned.
    - memory_lost    : average forgetting. LOWER = less forgotten.
    """
    out = {}
    for r in run:
        ttls = []
        for ti in range(1, min(n_tasks, len(r.curves))):
            rows = r.curves[ti]
            reached = None
            for (ep, new_acc, _old) in rows:
                if new_acc >= _LEARN_THRESH:
                    reached = ep + 1  # 1-indexed epoch
                    break
            ttls.append(reached if reached is not None else (len(rows) + 1))
        ttl = float(sum(ttls) / len(ttls)) if ttls else float("nan")
        out[r.name] = {
            "time_to_learn": ttl,
            "learning_gained": r.final_avg_accuracy() if len(r.acc_matrix) > 1
                               else r.acc_matrix[-1][0],
            "memory_lost": r.avg_forgetting(),
        }
    return out


def plot_metrics(all_runs: list[list[Result]], out_path: str, stream: str) -> None:
    """Three dedicated metric bar charts (mean +/- std across seeds), one per
    quantity the commenter asked for: time to learn, total learning gained,
    total memory lost. Each chart is one bar per condition, with the
    no-technique control (naive) on the same axes -> directly comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_tasks = max(len(r.acc_matrix) for r in all_runs[0])
    per_run = [_per_run_metrics(run, n_tasks) for run in all_runs]
    names = list(per_run[0].keys())
    colors = {r.name: r.color for r in all_runs[0]}

    def agg(metric: str):
        vals = {nm: [pr[nm][metric] for pr in per_run] for nm in names}
        means = [float(np.mean(vals[nm])) for nm in names]
        stds = [float(np.std(vals[nm])) for nm in names] if len(all_runs) > 1 \
               else [0.0] * len(names)
        return means, stds

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    specs = [
        ("time_to_learn", f"Time to learn\n(mean epochs to reach "
         f"{int(_LEARN_THRESH*100)}% new-task acc; lower = faster)", "lower is better"),
        ("learning_gained", "Total learning gained\n(final avg accuracy over all tasks; higher = more)", "higher is better"),
        ("memory_lost", "Total memory lost\n(avg forgetting; lower = less forgotten)", "lower is better"),
    ]
    x = np.arange(len(names))
    for ax, (metric, title, better) in zip(axes, specs):
        means, stds = agg(metric)
        ax.bar(x, means, yerr=stds, capsize=4,
               color=[colors.get(nm, "#9467bd") for nm in names],
               alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel("epochs" if metric == "time_to_learn" else "accuracy / forgetting")
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.03, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.text(0.98, 0.97, better, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="gray", style="italic")
    fig.suptitle(f"SAND \u2014 Experiment 1 metrics ({stream}, mean +/- std over "
                 f"{len(all_runs)} seed(s))\nwith-vs-without: naive is the no-technique control",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    metrics_path = out_path.replace(".png", "-metrics.png")
    fig.savefig(metrics_path, dpi=130)
    print(f"Plot saved to: {metrics_path}", flush=True)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", type=str, default="varied",
                   choices=["varied", "split", "permuted"],
                   help="task stream: 'varied' (DEFAULT; mixed drift-need -- some "
                        "tasks reuse digits the backbone knows, others are novel; "
                        "the setting where per-task adaptivity can win), 'split' "
                        "(disjoint digit pairs, all equally novel), 'permuted' "
                        "(pixel permutations).")
    p.add_argument("--n_tasks", type=int, default=5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32,
                   help="hidden layer width (small => capacity-scarce => drifting "
                        "the direction to fit a new task really hurts old tasks "
                        "=> the stability/plasticity trade-off is stressed)")
    p.add_argument("--shared_head", action=argparse.BooleanOptionalAction, default=True,
                   help="every task uses the SAME output neurons (head overwritten "
                        "-> output interference -> forgetting returns). DEFAULT ON. "
                        "Use --no-shared-head to give each task disjoint outputs "
                        "(no forgetting). Needed for the benchmark to stress drift.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--seed", type=int, default=0,
                   help="single-seed run (ignored if --seeds is given)")
    p.add_argument("--seeds", type=str, default="0,1,2",
                   help="comma-separated seeds for a multi-seed run with mean±std "
                        "aggregation (DEFAULT 0,1,2). Use a single value, e.g. "
                        "--seeds 0, for a single run.")
    # ablation grid -- 3 fixed fractions spanning the dial (low / med / high)
    p.add_argument("--fracs", type=str, default="0,0.4,1.0",
                   help="comma-separated fixed direction-LR fractions for the "
                        "ablation (DEFAULT 0,0.4,1.0 = freeze / best-fixed / full "
                        "drift). Pass a denser grid, e.g. 0,0.1,...,1.0, for the "
                        "full fan.")
    p.add_argument("--no_ablation", action="store_true",
                   help="skip the fixed-fraction ablation (just naive + SAND + joint)")
    p.add_argument("--skip_joint", action="store_true",
                   help="skip the joint upper bound (saves time)")
    # SAND hyperparameters
    p.add_argument("--max_frac", type=float, default=1.0,
                   help="SAND: max direction LR as a fraction of the gain LR "
                        "(applied when a task is fully surprising, surprise=1)")
    p.add_argument("--surprise_temp", type=float, default=1.0,
                   help="SAND: temperature shaping the cosine-conflict safety. "
                        "<1 sharpens (drifts more when safe), >1 softens. Default 1.0.")
    p.add_argument("--mem_size", type=int, default=256,
                   help="SAND: # of examples stored per past task in the tiny replay "
                        "memory (A-GEM-style). Needed to measure novelty relative to "
                        "old tasks via gradient cosine similarity. Standard in CL.")
    p.add_argument("--probe_batches", type=int, default=4,
                   help="SAND: # of batches averaged for the gradient probe signal")
    p.add_argument("--plot", type=str, default="result-exp-1.png",
                   help="path to save the results plot")
    p.add_argument("--log", type=str, default="exp1.log",
                   help="path to tee the run log to (DEFAULT exp1.log; use empty "
                        "string for no file). The script mirrors all console output "
                        "to this file so you don't need `| tee`.")
    args = p.parse_args()

    # tee stdout (and stderr) to the log file so `python3 run_experiment_1.py`
    # alone produces both console output AND the log file.
    if args.log:
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data); s.flush()
            def flush(self):
                for s in self.streams: s.flush()
        _logf = open(args.log, "w")
        sys.stdout = _Tee(sys.stdout, _logf)
        sys.stderr = _Tee(sys.stderr, _logf)

    fracs = [float(s) for s in args.fracs.split(",") if s.strip() != ""]

    # output head size: shared k-output head where k = max classes any task uses,
    # else 10-way. For 'varied' task 0 is 5-way so k>=5; for 'split' k=2.
    if args.shared_head:
        k = 2  # default for split
        # peek at the stream to size the head to the largest task
        if args.stream == "varied":
            k = 5
        args.n_outputs = k
    else:
        args.n_outputs = 10

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | stream: {args.stream} | n_tasks: {args.n_tasks} | "
          f"seed: {args.seed}", flush=True)

    print("Loading MNIST ...", flush=True)
    x_tr, y_tr, x_te, y_te = load_mnist(args.data_dir)

    def run_one(seed: int) -> list[Result]:
        """Run all conditions once for a given seed; return the list of Results."""
        torch.manual_seed(seed)
        tasks, joint_x, joint_y = build_stream(x_tr, y_tr, x_te, y_te, args)
        print(f"\n########## seed = {seed} ##########", flush=True)
        print(f"Tasks: {[t.name for t in tasks]}", flush=True)
        results: list[Result] = []
        n_cond = 1 + (0 if args.no_ablation else len(fracs)) + 1 + (0 if args.skip_joint else 1)
        idx = 0
        idx += 1; print(f"\n[{idx}/{n_cond}] naive baseline ...", flush=True)
        results.append(run_naive(tasks, device, args))
        if not args.no_ablation:
            for frac in fracs:
                idx += 1; print(f"\n[{idx}/{n_cond}] fixed fraction = {frac} (ablation) ...", flush=True)
                results.append(run_fixed_frac(tasks, device, args, frac))
        idx += 1; print(f"\n[{idx}/{n_cond}] SAND (surprise-adaptive) ...", flush=True)
        results.append(run_sand(tasks, device, args))
        if not args.skip_joint:
            idx += 1; print(f"\n[{idx}/{n_cond}] joint upper bound ...", flush=True)
            results.append(run_joint(tasks, joint_x, joint_y, device, args))
        return results

    # ---- decide single vs multi-seed ----
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""] if args.seeds else [args.seed]
    multi = len(seeds) > 1

    if not multi:
        results = run_one(seeds[0])
        print_summary(results)
        plot_results(results, args.plot, args.stream)
        plot_metrics([results], args.plot, args.stream)
        return

    # ---- multi-seed: run each, aggregate mean±std ----
    all_runs: list[list[Result]] = []
    for seed in seeds:
        all_runs.append(run_one(seed))
        # per-seed summary (compact) so you can see each run's numbers
        print_summary(all_runs[-1])

    print("\n" + "#" * 72)
    print(f"MULTI-SEED AGGREGATION  (seeds = {seeds})")
    print("#" * 72)
    # align results by condition name across runs
    names = [r.name for r in all_runs[0]]
    print(f"{'condition':<28}{'acc mean':>10}{'acc std':>9}{'fgt mean':>10}{'fgt std':>9}")
    print("-" * 66)
    agg: dict[str, dict] = {}
    for i, name in enumerate(names):
        accs = [run[i].final_avg_accuracy() if len(run[i].acc_matrix) > 1
                else run[i].acc_matrix[-1][0] for run in all_runs]
        fgts = [run[i].avg_forgetting() for run in all_runs]
        am, asd = st.mean(accs), (st.pstdev(accs) if len(accs) > 1 else 0.0)
        fm, fsd = st.mean(fgts), (st.pstdev(fgts) if len(fgts) > 1 else 0.0)
        agg[name] = {"acc": accs, "fgt": fgts, "acc_mean": am, "acc_std": asd,
                     "fgt_mean": fm, "fgt_std": fsd}
        print(f"{name:<28}{am:>10.3f}{asd:>9.3f}{fm:>10.3f}{fsd:>9.3f}")
    print("-" * 66)

    # ---- multi-seed go/no-go ----
    fixed_names = [n for n in names if n.startswith("fixed")]
    sand_name = "SAND"
    if fixed_names and sand_name in agg and len(agg[sand_name]["acc"]) > 1:
        fixed_means = [agg[n]["acc_mean"] for n in fixed_names]
        avg_fixed = st.mean(fixed_means)
        med_fixed = st.median(fixed_means)
        best_fixed = max(fixed_means)
        sand_acc = agg[sand_name]["acc_mean"]
        sand_std = agg[sand_name]["acc_std"]
        print("\n--- Multi-seed go/no-go (mean over seeds) ---")
        print("  (fairness: 'best fixed' = hindsight ceiling; 'avg/median fixed' = "
              "no-lookahead fair bar; SAND = online, untuned, per-task.)")
        print(f"  avg    fixed (fair bar) : acc={avg_fixed:.3f}")
        print(f"  median fixed           : acc={med_fixed:.3f}")
        print(f"  best   fixed (hindsight): acc={best_fixed:.3f}")
        print(f"  SAND (mean±std)        : acc={sand_acc:.3f} ± {sand_std:.3f}")
        diff = sand_acc - avg_fixed
        # rough significance: mean diff > 0 AND sand_std not huge relative to diff
        print(f"  SAND - avg_fixed       : {diff:+.3f}")
        if diff > 0.005 and diff > sand_std:
            print("  VERDICT: SAND beats fair bar clearly (mean gap > 0.5% AND > its std). GO.")
        elif diff > 0:
            print("  VERDICT: SAND slightly above fair bar but within noise. INCONCLUSIVE "
                  "(consider more seeds).")
        else:
            print("  VERDICT: fair bar >= SAND. Adaptive probe not earning its keep. NO-GO.")
    print("#" * 72)

    # plot the FIRST seed's run as the representative figure
    plot_results(all_runs[0], args.plot, args.stream)
    plot_metrics(all_runs, args.plot, args.stream)


if __name__ == "__main__":
    main()
