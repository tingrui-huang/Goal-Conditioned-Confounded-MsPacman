"""Phase 2.1b leakage-source-audit data + matched probe net/trainer.

Reuses the FROZEN Phase-2 episode split (seed 2606, 70/15/15), the exact four-frame
stack indices (oldest->newest, clamped to episode start), and the EXACT metrics +
episode-bootstrap from oxy4_train. The ONLY thing that varies across V1-V4 is the visual
input bank (which region transform + 1 vs 4 frames). Banks are built ONE AT A TIME on the
CPU (each ~1.7GB) and moved to the compute device per batch, so Colab does not OOM.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from seaquest_ccrl.data.dataset import SeaquestOfflineDataset
from seaquest_ccrl.models.sa_encoder import preprocess_frames
from seaquest_ccrl.probes.oxy4_data import SPLIT_SEED, K_STACK, FRAME_SIZE, oxygen_class
from seaquest_ccrl.probes.oxy4_train import _metrics, boot_ci  # identical metrics + bootstrap
from seaquest_ccrl.probes import oxy4_regions as R


class AuditData:
    """Metadata-only loader (no frames kept resident) + on-demand per-variant frame banks."""

    def __init__(self, root):
        self.root = root
        ds = SeaquestOfflineDataset(root, oracle=True)        # oracle so raw frames available
        self.files = ds.files
        actions, oxygen, ppos, done, lengths = [], [], [], [], []
        for f in self.files:
            d = np.load(f)                                    # lazy; frames not materialized here
            actions.append(np.asarray(d["actions"], dtype=np.int64))
            oxygen.append(np.asarray(d["oxygen"], dtype=np.float32))
            ppos.append(np.asarray(d["player_pos"], dtype=np.float32))
            done.append(np.asarray(d["done"], dtype=bool))
            lengths.append(len(actions[-1]))
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)
        self.n_ep = len(lengths)
        self.N = int(self.lengths.sum())
        self.episode_of = np.repeat(np.arange(self.n_ep), self.lengths)
        self.t_in_ep = np.concatenate([np.arange(L) for L in self.lengths])
        ep_start = np.repeat(self.offsets, self.lengths)
        ar = np.arange(self.N)
        cols = [np.maximum(ep_start, ar - (K_STACK - 1) + j) for j in range(K_STACK)]
        self.stack_idx = np.stack(cols, axis=1).astype(np.int64)   # (N,4) oldest->newest
        self.actions = np.concatenate(actions)
        self.oxygen = np.concatenate(oxygen)
        self.player_pos = np.concatenate(ppos, axis=0)             # (N,2) x,y
        self.done = np.concatenate(done)
        # episode-level split — IDENTICAL recipe + seed to Phase2Data
        rng = np.random.RandomState(SPLIT_SEED)
        order = rng.permutation(self.n_ep)
        ntr = int(round(0.70 * self.n_ep)); nva = int(round(0.15 * self.n_ep))
        self.train_ep = sorted(int(x) for x in order[:ntr])
        self.val_ep = sorted(int(x) for x in order[ntr:ntr + nva])
        self.test_ep = sorted(int(x) for x in order[ntr + nva:])

    def split_indices(self, which):
        eps = {"train": self.train_ep, "val": self.val_ep, "test": self.test_ep}[which]
        return np.where(np.isin(self.episode_of, eps))[0]

    def build_bank(self, variant, size=FRAME_SIZE):
        """(N,size,size,3) uint8 for one region variant. Raw frames are read per trajectory,
        the region transform is applied BEFORE resize, then the raw frames are freed."""
        banks = []
        for f in self.files:
            frames = np.load(f)["frames"]                          # (T,210,160,3) uint8 (raw)
            tf = np.stack([R.transform_raw(fr, variant) for fr in frames], axis=0)
            banks.append(preprocess_frames(tf, size))             # resize -> (T,size,size,3)
            del frames, tf
        return np.concatenate(banks, axis=0)

    def split_manifest(self):
        return {"split_seed": SPLIT_SEED, "n_episodes": int(self.n_ep),
                "train_episode_ids": self.train_ep, "val_episode_ids": self.val_ep,
                "test_episode_ids": self.test_ep, "frame_stack": K_STACK, "frame_size": FRAME_SIZE}


def stack_from_bank(bank, gidx, stack_idx, k):
    """(M,size,size,3k) uint8 torch on CPU. k=4 -> four-frame stack (oldest->newest); k=1 ->
    single newest frame at gidx."""
    if k == 1:
        st = bank[gidx][:, :, :, None, :]                          # (M,H,W,1,3)
    else:
        st = bank[stack_idx[gidx]]                                 # (M,4,H,W,3)
        st = np.transpose(st, (0, 2, 3, 1, 4))                     # (M,H,W,4,3)
    M, H, W = st.shape[0], st.shape[1], st.shape[2]
    return torch.from_numpy(np.ascontiguousarray(st.reshape(M, H, W, -1)))


class AuditProbeNet(nn.Module):
    """Same backbone as oxy4_net.ProbeNet, parameterized by input channels (3 for V1,
    12 for V2-V4). conv [32,64,128] 3x3 stride2 + Linear(flat+extra,256)->out."""

    def __init__(self, in_ch=12, extra_dim=0, out_dim=1, hidden=256, frame_size=FRAME_SIZE):
        super().__init__()
        self.extra_dim = extra_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_ch, frame_size, frame_size)).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Linear(flat + extra_dim, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, out_dim))

    def forward(self, frames_uint8, extras=None):
        x = frames_uint8.float().permute(0, 3, 1, 2) / 255.0
        h = self.conv(x).flatten(1)
        if self.extra_dim:
            h = torch.cat([h, extras], dim=1)
        return self.head(h)


def _set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train_audit_probe(data, bank, k, gi_tr, gi_va, gi_te, y_tr, y_te, task, out_dim,
                      in_ch, epochs=12, lr=3e-4, batch=256, seed=0, device="cpu",
                      random_init=False):
    """Mirror of oxy4_train.train_probe for an arbitrary frame bank (no scalar extras).
    Train-only target normalization; Adam; MSE/CE. random_init=True skips training (untrained
    CNN sanity control). Returns pred + per_row_loss + episode ids + metrics."""
    _set_seed(seed)
    if task == "reg":
        ymu = y_tr.mean(0); ysd = y_tr.std(0) + 1e-6
        yt = (y_tr - ymu) / ysd
    else:
        yt = y_tr
    model = AuditProbeNet(in_ch=in_ch, extra_dim=0, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batches(gi, y, shuffle):
        idx = np.random.permutation(len(gi)) if shuffle else np.arange(len(gi))
        for s in range(0, len(gi), batch):
            b = idx[s:s + batch]
            fr = stack_from_bank(bank, gi[b], data.stack_idx, k).to(device)
            yield fr, torch.as_tensor(y[b], device=device)

    if not random_init:
        for _ in range(epochs):
            model.train()
            for fr, yy in batches(gi_tr, yt, True):
                pred = model(fr)
                loss = F.mse_loss(pred, yy.float()) if task == "reg" else F.cross_entropy(pred, yy.long())
                opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); preds = []; perrow = []
    with torch.no_grad():
        for fr, yy in batches(gi_te, y_te, False):
            out = model(fr)
            if task == "reg":
                p = out.cpu().numpy() * ysd + ymu
                preds.append(p); perrow.append(((p - yy.cpu().numpy()) ** 2).mean(1))
            else:
                logp = F.log_softmax(out, 1).cpu().numpy()
                preds.append(np.exp(logp))
                perrow.append(-logp[np.arange(len(yy)), yy.cpu().numpy().astype(int)])
    P = np.concatenate(preds); per_row_loss = np.concatenate(perrow)
    return {"pred": P, "per_row_loss": per_row_loss, "episode": data.episode_of[gi_te],
            "y": y_te, "metrics": _metrics(task, P, y_te, out_dim)}
