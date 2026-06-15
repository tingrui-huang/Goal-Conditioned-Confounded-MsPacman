"""Frozen CleanRL Seaquest teacher adapter (runs inside the legacy JAX container).

The network architecture is NOT reimplemented from memory. We load the EXACT
`Network` / `Actor` / `Critic` / `make_env` definitions out of the checkpoint's
own downloaded training script by exec'ing that source with lightweight stubs for
the non-essential heavy imports (torch/tensorboard/tensorboardX/rlax/wandb).

Sampling convention (verified against both training scripts and the vendored
cleanrl eval helper):
    action = argmax(logits / T + gumbel_noise),  gumbel_noise = -log(-log(u))
This is the native Gumbel-Max trick. Native default temperature T = 1.0.

Recurrence: both candidate scripts are pure feed-forward Conv/Dense stacks. There
is NO LSTM/GRU/carry. teacher_recurrent_state = not_applicable.
"""
import os
import sys
import types
import importlib.util
import numpy as np


# --------------------------------------------------------------------------
# Load original source with stubs for non-essential heavy imports.
# --------------------------------------------------------------------------
def _install_stubs():
    """Stub modules that the training scripts import at top level but that are
    irrelevant to the Network/Actor/Critic/make_env definitions we need."""
    def stub(name, **attrs):
        if name in sys.modules:
            return
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    class _SummaryWriter:  # noqa: D401 - inert stub
        def __init__(self, *a, **k): pass
        def add_scalar(self, *a, **k): pass
        def add_text(self, *a, **k): pass
        def close(self): pass

    # torch is imported by the candidate-A script (tensorboard) AND by vendored
    # OCAtari (`from torch import nn`, guarded only against ModuleNotFoundError). If
    # real torch is absent we install a COMPLETE stub (torch, torch.nn,
    # torch.distributions.categorical, torch.utils.tensorboard) so both succeed.
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except Exception:
            stub("torch")
            stub("torch.nn", Module=object)
            stub("torch.distributions")
            stub("torch.distributions.categorical", Categorical=object)
            stub("torch.utils")
            stub("torch.utils.tensorboard", SummaryWriter=_SummaryWriter)
            t = sys.modules["torch"]
            t.nn = sys.modules["torch.nn"]
            t.distributions = sys.modules["torch.distributions"]
            sys.modules["torch.distributions"].categorical = sys.modules["torch.distributions.categorical"]
            t.utils = sys.modules["torch.utils"]
            sys.modules["torch.utils"].tensorboard = sys.modules["torch.utils.tensorboard"]
    # tensorboardX (candidate B)
    try:
        import tensorboardX  # noqa: F401
    except Exception:
        stub("tensorboardX", SummaryWriter=_SummaryWriter)
    # rlax (candidate B, training only)
    try:
        import rlax  # noqa: F401
    except Exception:
        stub("rlax")
    # wandb (optional)
    try:
        import wandb  # noqa: F401
    except Exception:
        stub("wandb")


def load_original_module(src_path, mod_name):
    """Exec the original training .py into a module object WITHOUT running main()."""
    _install_stubs()
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    mod = types.ModuleType(mod_name)
    mod.__file__ = src_path
    mod.__name__ = mod_name  # not "__main__" -> guards do not fire
    sys.modules[mod_name] = mod
    code = compile(src, src_path, "exec")
    exec(code, mod.__dict__)
    for attr in ("Network", "Actor", "Critic", "make_env"):
        if not hasattr(mod, attr):
            raise AssertionError(f"original source missing required symbol: {attr}")
    return mod


# --------------------------------------------------------------------------
# Teacher adapter
# --------------------------------------------------------------------------
class CleanRLSeaquestTeacher:
    """Frozen feed-forward IMPALA teacher loaded from a .cleanrl_model checkpoint."""

    OBS_SHAPE = (4, 84, 84)  # envpool atari: (stack=4, 84, 84) uint8, NCHW
    recurrent_state = "not_applicable"

    def __init__(self, ckpt_path, src_path, mod_name, template_action_dim=18):
        import jax
        import jax.numpy as jnp
        import flax
        self._jax = jax
        self._jnp = jnp
        self._flax = flax
        self.ckpt_path = ckpt_path
        self.src_path = src_path
        self.src = load_original_module(src_path, mod_name)

        # Build a structure template (leaf shapes irrelevant; flax restores arrays).
        net = self.src.Network()
        actor = self.src.Actor(action_dim=template_action_dim)
        critic = self.src.Critic()
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        dummy = np.zeros((1,) + self.OBS_SHAPE, dtype=np.uint8)
        net_p = net.init(k1, dummy)
        hid = net.apply(net_p, dummy)
        actor_p = actor.init(k2, hid)
        critic_p = critic.init(k3, hid)

        with open(ckpt_path, "rb") as f:
            args, (net_p, actor_p, critic_p) = flax.serialization.from_bytes(
                (None, (net_p, actor_p, critic_p)), f.read()
            )
        self.train_args = args
        self.network_params = net_p
        self.actor_params = actor_p
        self.critic_params = critic_p
        self._network = net
        # Real action_dim from the restored actor head: Dense_0 kernel (256, A)
        self.action_dim = int(
            actor_p["params"]["Dense_0"]["kernel"].shape[1]
        )
        self._actor = self.src.Actor(action_dim=self.action_dim)

        # jitted forward producing logits (and value)
        def _forward(net_p, actor_p, critic_p, obs):
            hidden = net.apply(net_p, obs)
            logits = self._actor.apply(actor_p, hidden)
            value = critic.apply(critic_p, hidden)
            return logits, value
        self._forward = jax.jit(_forward)

        # noise-schedule index (the only mutable "sampling state"; feed-forward net)
        self._noise_idx = 0
        self._validate_params()

    # -- mandated assertions ------------------------------------------------
    def _validate_params(self):
        import jax.numpy as jnp
        flat = self._flax.traverse_util.flatten_dict(
            {"network": self.network_params, "actor": self.actor_params,
             "critic": self.critic_params}
        )
        assert len(flat) > 0, "empty parameter tree"
        for k, v in flat.items():
            arr = np.asarray(v)
            assert np.all(np.isfinite(arr)), f"non-finite parameter at {k}"
        # actor head must map 256 -> action_dim
        assert self.actor_params["params"]["Dense_0"]["kernel"].shape[0] == 256
        # logits finite + bitwise-reproducible on a fixed input
        x = self._fixed_probe()
        l1 = np.asarray(self.logits(x))
        l2 = np.asarray(self.logits(x))
        assert np.array_equal(l1, l2), "logits not bitwise-reproducible"
        assert np.all(np.isfinite(l1)), "non-finite logits"
        assert l1.shape[-1] == self.action_dim

    def _fixed_probe(self):
        rng = np.random.RandomState(12345)
        return rng.randint(0, 256, size=(1,) + self.OBS_SHAPE, dtype=np.uint8)

    # -- core API -----------------------------------------------------------
    def preprocess(self, raw_frame_or_stack):
        """EnvPool already yields (B,4,84,84) uint8. Accept (4,84,84) or batched."""
        a = np.asarray(raw_frame_or_stack)
        if a.ndim == 3:
            a = a[None]
        assert a.shape[1:] == self.OBS_SHAPE, f"bad teacher obs shape {a.shape}"
        return a.astype(np.uint8)

    def logits(self, teacher_obs):
        obs = self.preprocess(teacher_obs)
        logits, _ = self._forward(self.network_params, self.actor_params,
                                  self.critic_params, obs)
        return np.asarray(logits)

    def value(self, teacher_obs):
        obs = self.preprocess(teacher_obs)
        _, v = self._forward(self.network_params, self.actor_params,
                             self.critic_params, obs)
        return np.asarray(v).squeeze(-1)

    @staticmethod
    def gumbel_from_uniform(u):
        u = np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        return -np.log(-np.log(u))

    def sample_action(self, teacher_obs, noise, temperature=1.0):
        """Native Gumbel-Max with EXTERNALLY supplied gumbel noise (shape (...,A)).

        action = argmax(logits / T + noise). Pass pre-generated noise so the same
        vector can be reused across forced-action branches.
        """
        logits = self.logits(teacher_obs)            # (B, A)
        noise = np.asarray(noise, dtype=np.float64)
        if noise.ndim == 1:
            noise = noise[None]
        scores = logits / float(temperature) + noise
        return np.argmax(scores, axis=-1).astype(np.int64)

    def greedy_action(self, teacher_obs):
        return np.argmax(self.logits(teacher_obs), axis=-1).astype(np.int64)

    # -- adapter "state" = position in pre-generated noise schedule ----------
    def get_state(self):
        return {"noise_idx": int(self._noise_idx),
                "recurrent_state": self.recurrent_state}

    def set_state(self, state):
        self._noise_idx = int(state["noise_idx"])
