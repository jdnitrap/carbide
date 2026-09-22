"""Global hyperparameters and run settings."""
import os
import torch


class Config:
    def __init__(self):
        # task1 Part 2: raised from the original d_model=64/n_layers=2/
        # d_state=16 (52,608 params) now that Part 1's chunked scan makes
        # scaling up ~linear instead of catastrophic. Sized against the
        # real ~5MB corpus (see dataset.py) rather than maxed out to the
        # "low single-digit millions" task1 literally names — 514,688
        # params here is ~9.7 bytes/param of training data; the larger
        # tier (d_model=512,n_layers=6,d_state=64, 2.26M params) is
        # verified to train cleanly too (see carbide/tests/) but was left
        # off by default since ~2.2 bytes/param on this corpus invites
        # overfitting rather than learning. Bump it yourself via the
        # Hyperparameters menu if you grow the dataset further.
        self.d_model = 256
        self.n_layers = 4
        self.d_state = 32
        self.batch_size = 1
        # 2026-09-16: bumped 128 -> 2048 to test whether longer context
        # helps the discourse-level MDBE columns (PDTB/clause-scan) and
        # generation coherence, which 128 bytes (~1 sentence) starves.
        # Throughput scales ~linearly with seq_len (SSM, not attention),
        # so expect ~16x fewer steps/sec than the 128 baseline.
        self.seq_len = 2048
        self.learning_rate = 3e-3
        # "layered" = the full L1/L2/L3 LayerStack (letter book, word book, sentence book): the
        # main model, by design. "beside" = only flags + grammar beside the byte embedding, no word
        # table and no sentence layer; it had the best held-out loss in the corrected ablation
        # (2026-09-21) and is kept as the loss baseline. "graph" = layered plus the graph memory.
        self.model_kind = "layered"
        # 2026-09-21, at the user's request ahead of moving this machine's work to a laptop
        # with a GPU: "auto" picks CUDA when it's present and falls back to CPU when it's not,
        # so the same checkout trains on either machine with no edit. "cuda"/"cpu" force one
        # side explicitly (see resolved_device()); `set train.device cuda|cpu|auto`.
        self.device_pref = "auto"
        self.graph_db = "graph_memory.db"  # built by: python -m carbide_modules.graphmem build-core
        self.steps_total = 3000
        self.checkpoint_dir = "carbide_checkpoints"
        self.data_file = None  # Track current dataset

        # task1 refinement #4: periodic MDBE-table snapshots during training
        # (not just the one-shot export in Save Outputs), so it's possible
        # to check whether same-class bytes actually drift closer together
        # in embedding space over training instead of assuming it.
        self.mdbe_snapshot_dir = "mdbe_snapshots"
        self.mdbe_snapshot_interval = 200  # steps

        # Bug found 2026-09-16: a full 100k-step run left the checkpoint on
        # disk stuck at whatever step was last manually saved (00:02),
        # because neither train_n_steps() nor background_training_thread()
        # ever called save_checkpoint() -- only an explicit menu action did.
        # Everything trained after that manual save only ever existed in the
        # running process's RAM and was lost the moment it exited, even
        # though "Save outputs" (which only writes CSV/PNG reports from
        # loss_history, never the weights) made it look like the run was
        # captured. This interval drives an automatic checkpoint save (same
        # default filename as "Resume from checkpoint" uses) during training
        # so that can't happen again.
        self.checkpoint_autosave_interval = 1000  # steps

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.mdbe_snapshot_dir):
            os.makedirs(self.mdbe_snapshot_dir)

    def resolved_device(self):
        """The actual torch.device training/generation/export should use right now."""
        if self.device_pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("train.device is set to cuda but no CUDA GPU is available on this machine")
            return torch.device("cuda")
        if self.device_pref == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
