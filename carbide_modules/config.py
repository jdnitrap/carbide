"""Global hyperparameters and run settings."""
import os


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
        self.seq_len = 128
        self.learning_rate = 3e-3
        self.steps_total = 3000
        self.checkpoint_dir = "carbide_checkpoints"
        self.data_file = None  # Track current dataset

        # task1 refinement #4: periodic MDBE-table snapshots during training
        # (not just the one-shot export in Save Outputs), so it's possible
        # to check whether same-class bytes actually drift closer together
        # in embedding space over training instead of assuming it.
        self.mdbe_snapshot_dir = "mdbe_snapshots"
        self.mdbe_snapshot_interval = 200  # steps

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.mdbe_snapshot_dir):
            os.makedirs(self.mdbe_snapshot_dir)


config = Config()
