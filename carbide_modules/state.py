"""Live training state shared between the menu loop and the background thread."""
import time
import threading

from .config import config


class TrainingState:
    def __init__(self):
        self.training_lock = threading.Lock()
        self.reset()

    def reset(self):
        self.step = 0
        self.loss_history = []
        self.paused = False
        self.current_loss = 0.0
        self.start_time = time.time()
        self.training_thread = None
        self.training_active = False
        self.training_target_steps = 0

    def get_elapsed(self):
        return time.time() - self.start_time

    def get_speed(self):
        elapsed = self.get_elapsed()
        if elapsed > 0 and self.step > 0:
            return self.step / elapsed
        return 0

    def to_dict(self):
        return {
            'step': self.step,
            'loss_history': self.loss_history,
            'config': {
                'd_model': config.d_model,
                'n_layers': config.n_layers,
                'd_state': config.d_state,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
            }
        }


train_state = TrainingState()
