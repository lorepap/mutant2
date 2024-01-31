import copy
from keras.callbacks import Callback
import json
import numpy as np
import copy

class TrainingCallback(Callback):
    def __init__(self, log_file_path):
        super(TrainingCallback, self).__init__()
        self.log_file_path = log_file_path
        
    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()
    
    def on_step_end(self, step, logs=None):
        # append logs to external file
        log_copy: dict = copy.deepcopy(logs)
        log_copy['observation']['obs'] = log_copy['observation']['obs'].tolist()
        log_copy['observation']['features'] = log_copy['observation']['features'].tolist()
        with open(self.log_file_path, 'a') as f:
            json.dump(log_copy, f, default=self.np_encoder)
            f.write('\n')
