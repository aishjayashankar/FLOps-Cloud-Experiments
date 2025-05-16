import torch
import threading

from flops_infra_drift.AutoEncoderDecoder import AutoEncoderDecoder
from flops_infra_drift.LSTMWeightPredictor import LSTMWeightPredictor
from flops_infra_drift.Utils import autoDecode, autoEncode

class ThreadedPredictor(threading.Thread):
    def __init__(self, ae: AutoEncoderDecoder, lstm: LSTMWeightPredictor, X_test: torch.Tensor, iteration):
        super().__init__()
        self.ae = ae
        self.lstm = lstm
        self.X_test = X_test
        self.iteration = iteration
        self.result = None

    def run(self):
        print(f"ThreadedPredictor::run() Running threaded prediction for: {self.iteration}")
        self.X_test = torch.stack(self.X_test).unsqueeze(0)

        self.X_test = autoEncode(self.ae, self.X_test)
        print(f"ThreadedPredictor::run() Iteration: {self.iteration}, Auto-encoding complete")

        self.lstm.eval()
        with torch.no_grad():
            predicted_weights = self.lstm(self.X_test)
        print(f"ThreadedPredictor::run() Iteration: {self.iteration}, prediction complete")

        predicted_weights = autoDecode(self.ae, predicted_weights)
        print(
            f"ThreadedPredictor::run() Iteration: {self.iteration}, Predicted weights shape after decoding: {predicted_weights.shape}"
        )

        self.result = predicted_weights