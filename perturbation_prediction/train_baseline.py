import numpy as np

from harness import run_model


class MeanModel:
    def fit(self, F, Y, n_cells):
        self.mu = Y.mean(axis=0)
        return self

    def predict(self, F):
        return np.tile(self.mu, (len(F), 1))


if __name__ == "__main__":
    run_model("baseline", MeanModel, features="raw")
