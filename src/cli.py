import hydra
from hydra.utils import instantiate

from torch import nn


@hydra.main(config_path="configs", config_name="catboost_config", version_base="1.2.0")
def train_from_folder(cfg):
    solver = instantiate(cfg.solver)
    solver.train()
    solver.save_model()
    print(max(solver.model.evals_result_['validation']['MAPE']))
    #results = solver.evaluate()
    #print(results)

if __name__ == "__main__":
    train_from_folder()