import torch
from torch import nn
from catboost import CatBoostRegressor, Pool
import hydra
from hydra.utils import instantiate, call, get_method

from pathlib import Path
from functools import partial

from utils import instantiate_from_config, get_callable_object_name

class NeuralSolver:
    def __init__(self, model, loss, n_epochs, optimizer,
                 train_loader, val_loader, metric_fns,
                 logger, device, save_dir, model_name):
        # TODO add to config
        self.main_metric_name = "spearmanr"
        self.best_metric_value = 0.0
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / model_name
        self.train_loader = instantiate(train_loader).loader
        self.val_loader = instantiate(val_loader).loader

        self.model = instantiate(
            model, n_categorical=self.train_loader.dataset.n_categorical,
            n_numerical=self.train_loader.dataset.n_numeric,
            vocab_sizes=self.train_loader.dataset.vocab_sizes,
        ).to(self.device)

        if hasattr(self.model, "return_logits"):
            self.use_logits = self.model.return_logits
        else:
            self.use_logits = False

        self.optimizer = instantiate(optimizer, params=self.model.parameters())
        self.loss_fn = instantiate_from_config(loss)
        self.metric_fns = [
            instantiate_from_config(metric_fn) for metric_fn in metric_fns
        ]
        self.data_iter = iter(self.train_loader)
        self.n_epochs = n_epochs
        self.print_every = logger.print_every
        self.eval_every = logger.eval_every
        self.save_every = logger.save_every
        self.verbose = logger.verbose

        self._reset()

    def _reset(self):
        self.epoch = 0
        self.loss_history = []

    def train_step(self):
        try:
            data, label = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            data, label = next(self.data_iter)
        prediction = self.model(data)
        if self.use_logits:
            loss = self.loss_fn(self.model.logits, label)
        else:
            loss = self.loss_fn(prediction, label)
        self.loss_history.append(loss.detach().cpu().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        n_iter = self.n_epochs * len(self.train_loader)
        for step in range(n_iter):
            self.train_step()
            if self.verbose and step % self.print_every == 0:
                print(f"(Iteration {step + 1} / {n_iter}) "
                      f"loss: {self.loss_history[-1]:.3f}")
            if step % self.eval_every == 0:
                (eval_metric_values, eval_predictions,
                 eval_labels) = self.evaluate()
                print(eval_metric_values)
                if eval_metric_values[self.main_metric_name] > self.best_metric_value:
                    self.best_metric_value = eval_metric_values[self.main_metric_name]
                    self.save_model()

            if step % self.save_every == 0:
                self.save_model()

    @torch.no_grad()
    def infer(self):
        predictions = []
        for data in self.val_loader:
            prediction = self.model(data)
            predictions.append(prediction)
        predictions = torch.cat(predictions)
        return predictions

    @torch.no_grad()
    def evaluate(self):
        predictions = []
        labels = []
        for data, label in self.val_loader:
            prediction = self.model(data)
            predictions.append(prediction)
            labels.append(label)
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        metrics = {}
        for metric_fn in self.metric_fns:
            metric_fn_name = get_callable_object_name(metric_fn)
            metric_value = metric_fn(predictions, labels)
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            metrics[metric_fn_name] = round(metric_value, 3)

        return metrics, predictions, labels

    def save_model(self):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, self.save_path)


class CatboostSolver:
    def __init__(self, model, loss,
                 train_loader, val_loader, metric_fns,
                 logger, device, save_dir, model_name,
                 early_stopping_rounds):
        # TODO add to config
        self.main_metric_name = "spearmanr"
        self.best_metric_value = 0.0
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / model_name
        self.train_loader = instantiate(train_loader).loader
        self.val_loader = instantiate(val_loader).loader
        self.train_pool = self.get_pool_from_data(self.train_loader)
        self.val_pool = self.get_pool_from_data(self.val_loader)
        self.early_stopping_rounds = early_stopping_rounds
        print(instantiate(loss))
        self.model = instantiate(
            model, loss_function=instantiate(loss), leaf_estimation_method='Gradient',
            leaf_estimation_iterations=1
        )
        self.metric_fns = [
            instantiate_from_config(metric_fn) for metric_fn in metric_fns
        ]

        self.logging = logger
        self.print_every = logger.print_every
        self.eval_every = logger.eval_every
        self.save_every = logger.save_every
        self.verbose = logger.verbose

        self._reset()

    def _reset(self):
        self.epoch = 0
        self.loss_history = []

    def get_pool_from_data(self, dataloader):
        return Pool(
            dataloader.dataset.data_df,
            label=dataloader.dataset.labels,
            cat_features=dataloader.dataset.categorical_features
        )

    def train(self):
        self.model.fit(
            self.train_pool,
            eval_set=self.val_pool,
            early_stopping_rounds=None,
            metric_period=self.logging.eval_every,
            plot_file=self.logging.plot_file,
            #save_snapshot=True,
            #snapshot_file=self.logging.snapshot_file,
        )

    def infer(self):
        return self.model.predict(self.val_pool)

    def evaluate(self):
        raise NotImplementedError("Evaluation is not implemented yet for CatboostSolver.")

    def save_model(self):
        self.model.save_model(self.save_path)