"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network.
You shouldn't need to make any changes to this file.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from eval_utils import calculate_perplexity

class Trainer:
    def __init__(self, config, model, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # Validation loader if provided
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                shuffle=False,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        best_val_loss = float('inf')
        no_improvement_counter = 0  # Counter to track consecutive iterations without improvement

        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device).long() for t in batch]
            x, y = batch
            # print("Batch: x: ", x.shape, "y: ", y.shape)            

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow


            # Validation step with early stopping
            if self.val_dataset is not None and self.iter_num % config.validation_interval == 0:
                val_loss = self.evaluate(val_loader)

                # Calculate relative improvement from the best validation loss
                if best_val_loss != float('inf'):
                    relative_change = abs((val_loss - best_val_loss) / best_val_loss)
                else:
                    relative_change = float('inf')

                print(f'Iteration {self.iter_num}, Validation Loss: {val_loss:.4f}, Relative Improvement: {relative_change:.4f}')

                if (val_loss < best_val_loss) and relative_change > config.min_relative_improvement:
                    best_val_loss = val_loss
                    no_improvement_counter = 0
                    print(f"Validation Loss Improved: {val_loss:.5f}; Resetting Patience")
                else:
                    no_improvement_counter += 1
                    print(f"Validation Loss Showed No Improvement: {val_loss:.5f}; Patience: {no_improvement_counter}")

                # Check for early stopping
                if no_improvement_counter >= config.patience:
                    print(f'No improvement for {config.patience} consecutive iterations. Stopping training.')
                    break

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        val_perplexity = None
        if self.val_dataset is not None:
            val_perplexity = calculate_perplexity(model, val_loader, device=self.device)
            # print(f"Train Perplexity: {calculate_perplexity(model, train_loader, device=self.device):.5f}")
            print(f"Val Perplexity: {val_perplexity:.5f}")

        return best_val_loss, val_perplexity

    def evaluate(self, val_loader):
        model, config = self.model, self.config
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = [t.to(self.device).long() for t in val_batch]
                val_x, val_y = val_batch
                val_logits, val_loss = model(val_x, val_y)
                total_loss += val_loss.item()
                num_batches += 1

        average_loss = total_loss / num_batches
        model.train()
        return average_loss