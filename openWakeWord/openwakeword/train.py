import torch
from torch import optim, nn
import torchinfo
import torchmetrics
import copy
import os
import sys
import tempfile
import uuid
import numpy as np
import scipy
import collections
import argparse
import logging
from tqdm import tqdm
import yaml
from pathlib import Path
import openwakeword
from openwakeword.data import generate_adversarial_texts, augment_clips, mmap_batch_generator
from openwakeword.utils import compute_features_from_generator
from openwakeword.utils import AudioFeatures
import inspect
from openwakeword import fix_wavs
from pathlib import Path


# =========================
# Path helpers (portable configs)
# =========================
import re as _re
def _is_abs_path(p: str) -> bool:
    if not isinstance(p, str) or not p:
        return False
    # Linux/mac absolute
    if p.startswith("/"):
        return True
    # Windows drive absolute: C:\ or C:/
    if _re.match(r"^[a-zA-Z]:[\\/]", p):
        return True
    # UNC paths
    if p.startswith("\\\\"):
        return True
    return False

def _resolve_from_config_dir(config_path: str, value):
    """
    Resolve relative paths from the directory where the YAML config lives.
    Supports str / list[str] / dict[str,str].
    """
    from pathlib import Path
    cfg_dir = Path(config_path).resolve().parent
    if isinstance(value, str):
        if _is_abs_path(value):
            return value
        # If it's a bare command (e.g., 'piper') without any path separators, keep as-is
        if ('/' not in value) and ('\\' not in value) and (':' not in value):
            return value
        return str((cfg_dir / value).resolve())
    if isinstance(value, list):
        out=[]
        for v in value:
            out.append(_resolve_from_config_dir(config_path, v))
        return out
    if isinstance(value, dict):
        out={}
        for k,v in value.items():
            out[k]=_resolve_from_config_dir(config_path, v)
        return out
    return value

def _resolve_config_paths(config_path: str, config: dict) -> dict:
    # Keys that represent paths
    path_keys = [
        "output_dir",
        "positive_samples_dir",
        "custom_positive_data",
        "false_positive_validation_data_path",
        "piper_voice_model",
        "piper_executable",
        "piper_sample_generator_path",
        "rir_paths",
        "background_paths",
        "feature_data_files",
    ]
    for k in path_keys:
        if k in config and config[k] is not None:
            config[k] = _resolve_from_config_dir(config_path, config[k])
    return config
# =========================
# Pretty logging helpers
# =========================

def _rglob_wavs(root: str):
    root_p = Path(root)
    if root_p.is_file() and root_p.suffix.lower() == ".wav":
        return [str(root_p)]
    if not root_p.exists():
        return []
    return [
        str(p) for p in root_p.rglob("*.wav")
        if p.is_file() and p.stat().st_size > 500
    ]

def _human(n: int) -> str:
    try:
        return f"{int(n):,}".replace(",", " ")
    except Exception:
        return str(n)


def _count_wavs(folder: str) -> int:
    try:
        return len(list(Path(folder).glob("*.wav")))
    except Exception:
        return 0


def _count_files(folder: str, exts=(".wav",)) -> int:
    try:
        folder = str(folder)
        if not os.path.isdir(folder):
            return 0
        c = 0
        for ext in exts:
            c += len(list(Path(folder).rglob(f"*{ext}")))
        return c
    except Exception:
        return 0


def stage(title: str):
    line = "=" * 72
    logging.info("\n%s\n%s\n%s", line, title, line)


def setup_logging(level: str = "INFO"):
    level = (level or "INFO").upper()
    log_level = getattr(logging, level, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Приглушаем шумные либы
    for noisy in ["torio", "torch", "tensorflow", "onnx", "onnx_tf", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


# Base model class for an openwakeword model
class Model(nn.Module):
    def __init__(self, n_classes=1, input_shape=(16, 96), model_type="dnn",
                 layer_dim=128, n_blocks=1, seconds_per_example=None):
        super().__init__()

        # Store inputs as attributes
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.seconds_per_example = seconds_per_example
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.best_models = []
        self.best_model_scores = []
        self.best_val_fp = 1000
        self.best_val_accuracy = 0
        self.best_val_recall = 0
        self.best_train_recall = 0

        # Define model (currently on fully-connected network supported)
        if model_type == "dnn":
            class FCNBlock(nn.Module):
                def __init__(self, layer_dim):
                    super().__init__()
                    self.fcn_layer = nn.Linear(layer_dim, layer_dim)
                    self.relu = nn.ReLU()
                    self.layer_norm = nn.LayerNorm(layer_dim)

                def forward(self, x):
                    return self.relu(self.layer_norm(self.fcn_layer(x)))

            class Net(nn.Module):
                def __init__(self, input_shape, layer_dim, n_blocks=1, n_classes=1):
                    super().__init__()
                    self.flatten = nn.Flatten()
                    self.layer1 = nn.Linear(input_shape[0] * input_shape[1], layer_dim)
                    self.relu1 = nn.ReLU()
                    self.layernorm1 = nn.LayerNorm(layer_dim)
                    self.blocks = nn.ModuleList([FCNBlock(layer_dim) for i in range(n_blocks)])
                    self.last_layer = nn.Linear(layer_dim, n_classes)
                    self.last_act = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
                    for block in self.blocks:
                        x = block(x)
                    x = self.last_act(self.last_layer(x))
                    return x

            self.model = Net(input_shape, layer_dim, n_blocks=n_blocks, n_classes=n_classes)

        elif model_type == "rnn":
            class Net(nn.Module):
                def __init__(self, input_shape, n_classes=1):
                    super().__init__()
                    self.layer1 = nn.LSTM(input_shape[-1], 64, num_layers=2, bidirectional=True,
                                          batch_first=True, dropout=0.0)
                    self.layer2 = nn.Linear(64 * 2, n_classes)
                    self.layer3 = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    out, h = self.layer1(x)
                    return self.layer3(self.layer2(out[:, -1]))

            self.model = Net(input_shape, n_classes)

        # Define metrics
        if n_classes == 1:
            self.fp = lambda pred, y: (y - pred <= -0.5).sum()
            self.recall = torchmetrics.Recall(task='binary')
            self.accuracy = torchmetrics.Accuracy(task='binary')
        else:
            def multiclass_fp(p, y, threshold=0.5):
                probs = torch.nn.functional.softmax(p, dim=1)
                neg_ndcs = y == 0
                fp = (probs[neg_ndcs].argmax(axis=1) != 0 & (probs[neg_ndcs].max(axis=1)[0] > threshold)).sum()
                return fp

            def positive_class_recall(p, y, negative_class_label=0, threshold=0.5):
                probs = torch.nn.functional.softmax(p, dim=1)
                pos_ndcs = y != 0
                rcll = (probs[pos_ndcs].argmax(axis=1) > 0
                        & (probs[pos_ndcs].max(axis=1)[0] >= threshold)).sum() / pos_ndcs.sum()
                return rcll

            def positive_class_accuracy(p, y, negative_class_label=0):
                probs = torch.nn.functional.softmax(p, dim=1)
                pos_preds = probs.argmax(axis=1) != negative_class_label
                acc = (probs[pos_preds].argmax(axis=1) == y[pos_preds]).sum() / pos_preds.sum()
                return acc

            self.fp = multiclass_fp
            self.acc = positive_class_accuracy
            self.recall = positive_class_recall

        self.n_fp = 0
        self.val_fp = 0

        # Define logging dict (in-memory)
        self.history = collections.defaultdict(list)

        # Define optimizer and loss
        self.loss = torch.nn.functional.binary_cross_entropy if n_classes == 1 else nn.functional.cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def save_model(self, output_path):
        """
        Saves the weights of a trained Pytorch model
        """
        if self.n_classes == 1:
            torch.save(self.model, output_path)

    def export_to_onnx(self, output_path, class_mapping=""):
        obj = self
        # Make simple model for export based on model structure
        if self.n_classes == 1:
            # Save ONNX model
            torch.onnx.export(self.model.to("cpu"), torch.rand(self.input_shape)[None, ], output_path,
                              output_names=[class_mapping])

        elif self.n_classes >= 1:
            class M(nn.Module):
                def __init__(self):
                    super().__init__()

                    # Define model
                    self.model = obj.model.to("cpu")

                def forward(self, x):
                    return torch.nn.functional.softmax(self.model(x), dim=1)

            # Save ONNX model
            torch.onnx.export(M(), torch.rand(self.input_shape)[None, ], output_path,
                              output_names=[class_mapping])

    def lr_warmup_cosine_decay(self,
                               global_step,
                               warmup_steps=0,
                               hold=0,
                               total_steps=0,
                               start_lr=0.0,
                               target_lr=1e-3
                               ):
        # Cosine decay
        learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold)
                                           / float(total_steps - warmup_steps - hold)))

        # Target LR * progress of warmup (=1 at the final warmup step)
        warmup_lr = target_lr * (global_step / warmup_steps)

        # Choose between `warmup_lr`, `target_lr` and `learning_rate`
        if hold > 0:
            learning_rate = np.where(global_step > warmup_steps + hold,
                                     learning_rate, target_lr)

        learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def forward(self, x):
        return self.model(x)

    def summary(self):
        return torchinfo.summary(self.model, input_size=(1,) + self.input_shape, device='cpu')

    def average_models(self, models=None):
        """Averages the weights of the provided models together to make a new model"""

        if models is None:
            models = self.best_models

        averaged_model = copy.deepcopy(models[0])
        averaged_model_dict = averaged_model.state_dict()

        for key in averaged_model_dict:
            averaged_model_dict[key] *= 0  # set to 0

        for model in models:
            model_dict = model.state_dict()
            for key, value in model_dict.items():
                averaged_model_dict[key] += value

        for key in averaged_model_dict:
            averaged_model_dict[key] /= len(models)

        averaged_model.load_state_dict(averaged_model_dict)
        return averaged_model

    def _select_best_model(self, false_positive_validate_data, val_set_hrs=11.3, max_fp_per_hour=0.5, min_recall=0.20):
        """
        Select the top model based on the false positive rate on the validation data
        """
        false_positive_rates = [0] * len(self.best_models)
        for batch in false_positive_validate_data:
            x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
            for mdl_ndx, model in tqdm(enumerate(self.best_models), total=len(self.best_models),
                                       desc="Find best checkpoints by false positive rate"):
                with torch.no_grad():
                    val_ps = model(x_val)
                    false_positive_rates[mdl_ndx] = false_positive_rates[mdl_ndx] + self.fp(val_ps, y_val[..., None]).detach().cpu().numpy()
        false_positive_rates = [fp / val_set_hrs for fp in false_positive_rates]

        candidate_model_ndx = [ndx for ndx, fp in enumerate(false_positive_rates) if fp <= max_fp_per_hour]
        candidate_model_recall = [self.best_model_scores[ndx]["val_recall"] for ndx in candidate_model_ndx]
        if max(candidate_model_recall) <= min_recall:
            logging.warning(f"No models with recall >= {min_recall} found!")
            return None
        else:
            best_model = self.best_models[candidate_model_ndx[np.argmax(candidate_model_recall)]]
            best_model_training_step = self.best_model_scores[candidate_model_ndx[np.argmax(candidate_model_recall)]]["training_step_ndx"]
            logging.info(f"Best model from training step {best_model_training_step} out of {len(candidate_model_ndx)}"
                         f"models has recall of {np.max(candidate_model_recall)} and false positive rate of"
                         f" {false_positive_rates[candidate_model_ndx[np.argmax(candidate_model_recall)]]}")

        return best_model

    def auto_train(self, X_train, X_val, false_positive_val_data, steps=50000, max_negative_weight=1000,
                   target_fp_per_hour=0.2):
        """A sequence of training steps that produce relatively strong models automatically"""

        # Get false positive validation data duration
        val_set_hrs = 11.3

        # Sequence 1
        stage("STEP 3/4 — TRAINING SEQUENCE 1")
        lr = 0.0001
        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(steps - int(steps * 0.25), steps, 20).astype(np.int64)
        self.train_model(
            X=X_train,
            X_val=X_val,
            false_positive_val_data=false_positive_val_data,
            max_steps=steps,
            negative_weight_schedule=weights,
            val_steps=val_steps, warmup_steps=steps // 5,
            hold_steps=steps // 3, lr=lr, val_set_hrs=val_set_hrs
        )

        # Sequence 2
        stage("STEP 3/4 — TRAINING SEQUENCE 2")
        lr = lr / 10
        steps = steps / 10

        if self.best_val_fp > target_fp_per_hour:
            max_negative_weight = max_negative_weight * 2
            logging.info("Increasing weight on negative examples to reduce false positives...")

        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(1, steps, 20).astype(np.int16)
        self.train_model(
            X=X_train,
            X_val=X_val,
            false_positive_val_data=false_positive_val_data,
            max_steps=steps,
            negative_weight_schedule=weights,
            val_steps=val_steps, warmup_steps=steps // 5,
            hold_steps=steps // 3, lr=lr, val_set_hrs=val_set_hrs
        )

        # Sequence 3
        stage("STEP 3/4 — TRAINING SEQUENCE 3")
        lr = lr / 10

        if self.best_val_fp > target_fp_per_hour:
            max_negative_weight = max_negative_weight * 2
            logging.info("Increasing weight on negative examples to reduce false positives...")

        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(1, steps, 20).astype(np.int16)
        self.train_model(
            X=X_train,
            X_val=X_val,
            false_positive_val_data=false_positive_val_data,
            max_steps=steps,
            negative_weight_schedule=weights,
            val_steps=val_steps, warmup_steps=steps // 5,
            hold_steps=steps // 3, lr=lr, val_set_hrs=val_set_hrs
        )

        # Merge best models
        stage("STEP 4/4 — MERGE BEST CHECKPOINTS")
        logging.info("Merging checkpoints above the 90th percentile into single model...")

        accuracy_percentile = np.percentile(self.history["val_accuracy"], 90)
        recall_percentile = np.percentile(self.history["val_recall"], 90)
        fp_percentile = np.percentile(self.history["val_fp_per_hr"], 10)

        models = []
        for model, score in zip(self.best_models, self.best_model_scores):
            if score["val_accuracy"] >= accuracy_percentile and \
                    score["val_recall"] >= recall_percentile and \
                    score["val_fp_per_hr"] <= fp_percentile:
                models.append(model)

        if len(models) > 0:
            combined_model = self.average_models(models=models)
        else:
            combined_model = self.model

        # Report validation metrics for combined model
        with torch.no_grad():
            for batch in X_val:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                val_ps = combined_model(x)

            combined_model_recall = self.recall(val_ps, y[..., None]).detach().cpu().numpy()
            combined_model_accuracy = self.accuracy(val_ps, y[..., None].to(torch.int64)).detach().cpu().numpy()

            combined_model_fp = 0
            for batch in false_positive_val_data:
                x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
                val_ps = combined_model(x_val)
                combined_model_fp += self.fp(val_ps, y_val[..., None])

            combined_model_fp_per_hr = (combined_model_fp / val_set_hrs).detach().cpu().numpy()

        logging.info(
            "\n################\nFinal Model Accuracy: %s"
            "\nFinal Model Recall: %s\nFinal Model False Positives per Hour: %s"
            "\n################\n",
            combined_model_accuracy, combined_model_recall, combined_model_fp_per_hr
        )

        return combined_model

    def predict_on_features(self, features, model=None):
        """
        Predict on Tensors of openWakeWord features corresponding to single audio clips
        """
        if len(features) < 3:
            features = features[None, ]

        features = features.to(self.device)
        predictions = []
        for x in tqdm(features, desc="Predicting on clips"):
            x = x[None, ]
            batch = []
            for i in range(0, x.shape[1] - 16, 1):  # step size of 1 (80 ms)
                batch.append(x[:, i:i + 16, :])
            batch = torch.vstack(batch)
            if model is None:
                preds = self.model(batch)
            else:
                preds = model(batch)
            predictions.append(preds.detach().cpu().numpy()[None, ])

        return np.vstack(predictions)

    def predict_on_clips(self, clips, model=None):
        """
        Predict on Tensors of 16-bit 16 khz audio data
        """
        F = AudioFeatures(device='cpu', ncpu=4)
        features = F.embed_clips(clips, batch_size=16)
        preds = self.predict_on_features(torch.from_numpy(features), model=model)
        return preds

    def export_model(self, model, model_name, output_dir):
        """Saves the trained openwakeword model to onnx"""
        if self.n_classes != 1:
            raise ValueError("Exporting models to both onnx and tflite with more than one class is currently not supported! "
                             "Use the `export_to_onnx` function instead.")

        logging.info(f"####\nSaving ONNX mode as '{os.path.join(output_dir, model_name + '.onnx')}'")
        model_to_save = copy.deepcopy(model)
        torch.onnx.export(model_to_save.to("cpu"), torch.rand(self.input_shape)[None, ],
                          os.path.join(output_dir, model_name + ".onnx"), opset_version=13)
        return None

    def train_model(self, X, max_steps, warmup_steps, hold_steps, X_val=None,
                    false_positive_val_data=None, positive_test_clips=None,
                    negative_weight_schedule=[1],
                    val_steps=[250], lr=0.0001, val_set_hrs=1):
        # Move models and main class to target device
        self.to(self.device)
        self.model.to(self.device)

        accumulation_steps = 1
        accumulated_samples = 0
        accumulated_predictions = torch.Tensor([]).to(self.device)
        accumulated_labels = torch.Tensor([]).to(self.device)

        for step_ndx, data in tqdm(enumerate(X, 0), total=max_steps, desc="Training"):
            x, y = data[0].to(self.device), data[1].to(self.device)
            y_ = y[..., None].to(torch.float32)

            # Update learning rates
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_warmup_cosine_decay(
                    step_ndx, warmup_steps=warmup_steps, hold=hold_steps,
                    total_steps=max_steps, target_lr=lr
                )

            self.optimizer.zero_grad()

            predictions = self.model(x)

            # Construct batch with only samples that have high loss
            neg_high_loss = predictions[(y == 0) & (predictions.squeeze() >= 0.001)]
            pos_high_loss = predictions[(y == 1) & (predictions.squeeze() < 0.999)]
            y = torch.cat((y[(y == 0) & (predictions.squeeze() >= 0.001)], y[(y == 1) & (predictions.squeeze() < 0.999)]))
            y_ = y[..., None].to(torch.float32)
            predictions = torch.cat((neg_high_loss, pos_high_loss))

            # Set weights for batch
            if len(negative_weight_schedule) == 1:
                w = torch.ones(y.shape[0]) * negative_weight_schedule[0]
                pos_ndcs = y == 1
                w[pos_ndcs] = 1
                w = w[..., None]
            else:
                if self.n_classes == 1:
                    w = torch.ones(y.shape[0]) * negative_weight_schedule[step_ndx]
                    pos_ndcs = y == 1
                    w[pos_ndcs] = 1
                    w = w[..., None]

            if predictions.shape[0] != 0:
                loss = self.loss(predictions, y_ if self.n_classes == 1 else y, w.to(self.device))
                loss = loss / accumulation_steps
                accumulated_samples += predictions.shape[0]

                if predictions.shape[0] >= 128:
                    accumulated_predictions = predictions
                    accumulated_labels = y_
                if accumulated_samples < 128:
                    accumulation_steps += 1
                    accumulated_predictions = torch.cat((accumulated_predictions, predictions))
                    accumulated_labels = torch.cat((accumulated_labels, y_))
                else:
                    loss.backward()
                    self.optimizer.step()
                    accumulation_steps = 1
                    accumulated_samples = 0

                    self.history["loss"].append(loss.detach().cpu().numpy())

                    fp = self.fp(accumulated_predictions, accumulated_labels if self.n_classes == 1 else y)
                    self.n_fp += fp
                    self.history["recall"].append(self.recall(accumulated_predictions, accumulated_labels).detach().cpu().numpy())

                    accumulated_predictions = torch.Tensor([]).to(self.device)
                    accumulated_labels = torch.Tensor([]).to(self.device)

            # Run validation and log validation metrics
            if step_ndx in val_steps and step_ndx > 1 and false_positive_val_data is not None:
                val_fp = 0
                for val_step_ndx, data in enumerate(false_positive_val_data):
                    with torch.no_grad():
                        x_val, y_val = data[0].to(self.device), data[1].to(self.device)
                        val_predictions = self.model(x_val)
                        val_fp += self.fp(val_predictions, y_val[..., None])
                val_fp_per_hr = (val_fp / val_set_hrs).detach().cpu().numpy()
                self.history["val_fp_per_hr"].append(val_fp_per_hr)

            # Get recall on test clips
            if step_ndx in val_steps and step_ndx > 1 and positive_test_clips is not None:
                tp = 0
                fn = 0
                for val_step_ndx, data in enumerate(positive_test_clips):
                    with torch.no_grad():
                        x_val = data[0].to(self.device)
                        batch = []
                        for i in range(0, x_val.shape[1] - 16, 1):
                            batch.append(x_val[:, i:i + 16, :])
                        batch = torch.vstack(batch)
                        preds = self.model(batch)
                        if any(preds >= 0.5):
                            tp += 1
                        else:
                            fn += 1
                self.history["positive_test_clips_recall"].append(tp / (tp + fn))

            if step_ndx in val_steps and step_ndx > 1 and X_val is not None:
                for val_step_ndx, data in enumerate(X_val):
                    with torch.no_grad():
                        x_val, y_val = data[0].to(self.device), data[1].to(self.device)
                        val_predictions = self.model(x_val)
                        val_recall = self.recall(val_predictions, y_val[..., None]).detach().cpu().numpy()
                        val_acc = self.accuracy(val_predictions, y_val[..., None].to(torch.int64))
                        val_fp = self.fp(val_predictions, y_val[..., None])

                self.history["val_accuracy"].append(val_acc.detach().cpu().numpy())
                self.history["val_recall"].append(val_recall)
                self.history["val_n_fp"].append(val_fp.detach().cpu().numpy())

                try:
                    fp_per_hr = float(self.history.get("val_fp_per_hr", [0])[-1])
                except Exception:
                    fp_per_hr = 0.0

                logging.info(
                    "[VAL step=%s] acc=%.4f recall=%.4f fp=%s fp/h=%.4f",
                    step_ndx,
                    float(val_acc.detach().cpu().numpy()),
                    float(val_recall),
                    int(val_fp.detach().cpu().numpy()) if hasattr(val_fp, "detach") else int(val_fp),
                    fp_per_hr
                )

            # Save models with a validation score above/below the 90th percentile
            if step_ndx in val_steps and step_ndx > 1:
                if self.history["val_n_fp"][-1] <= np.percentile(self.history["val_n_fp"], 50) and \
                   self.history["val_recall"][-1] >= np.percentile(self.history["val_recall"], 5):
                    self.best_models.append(copy.deepcopy(self.model))
                    self.best_model_scores.append({
                        "training_step_ndx": step_ndx,
                        "val_n_fp": self.history["val_n_fp"][-1],
                        "val_recall": self.history["val_recall"][-1],
                        "val_accuracy": self.history["val_accuracy"][-1],
                        "val_fp_per_hr": self.history.get("val_fp_per_hr", [0])[-1]
                    })
                    self.best_val_recall = self.history["val_recall"][-1]
                    self.best_val_accuracy = self.history["val_accuracy"][-1]

            if step_ndx == max_steps - 1:
                break


# Separate function to convert onnx models to tflite format
def convert_onnx_to_tflite(onnx_model_path, output_path):
    """Converts an ONNX version of an openwakeword model to the Tensorflow tflite format."""
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model, device="CPU")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tf_rep.export_graph(os.path.join(tmp_dir, "tf_model"))
        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(tmp_dir, "tf_model"))
        tflite_model = converter.convert()

        logging.info(f"####\nSaving tflite mode to '{output_path}'")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

    return None


if __name__ == '__main__':
    # Get training config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_config",
        help="The path to the training config file (required)",
        type=str,
        required=True
    )
    parser.add_argument(
        "--generate_clips",
        help="Execute the synthetic data generation process",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--augment_clips",
        help="Execute the synthetic data augmentation process",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing openwakeword features when the --augment_clips flag is used",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--train_model",
        help="Execute the model training process",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--convert_to_tflite",
        help="Convert the trained ONNX model to TFLite format",
        action="store_true",
        default="False",
        required=False
    )

    # ✅ Новый аргумент чисто для вывода
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="INFO (default) / WARNING / ERROR / DEBUG"
    )

    args = parser.parse_args()

    with open(args.training_config, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), yaml.Loader)

    # Resolve relative paths from config directory (portable)
    config = _resolve_config_paths(args.training_config, config)

    # ✅ Включаем красивый логгер
    setup_logging(args.log_level)

    stage("CONFIG")
    logging.info("model_name          : %s", config.get("model_name"))
    logging.info("target_phrase       : %s", config.get("target_phrase"))
    logging.info("output_dir          : %s", config.get("output_dir"))
    logging.info("positive_samples_dir: %s", config.get("positive_samples_dir"))
    logging.info("piper_voice_model   : %s", config.get("piper_voice_model"))
    logging.info("piper_executable    : %s", config.get("piper_executable"))
    logging.info("steps=%s | batch_n_per_class=%s | max_negative_weight=%s",
                 _human(config.get("steps", 0)), config.get("batch_n_per_class"), config.get("max_negative_weight"))

    scripts_path = os.path.abspath(config["piper_sample_generator_path"])
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    # ✅ Piper executable: portable resolution (Linux/Windows)
    import shutil
    _piper_exe = str(config.get("piper_executable", "")).strip()
    if _piper_exe:
        if os.path.isabs(_piper_exe):
            _resolved_piper = _piper_exe
        else:
            _resolved_piper = shutil.which(_piper_exe) or _piper_exe
        os.environ["PIPER_EXE"] = _resolved_piper
    try:
        scripts_path = os.path.abspath(config["piper_sample_generator_path"])
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        from generate_samples import generate_samples

        logging.info(f"generate_samples loaded from: {inspect.getfile(generate_samples)}")

    except ImportError as e:
        print(f"Ошибка: Не удалось найти файл generate_samples.py в {scripts_path}")
        print(f"Подробности: {e}")
        if args.generate_clips:
            sys.exit(1)

    # Define output locations
    config["output_dir"] = os.path.abspath(config["output_dir"])
    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])
    if not os.path.exists(os.path.join(config["output_dir"], config["model_name"])):
        os.mkdir(os.path.join(config["output_dir"], config["model_name"]))

    positive_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_train")
    positive_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_test")
    negative_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_train")
    negative_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_test")
    feature_save_dir = os.path.join(config["output_dir"], config["model_name"])

    stage("FOLDERS")
    logging.info("positive_train: %s", positive_train_output_dir)
    logging.info("positive_test : %s", positive_test_output_dir)
    logging.info("negative_train: %s", negative_train_output_dir)
    logging.info("negative_test : %s", negative_test_output_dir)
    logging.info("feature_save  : %s", feature_save_dir)

    # Get paths for impulse response and background audio files
    rir_paths = []
    for j in config["rir_paths"]:
        rir_paths.extend(_rglob_wavs(j))

    background_paths = []
    if len(config["background_paths_duplication_rate"]) != len(config["background_paths"]):
        config["background_paths_duplication_rate"] = [1] * len(config["background_paths"])

    for background_path, duplication_rate in zip(config["background_paths"],
                                                 config["background_paths_duplication_rate"]):
        wavs = _rglob_wavs(background_path)
        background_paths.extend(wavs * duplication_rate)

    stage("AUGMENT SOURCES")
    logging.info("RIR files            : %s", _human(len(rir_paths)))
    logging.info("Background wav files : %s", _human(len(background_paths)))
    logging.info("Background folders   : %s", config.get("background_paths"))

    if args.generate_clips is True:
        stage("STEP 1/4 — GENERATE CLIPS (PIPER)")

        # --------------------------
        # POSITIVE TRAIN
        # --------------------------
        os.makedirs(positive_train_output_dir, exist_ok=True)

        n_current_samples = len(os.listdir(positive_train_output_dir))
        logging.info("Positive TRAIN: existing=%s target=%s",
                     _human(n_current_samples), _human(config["n_samples"]))

        if n_current_samples <= 0.95 * config["n_samples"]:
            generate_samples(
                model=config["piper_voice_model"],
                text=config["target_phrase"],
                max_samples=config["n_samples"] - n_current_samples,
                batch_size=config["tts_batch_size"],
                noise_scales=[0.98],
                noise_scale_ws=[0.98],
                length_scales=[0.75, 1.0, 1.25],
                output_dir=positive_train_output_dir,
                auto_reduce_batch_size=True,
                file_names=[uuid.uuid4().hex + ".wav" for _ in range(config["n_samples"])]
            )
            torch.cuda.empty_cache()
            logging.info("DONE: Positive TRAIN generated -> %s", positive_train_output_dir)
        else:
            logging.warning("Skipping positive TRAIN generation (already enough samples)")

        stats = fix_wavs.fix_folder(positive_train_output_dir, target_sr=16000, backup=False, dry_run=False,
                                    delete_bak=True)
        logging.info("FIX_WAVS positive_train: %s", stats)

        # --------------------------
        # POSITIVE TEST
        # --------------------------
        os.makedirs(positive_test_output_dir, exist_ok=True)

        n_current_samples = len(os.listdir(positive_test_output_dir))
        logging.info("Positive TEST : existing=%s target=%s",
                     _human(n_current_samples), _human(config["n_samples_val"]))

        if n_current_samples <= 0.95 * config["n_samples_val"]:
            generate_samples(
                model=config["piper_voice_model"],
                text=config["target_phrase"],
                max_samples=config["n_samples_val"] - n_current_samples,
                batch_size=config["tts_batch_size"],
                noise_scales=[1.0],
                noise_scale_ws=[1.0],
                length_scales=[0.75, 1.0, 1.25],
                output_dir=positive_test_output_dir,
                auto_reduce_batch_size=True,
            )
            torch.cuda.empty_cache()
            logging.info("DONE: Positive TEST generated -> %s", positive_test_output_dir)
        else:
            logging.warning("Skipping positive TEST generation (already enough samples)")

        stats = fix_wavs.fix_folder(positive_test_output_dir, target_sr=16000, backup=False, dry_run=False,
                                    delete_bak=True)
        logging.info("FIX_WAVS positive_test: %s", stats)

        # --------------------------
        # NEGATIVE TRAIN
        # --------------------------
        os.makedirs(negative_train_output_dir, exist_ok=True)

        n_current_samples = len(os.listdir(negative_train_output_dir))
        logging.info("Negative TRAIN: existing=%s target=%s",
                     _human(n_current_samples), _human(config["n_samples"]))

        if n_current_samples <= 0.95 * config["n_samples"]:
            adversarial_texts = list(config.get("custom_negative_phrases", []))
            for target_phrase in config["target_phrase"]:
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=target_phrase,
                    N=config["n_samples"] // len(config["target_phrase"]),
                    include_partial_phrase=1.0,
                    include_input_words=0.2
                ))

            generate_samples(
                model=config["piper_voice_model"],
                text=adversarial_texts,
                max_samples=config["n_samples"] - n_current_samples,
                batch_size=max(1, config["tts_batch_size"] // 7),
                noise_scales=[0.98],
                noise_scale_ws=[0.98],
                length_scales=[0.75, 1.0, 1.25],
                output_dir=negative_train_output_dir,
                auto_reduce_batch_size=True,
                file_names=[uuid.uuid4().hex + ".wav" for _ in range(config["n_samples"])]
            )
            torch.cuda.empty_cache()
            logging.info("DONE: Negative TRAIN generated -> %s", negative_train_output_dir)
        else:
            logging.warning("Skipping negative TRAIN generation (already enough samples)")

        stats = fix_wavs.fix_folder(negative_train_output_dir, target_sr=16000, backup=False, dry_run=False,
                                    delete_bak=True)
        logging.info("FIX_WAVS negative_train: %s", stats)

        # --------------------------
        # NEGATIVE TEST
        # --------------------------
        os.makedirs(negative_test_output_dir, exist_ok=True)

        n_current_samples = len(os.listdir(negative_test_output_dir))
        logging.info("Negative TEST : existing=%s target=%s",
                     _human(n_current_samples), _human(config["n_samples_val"]))

        if n_current_samples <= 0.95 * config["n_samples_val"]:
            adversarial_texts = list(config.get("custom_negative_phrases", []))
            for target_phrase in config["target_phrase"]:
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=target_phrase,
                    N=config["n_samples_val"] // len(config["target_phrase"]),
                    include_partial_phrase=1.0,
                    include_input_words=0.2
                ))

            generate_samples(
                model=config["piper_voice_model"],
                text=adversarial_texts,
                max_samples=config["n_samples_val"] - n_current_samples,
                batch_size=max(1, config["tts_batch_size"] // 7),
                noise_scales=[1.0],
                noise_scale_ws=[1.0],
                length_scales=[0.75, 1.0, 1.25],
                output_dir=negative_test_output_dir,
                auto_reduce_batch_size=True,
            )
            torch.cuda.empty_cache()
            logging.info("DONE: Negative TEST generated -> %s", negative_test_output_dir)
        else:
            logging.warning("Skipping negative TEST generation (already enough samples)")

        stats = fix_wavs.fix_folder(negative_test_output_dir, target_sr=16000, backup=False, dry_run=False,
                                    delete_bak=True)
        logging.info("FIX_WAVS negative_test: %s", stats)

    # Set the total length of the training clips based on the ~median generated clip duration
    n = 50  # sample size
    positive_clips = [str(i) for i in Path(positive_test_output_dir).glob("*.wav")]
    duration_in_samples = []
    for i in range(n):
        # --- ИСПРАВЛЕННЫЙ БЛОК ПРОВЕРКИ ---
        positive_samples_dir = config["positive_samples_dir"]
        input_dir_path = Path(positive_samples_dir)

        positive_clips = [str(i) for i in input_dir_path.glob("*.wav")]
        valid_clips = [f for f in positive_clips if os.path.exists(f) and os.path.getsize(f) > 500]

        if len(valid_clips) == 0:
            logging.error("ОШИБКА: Скрипт не видит wav файлы в %s", positive_samples_dir)
            logging.error("Убедитесь, что вы положили .wav файлы именно в эту папку.")
            sys.exit(1)

        # Читаем один файл для проверки
        sr, dat = scipy.io.wavfile.read(valid_clips[np.random.randint(0, len(valid_clips))])
        duration_in_samples.append(len(dat))

    config["total_length"] = int(round(np.median(duration_in_samples) / 1000) * 1000) + 12000
    if config["total_length"] < 32000:
        config["total_length"] = 32000
    elif abs(config["total_length"] - 32000) <= 4000:
        config["total_length"] = 32000

    stage("CLIP LENGTH")
    logging.info("Positive samples found: %s wav (folder=%s)", _human(len(valid_clips)), positive_samples_dir)
    logging.info("total_length (samples): %s (~%.2f sec)", _human(config["total_length"]), config["total_length"] / 16000.0)

    # Do Data Augmentation
    if args.augment_clips is True:
        stage("STEP 2/4 — AUGMENT + FEATURES (.npy)")
        import shutil
        import random

        # 1. Очистка старых .npy, чтобы не мешались
        files_to_clean = [
            os.path.join(feature_save_dir, "positive_features_train.npy"),
            os.path.join(feature_save_dir, "negative_features_train.npy"),
            os.path.join(feature_save_dir, "positive_features_test.npy"),
            os.path.join(feature_save_dir, "negative_features_test.npy")
        ]
        for f in files_to_clean:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

        # 2. Копирование ТВОЕГО ГОЛОСА (Positive)
        source_files = [str(f) for f in Path(positive_samples_dir).glob("*.wav") if f.stat().st_size > 500]
        if not source_files:
            logging.error("ОШИБКА: Нет файлов в %s", positive_samples_dir)
            sys.exit(1)

        if len(os.listdir(positive_train_output_dir)) < 5:
            logging.info("Copying positives -> train/test (total=%s)", _human(len(source_files)))
            split_idx = int(len(source_files) * 0.8)
            for f in source_files[:split_idx]:
                shutil.copy(f, os.path.join(positive_train_output_dir, os.path.basename(f)))
            for f in source_files[split_idx:]:
                shutil.copy(f, os.path.join(positive_test_output_dir, os.path.basename(f)))

        # 3. Создание НЕГАТИВНЫХ ПРИМЕРОВ (Negative) - FIX: НЕ копируем POSITIVE в NEGATIVE
        if len(os.listdir(negative_train_output_dir)) < 5:
            logging.info("Negative folder empty -> creating negatives from background noise")

            # background_paths already contains paths to wav files
            bg_files = [
                p for p in background_paths
                if str(p).lower().endswith(".wav") and os.path.exists(p) and os.path.getsize(p) > 500
            ]

            if not bg_files:
                logging.error("ОШИБКА: Не найдены background wav файлы! Проверь config['background_paths']")
                sys.exit(1)

            selected_noise = random.sample(bg_files, min(len(bg_files), 100))
            for f in selected_noise:
                dest = os.path.join(negative_train_output_dir, "noise_" + os.path.basename(f))
                shutil.copy(f, dest)

            for f in selected_noise[:20]:
                dest = os.path.join(negative_test_output_dir, "noise_" + os.path.basename(f))
                shutil.copy(f, dest)

        # 4. Подготовка списков
        pos_train = [str(i) for i in Path(positive_train_output_dir).glob("*.wav")] * config["augmentation_rounds"]
        pos_test = [str(i) for i in Path(positive_test_output_dir).glob("*.wav")] * config["augmentation_rounds"]
        neg_train = [str(i) for i in Path(negative_train_output_dir).glob("*.wav")] * config["augmentation_rounds"]
        neg_test = [str(i) for i in Path(negative_test_output_dir).glob("*.wav")] * config["augmentation_rounds"]

        logging.info("POS train wav: %s", _human(len(pos_train)))
        logging.info("NEG train wav: %s", _human(len(neg_train)))
        logging.info("POS test  wav: %s", _human(len(pos_test)))
        logging.info("NEG test  wav: %s", _human(len(neg_test)))

        # 5. Функция запуска генерации + ФИКС WINDOWS LOCK
        def run_gen(clips, name, out_path, count):
            logging.info("FEATURES: %s -> %s (items=%s)", name, out_path, _human(count))
            if count == 0:
                logging.warning("Skip %s (empty)", name)
                return

            gen = augment_clips(
                clips,
                total_length=config["total_length"],
                batch_size=config["augmentation_batch_size"],
                background_clip_paths=background_paths,
                RIR_paths=rir_paths
            )

            compute_features_from_generator(
                gen,
                n_total=count,
                clip_duration=config["total_length"],
                output_file=out_path,
                device="cpu",
                ncpu=8
            )

            # Windows lock / weird tmp name fix
            bad_name_1 = out_path.replace(".npy", "2.npy")
            bad_name_2 = out_path.replace("train.npy", "trai2.npy")

            if not os.path.exists(out_path):
                if os.path.exists(bad_name_1):
                    logging.warning("Fix Windows lock rename: %s -> %s", bad_name_1, out_path)
                    try:
                        os.rename(bad_name_1, out_path)
                    except:
                        pass
                elif os.path.exists(bad_name_2):
                    logging.warning("Fix Windows lock rename: %s -> %s", bad_name_2, out_path)
                    try:
                        os.rename(bad_name_2, out_path)
                    except:
                        pass

            if os.path.exists(out_path):
                logging.info("OK: saved %s (%s bytes)", out_path, _human(os.path.getsize(out_path)))
            else:
                logging.warning("WARN: expected file not found: %s", out_path)

        n_cpus = 1

        run_gen(pos_train, "Positive Train", files_to_clean[0], len(pos_train))
        run_gen(neg_train, "Negative Train", files_to_clean[1], len(neg_train))
        run_gen(pos_test, "Positive Test", files_to_clean[2], len(pos_test))
        run_gen(neg_test, "Negative Test", files_to_clean[3], len(neg_test))

        logging.info("=== FEATURES GENERATION FINISHED ===")

        def create_gen(clips):
            return augment_clips(clips, total_length=config["total_length"],
                                 batch_size=config["augmentation_batch_size"],
                                 background_clip_paths=background_paths, RIR_paths=rir_paths)

            # 4. Расчет признаков (FEATURES) с исправленными аргументами
            n_cpus = (os.cpu_count() or 2) // 2
            device = "cpu"
            duration = config["total_length"]  # Длительность клипа из конфига

            print("--- Генерируем признаки (Positive Train) ---")
            compute_features_from_generator(create_gen(pos_train_files),
                                            n_total=len(pos_train_files),
                                            clip_duration=duration,
                                            output_file=files_to_clean[0],
                                            device=device, ncpu=n_cpus)

            print("--- Генерируем признаки (Negative Train) ---")
            n_neg_train = len(neg_train_files) if len(neg_train_files) > 0 else 1000
            compute_features_from_generator(create_gen(neg_train_files),
                                            n_total=n_neg_train,
                                            clip_duration=duration,
                                            output_file=files_to_clean[1],
                                            device=device, ncpu=n_cpus)

            print("--- Генерируем признаки (Positive Test) ---")
            compute_features_from_generator(create_gen(pos_test_files),
                                            n_total=len(pos_test_files),
                                            clip_duration=duration,
                                            output_file=files_to_clean[2],
                                            device=device, ncpu=n_cpus)

            print("--- Генерируем признаки (Negative Test) ---")
            n_neg_test = len(neg_test_files) if len(neg_test_files) > 0 else 500
            compute_features_from_generator(create_gen(neg_test_files),
                                            n_total=n_neg_test,
                                            clip_duration=duration,
                                            output_file=files_to_clean[3],
                                            device=device, ncpu=n_cpus)

    # Create openwakeword model
    if args.train_model is True:
        stage("STEP 3/4 — PREPARE TRAINING")

        def get_pos_label(x):
            return [1 for _ in x]

        def get_neg_label(x):
            return [0 for _ in x]

        F = openwakeword.utils.AudioFeatures(device='cpu')

        test_feat_path = os.path.join(feature_save_dir, "positive_features_test.npy")
        if not os.path.exists(test_feat_path):
            test_feat_path = os.path.join(feature_save_dir, "negative_features_test.npy")

        input_shape = np.load(test_feat_path).shape[1:]
        logging.info("input_shape: %s", input_shape)

        oww = Model(
            n_classes=1,
            input_shape=input_shape,
            model_type=config["model_type"],
            layer_dim=config["layer_size"],
            n_blocks=config.get("n_layers", 1),
            seconds_per_example=1280 * input_shape[0] / 16000
        )

        def f(x, n=input_shape[0]):
            """Simple transformation function to ensure negative data is the appropriate shape"""
            if n > x.shape[1] or n < x.shape[1]:
                x = np.vstack(x)
                new_batch = np.array([x[i:i + n, :] for i in range(0, x.shape[0] - n, n)])
            else:
                return x
            return new_batch

        data_transforms = {key: f for key in config["feature_data_files"].keys()}
        label_transforms = {}
        for key in ["positive"] + list(config["feature_data_files"].keys()) + ["adversarial_negative"]:
            if key == "positive":
                label_transforms[key] = get_pos_label
            else:
                label_transforms[key] = get_neg_label

        config["feature_data_files"]['positive'] = os.path.join(feature_save_dir, "positive_features_train.npy")
        config["feature_data_files"]['adversarial_negative'] = os.path.join(feature_save_dir, "negative_features_train.npy")

        n_per_class_dict = {key: config["batch_n_per_class"] for key in config["feature_data_files"].keys()}

        batch_generator = mmap_batch_generator(
            config["feature_data_files"],
            n_per_class=n_per_class_dict,
            data_transform_funcs=data_transforms,
            label_transform_funcs=label_transforms
        )

        class IterDataset(torch.utils.data.IterableDataset):
            def __init__(self, generator):
                self.generator = generator

            def __iter__(self):
                return self.generator

        X_train = torch.utils.data.DataLoader(
            IterDataset(batch_generator),
            batch_size=None,
            num_workers=0,
            prefetch_factor=None
        )

        # --- validation FP set ---
        fp_path = config.get("false_positive_validation_data_path", "")
        if fp_path and os.path.exists(fp_path) and os.path.getsize(fp_path) > 0:
            logging.info("Loading external validation data from %s", fp_path)
            X_val_fp = np.load(fp_path)
            X_val_fp = np.array(
                [X_val_fp[i:i + input_shape[0]] for i in range(0, X_val_fp.shape[0] - input_shape[0], 1)]
            )
        else:
            logging.warning("No valid external validation file. Using generated NEGATIVE TEST features.")
            neg_test_path = os.path.join(feature_save_dir, "negative_features_test.npy")
            if os.path.exists(neg_test_path) and os.path.getsize(neg_test_path) > 100:
                temp_neg = np.load(neg_test_path)
                limit = min(2000, temp_neg.shape[0])
                X_val_fp = temp_neg[:limit]
            else:
                X_val_fp = np.zeros((10, input_shape[0], input_shape[1])).astype(np.float32)

        X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
            batch_size=len(X_val_fp_labels) if len(X_val_fp_labels) > 0 else 1
        )

        X_val_pos = np.load(os.path.join(feature_save_dir, "positive_features_test.npy"))
        X_val_neg = np.load(os.path.join(feature_save_dir, "negative_features_test.npy"))
        labels = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)

        X_val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
                torch.from_numpy(labels)
            ),
            batch_size=len(labels)
        )

        # Run auto training
        best_model = oww.auto_train(
            X_train=X_train,
            X_val=X_val,
            false_positive_val_data=X_val_fp,
            steps=config["steps"],
            max_negative_weight=config["max_negative_weight"],
            target_fp_per_hour=config["target_false_positives_per_hour"],
        )

        stage("EXPORT")
        oww.export_model(model=best_model, model_name=config["model_name"], output_dir=config["output_dir"])

        if args.convert_to_tflite:
            stage("TFLITE CONVERT")
            convert_onnx_to_tflite(
                os.path.join(config["output_dir"], config["model_name"] + ".onnx"),
                os.path.join(config["output_dir"], config["model_name"] + ".tflite")
            )
