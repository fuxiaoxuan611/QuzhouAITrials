import math
from numbers import Number

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure, KVWriter


class CometOutputFormat(KVWriter):
    """Forward scalar metrics and figures dumped by the SB3 logger to Comet."""

    def __init__(self, experiment):
        self.experiment = experiment

    def write(self, key_values, key_excluded, step=0):
        metrics = {}
        for key, value in key_values.items():
            excluded = key_excluded.get(key)
            if excluded is not None and "comet" in excluded:
                continue
            if isinstance(value, Figure):
                self.experiment.log_figure(
                    figure_name=_figure_name(key, step),
                    figure=value.figure,
                    step=step,
                    metadata={"sb3_key": key},
                )
                continue
            scalar = _numeric_scalar(value)
            if scalar is not None:
                metrics[key] = scalar
        if metrics:
            self.experiment.log_metrics(metrics, step=step)

    def close(self):
        pass


class CometLoggerCallback(BaseCallback):
    """Attach Comet as an output format after SB3 initializes its logger."""

    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.output_format = CometOutputFormat(experiment)

    def _on_training_start(self):
        if not any(isinstance(output, CometOutputFormat) for output in self.logger.output_formats):
            # Log figures before TensorBoard handles their close=True flag.
            self.logger.output_formats.insert(0, self.output_format)

    def _on_step(self):
        return True


def _numeric_scalar(value):
    if isinstance(value, Number):
        scalar = value
    elif hasattr(value, "numel") and value.numel() == 1:
        scalar = value.item()
    elif hasattr(value, "size") and value.size == 1:
        scalar = value.item()
    else:
        return None

    try:
        return scalar if math.isfinite(float(scalar)) else None
    except (TypeError, ValueError):
        return None


def _figure_name(key, step):
    if step is None:
        return key
    return f"{key}-step-{step}"
