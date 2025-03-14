

# class AverageMeter():
#     """Computes and stores the average and current value."""

#     def __init__(self,
#                  name: str,
#                  fmt: str = ":f") -> None:
#         """Construct an AverageMeter module.

#         :param name: Name of the metric to be tracked.
#         :param fmt: Output format string.
#         """
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         """Reset internal states."""
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val: float, n: int = 1) -> None:
#         """Update internal states given new values.

#         :param val: New metric value.
#         :param n: Step size for update.
#         """
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count if self.count > 0 else 0

#     def __str__(self):
#         """Get string name of the object."""
#         fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
#         return fmtstr.format(**self.__dict__)


class AverageMeter:
    """Computes and stores the average and current value of a metric."""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        """Initialize an AverageMeter instance.

        Args:
            name (str): Name of the metric.
            fmt (str): Format for logging the metric.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all stored values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter with a new value.

        Args:
            val (float): New value.
            n (int): Number of samples.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self) -> str:
        """Return formatted metric string."""
        fmtstr = "{name} {val" + self.fmt + "} (Avg: {avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MultiMetricLogger:
    """Tracks multiple metrics, including global and personalized accuracy."""

    def __init__(self):
        self.meters = {}

    def update(self, metric_name: str, value: float, n: int = 1) -> None:
        """Update a specific metric.

        Args:
            metric_name (str): Name of the metric.
            value (float): New value.
            n (int): Number of samples.
        """
        if metric_name not in self.meters:
            self.meters[metric_name] = AverageMeter(metric_name)
        self.meters[metric_name].update(value, n)

    def log(self) -> None:
        """Log all stored metrics."""
        log_str = " | ".join(str(meter) for meter in self.meters.values())
        print(log_str)
