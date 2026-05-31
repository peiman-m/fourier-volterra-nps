"""Lightning callbacks for the experiment loop.

``PlotterCallback`` owns the periodic plot side-effect (driven by
``misc.plot_fn`` / ``plot_interval`` / ``num_plots``);
``LogPerformanceCallback`` owns optional hardware-performance logging.
``lightning_wrapper.py`` owns only the training-loop module itself.
"""

from .performance import LogPerformanceCallback
from .plotter import PlotterCallback

__all__ = [
    "LogPerformanceCallback",
    "PlotterCallback",
]
