import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

from nps.models.base import BaseNeuralProcess

from ..data import BaseBatch
from ..experiment.forward_wrappers import ModelForwardWrapper, get_forward_wrapper


class BaseNeuralProcessPlotter(ABC):
    """A class to handle plotting of Neural Process model predictions."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    def _get_forward_wrapper(
        self, model: BaseNeuralProcess, batch: BaseBatch
    ) -> ModelForwardWrapper | None:
        """Get an appropriate forward wrapper for the model and batch.

        This method tries to get a forward wrapper from the registry
        based on model and batch types. If not found, a warning is
        printed and None is returned.

        Args:
            model: Neural Process model.
            batch: Batch instance.

        Returns:
            Forward wrapper function or None if not found.
        """
        # Try to get wrapper from registry
        wrapper = get_forward_wrapper(model, batch)
        if wrapper is None:
            warnings.warn(
                "Could not find a suitable forward wrapper "
                f"for model {model.__class__.__name__} "
                f"with batch type {batch.__class__.__name__}. "
                "Falling back to direct model calls."
            )
        return wrapper

    def _handle_figure_output(
        self,
        fig: plt.Figure,
        filename: str | Path,
        savefig: bool = False,
        logging: bool = True,
        show_plots: bool = False,
    ) -> None:
        """Handle figure output: log to wandb as PNG and/or save as PDF and/or show.

        Args:
            fig: Matplotlib figure to handle
            filename: Base filename/path for saving
            savefig: Whether to save the figure as PDF to disk
            logging: Whether to log the figure to wandb as PNG
            show_plots: Whether to display the figure
        """
        # Convert to Path for easier manipulation
        if isinstance(filename, str):
            filename = Path(filename)

        # Log to wandb as PNG if enabled and wandb.run exists.
        # Logging diagnostic figures is best-effort: wandb stages each image
        # as a PNG under $TMPDIR (node-local /tmp on SLURM) and then
        # shutil.move()s it into the run directory. When the two live on
        # different filesystems (node-local tmp vs networked scratch) the move
        # falls back to copy2, whose copystat() step can race with wandb's
        # async media handler and raise FileNotFoundError. A failure here must
        # never kill a multi-hour training run, so we catch, warn, and carry on.
        if wandb.run and logging:
            try:
                wandb.log({str(filename): wandb.Image(fig)})
            except Exception as e:
                warnings.warn(f"Failed to log figure '{filename}' to wandb: {e}")

        # Save as PDF if requested
        if savefig:
            # Ensure directory exists
            filename.parent.mkdir(parents=True, exist_ok=True)
            # Change extension to PDF
            pdf_filename = filename.with_suffix(".pdf")
            plt.savefig(pdf_filename, bbox_inches="tight", format="pdf")

        # Show plot if requested
        if show_plots:
            plt.show()

    @abstractmethod
    def __call__(
        self,
        model: BaseNeuralProcess,
        batches: list[BaseBatch],
        name: str = "plot",
        **kwargs,
    ) -> None:
        """
        Generate and display/save plots for given batches.

        Args:
            model (BaseNeuralProcess): The model used for predictions.
            batches (list[BaseBatch]): A list of data batches.
            name (str): Name for saving figures.
            **kwargs: Additional arguments to be passed to the model.
        """
        pass
