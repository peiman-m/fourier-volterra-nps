import os
from pathlib import Path

import pandas as pd
import torch

from ..base import BaseIterableDataset
from .sim_dataset import PredPreyBatch

_LYNXHARE_URL = "http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt"
_LYNXHARE_FILENAME = "LynxHare.txt"


def _download_if_missing(path: Path) -> None:
    """Download LynxHare.txt to `path` if it does not already exist.

    Wraps the HTTP fetch so that any partial file is deleted on failure before
    re-raising, preventing a corrupt file from being silently reused on the next run.
    """
    if path.exists():
        return

    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_LYNXHARE_URL, path)
    except Exception as exc:
        if path.exists():
            path.unlink()
        raise RuntimeError(
            f"Failed to download LynxHare.txt from {_LYNXHARE_URL}. "
            f"Place the file manually at {path} and retry."
        ) from exc


class PredPreyRealDataset(BaseIterableDataset):
    """
    Hudson Bay hare-lynx real data for zero-shot sim-to-real evaluation.

    Loads the 91-point annual time series (years 1845–1935), rescales time and
    population, and yields `PredPreyBatch` instances whose context/query splits
    are random partitions of the full trajectory.  Because both species are always
    present, all 91 points appear either as context or query in every batch element.

    The dataset produces the same `PredPreyBatch` type as `PredPreySimDataset`, so
    it works with the existing forward wrappers, metric wrappers, and plotter with
    zero additional registration.

    Args:
        min_nc: Minimum number of context points per batch element.
        max_nc: Maximum number of context points per batch element.
            Must satisfy ``1 <= min_nc <= max_nc < T`` (where T = 91) so that at
            least one query point always remains.
        population_rescale_factor: Multiplicative scale applied to both populations.
        time_rescale_factor: Multiplicative scale applied to the normalised year axis
            (years since 1845).  With the default 0.1 the rescaled range is 0–9.
        data_path: Directory in which ``LynxHare.txt`` is cached.  Downloaded
            automatically on first use; subsequent runs load from the cache.
        **kwargs: Forwarded to ``BaseIterableDataset`` (``samples_per_epoch``,
            ``batch_size``, ``deterministic``, ``deterministic_seed``, ``drop_last``).
    """

    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        population_rescale_factor: float = 0.01,
        time_rescale_factor: float = 0.1,
        data_path: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # ── Download / load ──────────────────────────────────────────────────
        data_dir = Path(data_path)
        cache_path = data_dir / _LYNXHARE_FILENAME

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            _download_if_missing(cache_path)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # delim_whitespace=True is deprecated since pandas 2.0; use sep=r'\s+'.
        # header=None + index_col=0: year becomes the index, columns are labelled [1, 2]
        # for hare and lynx respectively.
        df = pd.read_csv(cache_path, header=None, sep=r'\s+', index_col=0)

        years = df.index.to_numpy(dtype=float)
        populations = df[[1, 2]].to_numpy(dtype=float)  # [T, 2]: hare, lynx

        T = len(years)

        # ── Validate ─────────────────────────────────────────────────────────
        # Cast to int first so all comparisons use the values that will actually
        # be stored (avoids edge cases with float inputs like min_nc=1.9).
        self.min_nc = int(min_nc)
        self.max_nc = int(max_nc)

        if self.min_nc < 1:
            raise ValueError(f"min_nc must be >= 1, got {self.min_nc}")
        if self.min_nc > self.max_nc:
            raise ValueError(f"min_nc ({self.min_nc}) must be <= max_nc ({self.max_nc})")
        if self.max_nc >= T:
            raise ValueError(
                f"max_nc ({self.max_nc}) must be < T ({T}); otherwise nc could equal T "
                "leaving zero query points."
            )

        # ── Rescale and store ─────────────────────────────────────────────────
        times = (years - years.min()) * time_rescale_factor      # normalise then scale
        pops = populations * population_rescale_factor

        # _times: [T, 1]  — shape required for advanced indexing to yield [B, n, 1]
        # _populations: [T, 2]
        self._times = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
        self._populations = torch.tensor(pops, dtype=torch.float32)

        print(
            f"[{type(self).__name__}] "
            f"T={T}, nc=[{min_nc}, {max_nc}], "
            f"time_range=[{times.min():.1f}, {times.max():.1f}]"
        )

    def generate_batch(self) -> PredPreyBatch:
        # nc is sampled once per batch; all B elements share the same nc so that
        # nq = T - nc is identical across elements and tensors stack without padding.
        nc = torch.randint(self.min_nc, self.max_nc + 1, ()).item()  # Python int, required
        T = self._times.shape[0]
        B = self.batch_size

        idc = torch.stack([torch.randperm(T)[:nc] for _ in range(B)])  # [B, nc]

        # Query (held-out): complement of context — nq = T - nc, same for every element.
        # pin device to match self._times so torch.isin doesn't hit a CPU/GPU mismatch.
        dev = self._times.device
        idq = torch.stack([
            torch.where(~torch.isin(torch.arange(T, device=dev), idc[b]))[0]
            for b in range(B)
        ])  # [B, nq]

        xc = self._times[idc]        # [B, nc, 1]
        yc = self._populations[idc]  # [B, nc, 2]
        xq = self._times[idq]        # [B, nq, 1]
        yq = self._populations[idq]  # [B, nq, 2]

        return PredPreyBatch(
            x=torch.cat([xc, xq], dim=1),
            y=torch.cat([yc, yq], dim=1),
            xc=xc,
            yc=yc,
            xq=xq,
            yq=yq,
            # .clone() is required: .expand() returns a view sharing memory with
            # self._times / self._populations.  An in-place op on the returned batch
            # would corrupt stored tensors and break all subsequent batches.
            x_dense=self._times.unsqueeze(0).expand(B, -1, -1).clone(),
            y_dense=self._populations.unsqueeze(0).expand(B, -1, -1).clone(),
        )
