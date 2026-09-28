"""Workaround for a stable-baselines3 / torch checkpoint-loading bug.

``stable_baselines3.common.save_util.load_from_zip_file`` reads each stored
``.pth`` member with ``zipfile.ZipFile.open(...)`` and hands that stream
object straight to ``torch.load``. On this project's tested environment
(Windows, Python 3.12, torch 2.8/2.14 -- i.e. the whole torch range
stable-baselines3==2.9.0 supports) torch's C++ zip reader cannot parse that
``zipfile.ZipExtFile`` stream directly and raises:

    RuntimeError: PytorchStreamReader failed reading file
    .data/serialization_id: file read failed ...

even though the checkpoint itself is completely intact: reading the exact
same bytes through an in-memory ``io.BytesIO`` buffer first loads without
any error. This was verified directly (see README's "Known issue" note)
and is reproducible with every SB3/torch combination available at the time
of writing, so no dependency version bump fixes it, and it is not something
this project's own code can avoid (the bug is inside SB3's loader, which
every ``<Algorithm>.load(...)`` call goes through).

This module patches ``torch.load`` so that, if -- and only if -- it fails
with that exact error on a file-like object, it retries once by fully
reading that object into memory first. It never changes behaviour for the
common case (loading directly from a path, or from an environment where the
bug doesn't reproduce): the original ``torch.load`` is always tried first,
and this only intervenes after it has already raised the specific error
above. Importing ``src.agents`` applies this patch as a side effect, since
every script in this package that loads a trained model needs it.
"""

from __future__ import annotations

import io

import torch

_original_torch_load = torch.load


def _patched_torch_load(f, *args, **kwargs):
    try:
        return _original_torch_load(f, *args, **kwargs)
    except RuntimeError as exc:
        if "PytorchStreamReader" not in str(exc) or not hasattr(f, "read"):
            raise
        if hasattr(f, "seek"):
            try:
                f.seek(0)
            except (OSError, io.UnsupportedOperation):
                pass
        return _original_torch_load(io.BytesIO(f.read()), *args, **kwargs)


torch.load = _patched_torch_load
