"""Memory-stable DataLoader construction and lifecycle helpers."""

import ctypes
import os
import signal
import sys

import torch


def configure_worker_lifecycle(worker_id):
    """Keep a loader worker lightweight and terminate it with its parent."""
    del worker_id
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(0)
    except (ImportError, AttributeError):
        pass

    if not sys.platform.startswith('linux'):
        return
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux PR_SET_PDEATHSIG: ask the kernel to terminate an orphaned worker.
    if libc.prctl(1, signal.SIGTERM) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Close the race where the parent exits immediately before prctl().
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def build_loader_kwargs(workers, persistent_workers=False,
                        prefetch_factor=1, pin_memory=True):
    """Return valid DataLoader kwargs with bounded prefetched batches."""
    workers = int(workers)
    prefetch_factor = int(prefetch_factor)
    if workers < 0:
        raise ValueError('workers must be non-negative')
    if prefetch_factor < 1:
        raise ValueError('prefetch_factor must be positive')
    kwargs = {
        'num_workers': workers,
        'pin_memory': bool(pin_memory),
    }
    if workers > 0:
        kwargs.update({
            'persistent_workers': bool(persistent_workers),
            'prefetch_factor': prefetch_factor,
            'worker_init_fn': configure_worker_lifecycle,
        })
    return kwargs


def shutdown_data_loader(loader):
    """Eagerly stop persistent workers retained by a DataLoader."""
    if loader is None:
        return False
    iterator = getattr(loader, '_iterator', None)
    if iterator is None:
        return False
    shutdown = getattr(iterator, '_shutdown_workers', None)
    if callable(shutdown):
        shutdown()
    loader._iterator = None
    return True


def _process_children(pid):
    path = '/proc/{}/task/{}/children'.format(pid, pid)
    try:
        with open(path, 'r', encoding='ascii') as handle:
            return [int(value) for value in handle.read().split()]
    except (OSError, ValueError):
        return []


def _process_rss_bytes(pid):
    try:
        with open('/proc/{}/statm'.format(pid), 'r', encoding='ascii') as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf('SC_PAGE_SIZE')
    except (OSError, IndexError, ValueError):
        return 0


def _process_pss_bytes(pid):
    try:
        with open(
                '/proc/{}/smaps_rollup'.format(pid),
                'r', encoding='ascii') as handle:
            for line in handle:
                if line.startswith('Pss:'):
                    return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return _process_rss_bytes(pid)


def _process_tree_memory_bytes(root_pid, reader):
    pending = [root_pid]
    seen = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += reader(pid)
        pending.extend(_process_children(pid))
    return total


def process_tree_rss_mb(root_pid=None):
    """Measure current RSS of this process and all live loader workers."""
    root_pid = os.getpid() if root_pid is None else int(root_pid)
    return _process_tree_memory_bytes(
        root_pid, _process_rss_bytes) / (1024.0 ** 2)


def process_tree_pss_mb(root_pid=None):
    """Measure proportional memory without double-counting shared pages."""
    root_pid = os.getpid() if root_pid is None else int(root_pid)
    return _process_tree_memory_bytes(
        root_pid, _process_pss_bytes) / (1024.0 ** 2)
