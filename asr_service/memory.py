"""RAM guard for isolated model processes."""
import os
import time
import psutil


class MemorySafetyError(RuntimeError):
    pass


def check_memory(headroom=None):
    threshold = float(headroom if headroom is not None else os.environ.get('ASR_RAM_HEADROOM_GB', '3')) * 2**30
    if threshold <= 0:
        raise ValueError('RAM headroom must be positive')
    available = psutil.virtual_memory().available
    if available < threshold:
        raise MemorySafetyError(f'RAM safety abort: available={available/2**30:.2f} GiB; required={threshold/2**30:.2f} GiB')
    return available


def supervise(process, progress, interval=1):
    started, peak, last = time.monotonic(), 0, 0
    try:
        while process.poll() is None:
            if time.monotonic()-started > float(os.environ.get('ASR_CPU_MAX_SECONDS','10800')):
                raise RuntimeError('CPU model runtime deadline exceeded; process stopped')
            check_memory()
            try:
                root = psutil.Process(process.pid)
                peak = max(peak, sum(p.memory_info().rss for p in [root, *root.children(recursive=True)] if p.is_running()))
            except psutil.NoSuchProcess:
                pass
            if time.monotonic()-last >= 5:
                progress(30, f'cpu_running elapsed={time.monotonic()-started:.1f}s peak_ram_gib={peak/2**30:.3f}')
                last = time.monotonic()
            time.sleep(interval)
    except BaseException as exc:
        try:
            root = psutil.Process(process.pid)
            for child in root.children(recursive=True): child.kill()
            root.kill()
        except psutil.NoSuchProcess:
            pass
        process.wait(timeout=10)
        exc.metrics = {'peak_process_tree_rss_bytes':peak,'elapsed_seconds':time.monotonic()-started,'exit_code':process.returncode,'memory_guard_aborted':True}
        raise
    return {'peak_process_tree_rss_bytes': peak, 'elapsed_seconds': time.monotonic()-started, 'exit_code': process.returncode}
