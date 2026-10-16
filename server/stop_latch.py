"""A software stop survives controller restarts until explicit resume.

Flag presence is the durable state; it contains no motor commands. set() always
latches the in-memory Event first and never raises an I/O error, so callers can
continue to hold immediately. clear() is reserved for a validated user resume:
failure or a newer stop request keeps the Event set and raises RuntimeError.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path


class PersistentStopEvent(threading.Event):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.persistence_error = ''
        self._state_lock = threading.RLock()
        self._disk_lock = threading.RLock()
        self._stop_version = 0
        try:
            self.path.lstat()  # Any presence, including a broken symlink, stops.
        except FileNotFoundError:
            pass
        except OSError as exc:
            super().set()
            self.persistence_error = str(exc)
        else:
            super().set()

    def _sync_directory(self):
        descriptor = os.open(self.path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _persist_stop(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self._sync_directory()

    def set(self):
        with self._state_lock:
            self._stop_version += 1
            super().set()
        with self._disk_lock:
            # An explicit later clear may win while this writer waits for I/O.
            if not self.is_set():
                return
            try:
                self._persist_stop()
                self.persistence_error = ''
            except OSError as exc:
                self.persistence_error = str(exc)

    def _erase_stop(self):
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        self._sync_directory()

    def clear(self):
        with self._state_lock:
            version = self._stop_version
        with self._disk_lock:
            try:
                with self._state_lock:
                    if version != self._stop_version:
                        raise RuntimeError('恢復期間收到新的停止要求，保持停止')
                self._erase_stop()
                with self._state_lock:
                    if version != self._stop_version:
                        raise RuntimeError('恢復期間收到新的停止要求，保持停止')
                    super().clear()
                self.persistence_error = ''
            except (OSError, RuntimeError) as exc:
                with self._state_lock:
                    super().set()
                # If unlink succeeded but fsync/new-stop detection failed,
                # recreate the flag before returning a failed resume.
                try:
                    self._persist_stop()
                except OSError:
                    pass
                self.persistence_error = str(exc)
                raise RuntimeError('無法解除已保存的停止狀態：' + str(exc)) from exc
