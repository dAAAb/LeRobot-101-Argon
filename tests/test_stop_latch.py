"""Persistent stop tests use temporary files and fake robot dependencies only."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from server.stop_latch import PersistentStopEvent


class StopLatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'software-stop'
        self.latch = PersistentStopEvent(self.path)

    def test_fresh_project_starts_unset_and_stop_survives_reconstruction(self):
        self.assertFalse(self.latch.is_set())
        self.assertFalse(self.latch.wait(.001))
        self.latch.set()
        self.assertTrue(self.latch.wait(0))
        self.assertTrue(self.path.exists())
        replacement = PersistentStopEvent(self.path)
        self.assertTrue(replacement.is_set())

    def test_repeated_set_is_idempotent_and_explicit_clear_survives_restart(self):
        self.latch.set()
        inode = self.path.stat().st_ino
        self.latch.set()
        self.assertEqual(self.path.stat().st_ino, inode)
        self.latch.clear()
        self.assertFalse(self.latch.is_set())
        self.assertFalse(self.path.exists())
        self.assertFalse(PersistentStopEvent(self.path).is_set())
        self.latch.clear()

    def test_set_latches_before_slow_io_and_write_failure_does_not_block_hold(self):
        latched = []
        def fail():
            latched.append(self.latch.is_set())
            raise PermissionError('read-only filesystem')
        held = []
        with patch.object(self.latch, '_persist_stop', side_effect=fail):
            self.latch.set()
            held.append(True)  # The caller's next hold operation still runs.
        self.assertEqual(latched, [True])
        self.assertEqual(held, [True])
        self.assertTrue(self.latch.is_set())
        self.assertIn('read-only', self.latch.persistence_error)
        self.latch.set()  # Persistence can recover on a later explicit stop.
        self.assertTrue(PersistentStopEvent(self.path).is_set())

    def test_set_wakes_waiters_even_while_disk_lock_is_busy(self):
        with self.latch._disk_lock:
            worker = threading.Thread(target=self.latch.set)
            worker.start()
            self.assertTrue(self.latch.wait(.5))
            self.assertTrue(worker.is_alive())
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(self.path.exists())

    def test_unlink_failure_keeps_stop_in_memory_and_across_restart(self):
        self.latch.set()
        with patch.object(Path, 'unlink', side_effect=PermissionError('cannot erase flag')):
            with self.assertRaisesRegex(RuntimeError, '無法解除'):
                self.latch.clear()
        self.assertTrue(self.latch.is_set())
        self.assertTrue(PersistentStopEvent(self.path).is_set())

    def test_failed_directory_sync_after_unlink_recreates_stop(self):
        self.latch.set()
        with patch.object(self.latch, '_sync_directory', side_effect=OSError('fsync failed')):
            with self.assertRaisesRegex(RuntimeError, 'fsync failed'):
                self.latch.clear()
        self.assertTrue(self.latch.is_set())
        self.assertTrue(self.path.exists())
        self.assertTrue(PersistentStopEvent(self.path).is_set())

    def test_new_stop_during_clear_wins_and_is_not_erased(self):
        self.latch.set()
        erase = self.latch._erase_stop
        def stop_during_erase():
            erase()
            self.latch.set()
        with patch.object(self.latch, '_erase_stop', side_effect=stop_during_erase):
            with self.assertRaisesRegex(RuntimeError, '新的停止要求'):
                self.latch.clear()
        self.assertTrue(self.latch.is_set())
        self.assertTrue(PersistentStopEvent(self.path).is_set())

    def test_flag_presence_and_unreadable_startup_fail_closed(self):
        self.path.symlink_to('missing-target')
        self.assertTrue(PersistentStopEvent(self.path).is_set())
        with patch.object(Path, 'lstat', side_effect=PermissionError('cannot inspect stop flag')):
            stopped = PersistentStopEvent(self.path)
        self.assertTrue(stopped.is_set())
        self.assertIn('cannot inspect', stopped.persistence_error)

    def test_robot_constructor_adopts_latch_without_connecting_and_only_valid_resume_clears(self):
        from server.robot import Robot
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'config/robot.json').read_text())
        calibration = json.loads((root/'config/follower-calibration.json').read_text())
        self.latch.set()
        with patch.object(Robot, 'connect', side_effect=AssertionError('no hardware')):
            robot = Robot(config, calibration, self.temp.name, Mock())
            self.assertIsInstance(robot.stop, PersistentStopEvent)
            self.assertTrue(robot.stop.is_set())
            with patch.object(robot.gripper_power, 'assert_motion_allowed', side_effect=RuntimeError('pending contact')):
                with self.assertRaisesRegex(RuntimeError, 'pending contact'):
                    robot.resume()
            self.assertTrue(self.path.exists())
            self.assertTrue(robot.stop.is_set())
            robot.resume()  # Explicit, successful existing resume validation.
        self.assertFalse(robot.stop.is_set())
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
