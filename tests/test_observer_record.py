"""File-only engineering recorder checks; no camera or robot connections."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.observer_record import record


class FileArrivalClock:
    def __init__(self,on_sleep=None):
        self.start_ns=1_790_000_000_000_000_000;self.elapsed_ns=0;self.on_sleep=on_sleep
    def time_ns(self):return self.start_ns+self.elapsed_ns
    def time(self):return self.time_ns()/1e9
    def monotonic(self):return 100.+self.elapsed_ns/1e9
    def sleep(self,seconds):
        self.elapsed_ns+=round(seconds*1e9)
        if self.on_sleep:self.on_sleep(self)
    def patch(self):
        return patch.multiple('scripts.observer_record.time',time=self.time,time_ns=self.time_ns,
                              monotonic=self.monotonic,sleep=self.sleep)


class ObserverRecordTests(unittest.TestCase):
    def test_live_file_has_bound_timestamps_hash_and_no_success_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);latest=root/'latest.jpg';output=root/'record'
            ok,data=cv2.imencode('.jpg',np.full((24,32,3),125,dtype=np.uint8))
            self.assertTrue(ok)
            clock=FileArrivalClock()
            # A file exists at startup, but only a later arrival may be archived.
            latest.write_bytes(data.tobytes())
            old_stamp=clock.start_ns-500_000_000
            os.utime(latest,ns=(old_stamp,old_stamp))
            def arrive(current):
                if current.elapsed_ns==100_000_000:
                    latest.write_bytes(data.tobytes())
                    os.utime(latest,ns=(current.time_ns(),current.time_ns()))
            clock.on_sleep=arrive
            with clock.patch(),contextlib.redirect_stdout(io.StringIO()):record(latest,output,.15,10)
            meta=json.loads((output/'session.json').read_text())
            rows=[json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);entry=rows[0]
            self.assertEqual(entry['image_size'],[32,24])
            self.assertEqual(entry['sha256'],hashlib.sha256((output/entry['file']).read_bytes()).hexdigest())
            self.assertGreaterEqual(entry['host_monotonic'],meta['started_monotonic'])
            self.assertEqual(meta['started_wall_time_ns'],clock.start_ns)
            self.assertEqual(entry['file_mtime_ns'],clock.start_ns+100_000_000)
            self.assertGreaterEqual(entry['file_mtime_ns'],meta['started_wall_time_ns'])
            self.assertEqual(meta['source'],'engineering_observer')
            self.assertFalse(meta['production_eligible']);self.assertFalse(meta['physical_success_verified'])
            self.assertIn(entry['file'],(output/'video.ffconcat').read_text())

    def test_pre_session_cache_under_one_second_old_is_not_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);latest=root/'latest.jpg';output=root/'record'
            cv2.imwrite(str(latest),np.zeros((24,32,3),dtype=np.uint8))
            clock=FileArrivalClock();stamp=clock.start_ns-500_000_000
            os.utime(latest,ns=(stamp,stamp))
            with clock.patch(),contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'No fresh'):
                record(latest,output,.15,10)
            self.assertLess(clock.time_ns()-stamp,1_000_000_000)
            self.assertEqual((output/'frames.jsonl').read_text(),'')
            self.assertEqual(list(output.glob('frame-*.jpg')),[])
            meta=json.loads((output/'session.json').read_text())
            self.assertEqual(meta['started_wall_time_ns'],clock.start_ns)
            self.assertEqual(meta['frame_count'],0)

    def test_stale_cache_is_not_recorded_as_a_new_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);latest=root/'latest.jpg';output=root/'record'
            cv2.imwrite(str(latest),np.zeros((24,32,3),dtype=np.uint8))
            old=time.time()-60;os.utime(latest,(old,old))
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'No fresh'):
                record(latest,output,.05,10)
            self.assertEqual((output/'frames.jsonl').read_text(),'')

    def test_existing_evidence_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(FileExistsError):record(root/'latest.jpg',root,1)


if __name__=='__main__':unittest.main()
