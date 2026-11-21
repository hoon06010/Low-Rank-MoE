"""Offline tests for the private reserved-invocation CUDA worker."""
import importlib.util
from pathlib import Path
import tempfile
import json
import os
import unittest
from unittest.mock import patch

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('continuation_worker',HERE/'continuation_worker.py')
worker=importlib.util.module_from_spec(spec)
if Path(spec.origin).exists(): spec.loader.exec_module(worker)

class WorkerTests(unittest.TestCase):
    def test_worker_requires_reserved_parent_envelope(self):
        self.assertTrue(callable(getattr(worker,'validate_reservation',None)),'reserved continuation worker missing')

    def test_mismatched_parent_or_expired_reservation_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'reservation.json'
            envelope={'status':'reserved','id':'i','parent_pid':os.getppid()+1,'expires_at':1000,
                      'spec':{'seed':401,'phase':'recovery'},'run_id':'same'}
            path.write_text(json.dumps(envelope))
            with patch.dict(os.environ,{'SEED401_RESERVED_INVOCATION':'i'}):
                with self.assertRaisesRegex(ValueError,'reservation'):
                    worker.validate_reservation(path,Path(tmp)/'branch',now=100)

if __name__=='__main__': unittest.main()
