"""Run: python -m unittest discover -s tests -p test_submission.py -v"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import process_directory
from src.final_dsp import enhance


class SubmissionIO(unittest.TestCase):
    def test_cli_from_other_directory_preserves_metadata(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / 'input', root / 'output'
            source.mkdir()
            cases = [(8000, 1, 17), (16000, 2, 4097), (22050, 1, 5001),
                     (44100, 2, 7003), (48000, 1, 1), (16000, 1, 0),
                     (16000, 1, 48000), (16000, 1, 1023)]
            for i, (rate, channels, frames) in enumerate(cases):
                x = rng.normal(0, .03, (frames, channels))
                if i == 6:
                    x[:] = 0
                sf.write(source / f'한글_{i}.WAV', x, rate, subtype='PCM_24')
            (source / 'ignore.txt').write_text('Not audio')
            subprocess.run([sys.executable, str(ROOT / 'main.py'), '--input', str(source),
                            '--output', str(destination), '--workers', '2'], cwd=root, check=True)
            self.assertEqual({p.name for p in destination.iterdir()},
                             {p.name for p in source.glob('*.WAV')})
            for path in source.glob('*.WAV'):
                before, after = sf.info(path), sf.info(destination / path.name)
                self.assertEqual((before.frames, before.channels, before.samplerate),
                                 (after.frames, after.channels, after.samplerate))
                self.assertEqual(after.subtype, 'PCM_16')
                self.assertTrue(np.isfinite(sf.read(destination / path.name)[0]).all())

    def test_reject_same_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            sf.write(p / 'x.wav', np.zeros(100), 16000)
            before = (p / 'x.wav').read_bytes()
            with self.assertRaises(ValueError):
                process_directory(p, p)
            self.assertEqual(before, (p / 'x.wav').read_bytes())

    def test_invalid_audio_and_empty_input(self):
        with self.assertRaises(ValueError):
            enhance(np.array([np.nan]), 16000)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                process_directory(directory, Path(directory) / 'out')

    def test_archived_arithmetic_and_no_evaluation_imports(self):
        x = np.random.default_rng(9).normal(0, .03, 9000)
        actual = enhance(x)
        self.assertNotIn('bench', sys.modules)
        self.assertNotIn('pandas', sys.modules)
        self.assertNotIn('pesq', sys.modules)
        spec = importlib.util.spec_from_file_location('archived_depth',
                    ROOT / 'experiments/dsp_20261117/depth_round/experiment.py')
        archived = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(archived)
        np.testing.assert_array_equal(actual, archived.estimate(x, '12_1500'))


if __name__ == '__main__':
    unittest.main()
