import gc, hashlib, importlib.util, json, tracemalloc
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from horizon import codec
spec = importlib.util.spec_from_file_location('baseline', 'benchmarks/reference/codec_v1.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
report = {
    'scope': 'tracemalloc peak for one whole delta_zstd decode; input/container allocated before tracing; excludes untraced native allocations and is not process RSS',
    'repeats': 5,
    'baseline_sha256': hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
    'current_sha256': hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest(),
    'cases': [],
}
for name in ('N37E127', 'N46E007'):
    arr = np.load('data/terrain/' + name + '.npy', allow_pickle=False)
    blob = codec.encode(arr)
    entry = {'tile': name, 'shape': list(arr.shape), 'raw_bytes': arr.nbytes}
    for label, module in [('baseline', baseline), ('current', codec)]:
        peaks = []
        for _ in range(5):
            gc.collect()
            tracemalloc.start()
            out = module.decode(blob)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert np.array_equal(arr, out)
            peaks.append(peak)
            del out
        entry[label + '_peak_bytes'] = peaks
    report['cases'].append(entry)
with open('benchmarks/decode-memory-v2.json', 'x', encoding='utf-8') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
print(json.dumps(report, indent=2))
