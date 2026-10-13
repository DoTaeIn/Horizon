'''Independent memory audit of Claude's cache classes; no cache implementation.

Fixed evidence replay, not the unfinished general-purpose Claude memory CLI.
Each probe runs one existing loader in a fresh interpreter, without a source
heightmap or other loader arms. Timing reports are produced separately.
'''
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizon.town_map import make_town_map

PROBE = r'''
import gc,hashlib,json,platform,resource,sys,tracemalloc
from horizon.town_cache import ChunkRegionCache,WholeMapResident
from horizon.town_region import load_region
req=json.load(sys.stdin)
def rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system()=='Darwin' else value*1024)
gc.collect();baseline=rss();digest=hashlib.sha256();obj=None
tracemalloc.start()
if req['arm']=='whole_resident':obj=WholeMapResident(req['whole'])
elif req['arm'].startswith('lru'):obj=ChunkRegionCache(req['chunked'],max_chunks=int(req['arm'][3:]))
startup_current,startup_peak=tracemalloc.get_traced_memory();startup_rss=rss()
tracemalloc.reset_peak()
retained_peak=0 if obj is None else obj.resident_bytes
output_peak=0;read_bytes=0 if obj is None else obj.startup_metrics['bytes_read']
hits=0;misses=0
for roi in req['rois']:
    if obj is None:out,metrics=load_region(req['chunked'],roi)
    else:out,metrics=obj.load(roi)
    assert out.shape==tuple(roi[2:]) and out.dtype.str=='<u2' and out.flags.owndata and out.flags.c_contiguous
    digest.update(memoryview(out))
    output_peak=max(output_peak,out.nbytes)
    retained_peak=max(retained_peak,0 if obj is None else obj.resident_bytes)
    read_bytes+=metrics['bytes_read'];hits+=metrics.get('cache_hits',0);misses+=metrics.get('cache_misses',metrics['chunks_loaded'])
    del out,metrics
current,movement_peak=tracemalloc.get_traced_memory();process_peak=rss()
retained_final=0 if obj is None else obj.resident_bytes
tracemalloc.stop()
if obj is not None:obj.close()
actual=digest.hexdigest();assert actual==req['expected_sha256'],(actual,req['expected_sha256'])
print(json.dumps({'output_sequence_sha256':actual,'resident_decoded_bytes_peak':retained_peak,'resident_decoded_bytes_final':retained_final,'one_output_bytes_peak':output_peak,'tracemalloc_startup_current_bytes':startup_current,'tracemalloc_startup_peak_bytes':startup_peak,'tracemalloc_end_current_bytes':current,'tracemalloc_movement_peak_bytes':movement_peak,'tracemalloc_overall_peak_bytes':max(startup_peak,movement_peak),'process_rss_baseline_hwm_bytes':baseline,'process_rss_startup_hwm_bytes':startup_rss,'process_rss_overall_hwm_bytes':process_peak,'bytes_read_including_startup':read_bytes,'cache_hits':hits,'cache_misses':misses}))
'''

def main():
    target=ROOT/'benchmarks/town-cache-memory-audit.json'
    if target.exists():raise FileExistsError(target)
    reports=[]
    for suffix in ('1024-v1','1024-seed2-v1'):
        timing_path=ROOT/f'benchmarks/town-cache-{suffix}.json'
        timing=json.loads(timing_path.read_text())
        schedule_path=ROOT/timing['settings']['schedule']
        schedule=json.loads(schedule_path.read_text())
        arr=make_town_map(size=timing['settings']['size'],seed=timing['settings']['seed'])
        assert hashlib.sha256(arr.tobytes()).hexdigest()==timing['map']['sha256']
        size=arr.shape[0];arms=['stateless','lru4','lru8','whole_resident']
        for case in timing['cases']:
            for kind,positions in schedule['trajectories'].items():
                rois=[[max(0,min(r-128,size-256)),max(0,min(c-128,size-256)),256,256] for r,c in positions]
                expected=hashlib.sha256()
                for r,c,h,w in rois:expected.update(memoryview(np.ascontiguousarray(arr[r:r+h,c:c+w])))
                for arm in arms:
                    samples=[]
                    for repeat in range(3):
                        req={'arm':arm,'whole':case['whole_path'],'chunked':case['chunked_path'],'rois':rois,'expected_sha256':expected.hexdigest()}
                        done=subprocess.run([sys.executable,'-c',PROBE],input=json.dumps(req),text=True,capture_output=True,cwd=ROOT,check=True)
                        value=json.loads(done.stdout);value['repeat']=repeat;samples.append(value)
                    reports.append({'seed':timing['settings']['seed'],'method':case['method'],'trajectory':kind,'arm':arm,'samples':samples})
                print('Memory audit',suffix,case['method'],kind,flush=True)
    report={'scope':'Independent audit probes of existing Claude loaders, fresh process per arm/path/codec/repetition; warm_buffered only, identical recorded paths; no source map/other arms in probe. Tracemalloc covers traced Python/NumPy allocations and small audit bookkeeping, excludes untraced native allocations. RSS is absolute process high-water mark including runtime/imports, not attributable cache bytes; baseline is also a high-water mark, not current RSS. Exact retained decoded bytes exclude index/objects/output/scratch. Only one output is kept and then deleted per step.','probe_count':len(reports)*3,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'timing_reports_sha256':{f'town-cache-{s}.json':hashlib.sha256((ROOT/f'benchmarks/town-cache-{s}.json').read_bytes()).hexdigest() for s in ('1024-v1','1024-seed2-v1')},'cases':reports}
    with target.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('Saved',target,'probes',report['probe_count'])

if __name__=='__main__':main()
