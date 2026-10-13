"""Independent audit of final CLI raw timings, LRU transitions and memory evidence."""
import collections
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stats(values):
    a = np.asarray(values) / 1e6
    return {k: round(float(v), 6) for k, v in zip(('median_ms','p95_ms','max_ms'),(np.median(a),np.percentile(a,95,method='linear'),max(a)))} if len(a) else None

def audit(path):
    r=json.loads(path.read_text()); s=json.loads((ROOT/r['settings']['schedule']).read_text())
    assert digest(ROOT/r['settings']['schedule'])==r['verification']['schedule_sha256']
    for p,h in {**s['source_files_sha256'],**r['source_files_sha256']}.items(): assert digest(ROOT/p)==h,p
    assert r['map']['sha256']==s['map']['sha256']
    arms=['stateless']+[f'lru{n}' for n in r['settings']['cache_sizes']]+['whole_resident']
    size=r['settings']['size']; viewport=r['settings']['viewport']; cells=[]; loads=0
    for case in r['cases']:
        data=Path(case['chunked_path']).read_bytes(); whole=Path(case['whole_path']).read_bytes()
        for k,b in [('whole',whole),('chunked',data)]: assert hashlib.sha256(b).hexdigest()==case['hashes_before'][k]==case['hashes_after'][k]==case['hashes'][k+'_sha256']
        _,_,_,_,cs,rows,cols,count=struct.unpack_from('<4sBBHIIII',data)
        entries=[struct.unpack_from('<IIIIQQ',data,24+i*32) for i in range(count)];meta=24+count*32
        coords=lambda indices:[[i//cols,i%cols] for i in indices]
        for mode,kinds in case['modes'].items():
            for ki,(kind,run) in enumerate(kinds.items()):
                flat={a:[] for a in arms};starts={a:[] for a in arms};totals={a:[] for a in arms};first={a:[] for a in arms};movement={a:[] for a in arms}
                assert len(run['repeats'])==r['settings']['repeats']
                for ri,rep in enumerate(run['repeats']):
                    rot=(ki+ri)%len(arms); assert rep['construct_order']==arms[rot:]+arms[:rot]
                    lrus={a:collections.OrderedDict() for a in arms if a.startswith('lru')}
                    for a,m in rep['startup_metrics'].items():
                        assert m['total_ns']==m['read_ns']+m['decode_ns']
                        expected=0 if a=='stateless' else len(whole) if a=='whole_resident' else meta
                        assert m['bytes_read']==m['bytes_requested']==expected
                        starts[a].append(m)
                    assert len(rep['frames'])==r['settings']['steps']
                    for fi,f in enumerate(rep['frames']):
                        center=s['trajectories'][kind][fi];roi=[max(0,min(x-viewport//2,size-viewport)) for x in center]+[viewport,viewport];assert f['roi']==roi
                        rot=(ki+ri+fi)%len(arms);assert f['arm_order']==arms[rot:]+arms[:rot]
                        rr,cc,h,w=roi;selected=[i for i,e in enumerate(entries) if e[0]<rr+h and e[0]+e[2]>rr and e[1]<cc+w and e[1]+e[3]>cc]
                        for a,m in f['arms'].items():
                            assert m['total_ns']==m['read_ns']+m['decode_ns'] and min(m['read_ns'],m['decode_ns'])>=0
                            if a.startswith('lru'):
                                lru=lrus[a];hits=[i for i in selected if i in lru];misses=[i for i in selected if i not in lru];evicted=[]
                                for i in hits:lru.move_to_end(i)
                                for i in misses:
                                    if len(lru)==int(a[3:]):evicted.append(lru.popitem(last=False)[0])
                                    lru[i]=True
                                assert m['hit_coordinates']==coords(hits) and m['miss_coordinates']==coords(misses) and m['evicted_coordinates']==coords(evicted)
                                assert m['cache_hits']==len(hits) and m['cache_misses']==m['chunks_loaded']==len(misses) and m['evictions']==len(evicted)
                                assert m['chunk_coordinates']==coords(selected) and m['chunks_requested']==len(selected)
                                expected=sum(entries[i][5] for i in misses);calls=len(misses);resident=sum(entries[i][2]*entries[i][3]*2 for i in lru)
                            elif a=='stateless':
                                expected=meta+sum(entries[i][5] for i in selected);calls=2+len(selected);resident=0
                                assert m['cache_misses']==len(selected) and m['cache_hits']==0
                            else: expected=0;calls=0;resident=size*size*2
                            assert m['bytes_read']==m['bytes_requested']==expected and m['read_calls']==m['seek_calls']==calls
                            assert m['resident_bytes']==resident
                            flat[a].append(m);loads+=1
                    for a in arms:
                        ns=[f['arms'][a]['total_ns'] for f in rep['frames']];start=rep['startup_metrics'][a]['total_ns']
                        assert rep['trajectory_ns'][a]=={'startup_total_ns':start,'frames_total_ns':sum(ns),'total_ns_including_startup':start+sum(ns)}
                        totals[a].append(start+sum(ns));first[a].append(start+ns[0]);movement[a].append(sum(ns))
                summary={'method':case['method'],'mode':mode,'trajectory':kind,'arms':{}}
                for a,frames in flat.items():
                    agg=run['aggregate'][a];hit=[m['total_ns'] for m in frames if m['cache_misses']==0];miss=[m['total_ns'] for m in frames if m['cache_misses']>0]
                    for k,v in [('frame_total',[m['total_ns'] for m in frames]),('startup_total',[m['total_ns'] for m in starts[a]]),('first_viewport',first[a]),('trajectory_total_including_startup',totals[a]),('trajectory_total_movement_only',movement[a]),('frame_total_hit_only',hit),('frame_total_miss_involving',miss)]:assert agg[k]==stats(v),(a,k)
                    requested=sum(m['chunks_requested'] for m in frames);rate=sum(m['cache_hits'] for m in frames)/requested if requested else None
                    assert agg['hit_rate']==rate and agg['evictions_total']==sum(m['evictions'] for m in frames)
                    assert agg['bytes_read_total_including_startup']==sum(m['bytes_read'] for m in frames)+sum(m['bytes_read'] for m in starts[a])
                    assert agg['resident_bytes_max']==max(m['resident_bytes'] for m in frames) and agg['resident_bytes_end']==frames[-1]['resident_bytes']
                    summary['arms'][a]=agg
                cells.append(summary)
    memory=r['memory'];assert digest(memory['output_path'])==memory['output_sha256'];mr=memory['report'];assert mr==json.loads(Path(memory['output_path']).read_text())
    assert mr['map']['sha256']==r['map']['sha256']
    for p,h in mr['source_files_sha256'].items():assert digest(ROOT/p)==h,p
    import sys
    sys.path.insert(0,str(ROOT))
    from horizon.town_map import make_town_map
    arr=make_town_map(size=size,seed=r['settings']['seed']);assert hashlib.sha256(arr.tobytes()).hexdigest()==r['map']['sha256']
    probes=0
    for c in mr['cases']:
        cell=next(x for x in cells if x['method']==c['method'] and x['trajectory']==c['trajectory'] and x['mode']=='warm_buffered');arm=cell['arms'][c['arm']]
        expected=hashlib.sha256()
        for center in s['trajectories'][c['trajectory']]:
            rr,cc=[max(0,min(x-viewport//2,size-viewport)) for x in center];expected.update(memoryview(np.ascontiguousarray(arr[rr:rr+viewport,cc:cc+viewport])))
        assert len(c['samples'])==r['settings']['memory_repeats']
        for m in c['samples']:
            assert m['output_sequence_sha256']==expected.hexdigest()
            assert m['resident_decoded_bytes_peak']==arm['resident_bytes_max'] and m['resident_decoded_bytes_end']==arm['resident_bytes_end']
            assert m['one_output_bytes_peak']==viewport*viewport*2
            assert m['bytes_read_including_startup']*r['settings']['repeats']==arm['bytes_read_total_including_startup']
            assert m['tracemalloc_overall_peak_bytes']>=m['tracemalloc_movement_current_bytes']>=m['resident_decoded_bytes_end']
            if m['process_rss_overall_hwm_bytes'] is not None:assert m['process_rss_overall_hwm_bytes']>=m['process_rss_startup_hwm_bytes']>=m['process_rss_baseline_hwm_bytes']
            probes+=1
    return {'file':path.name,'sha256':digest(path),'seed':r['settings']['seed'],'loads':loads,'memory_probes':probes,'cells':cells}

def main():
    reports=[audit(ROOT/f'benchmarks/town-cache-{suffix}-v2.json') for suffix in ('1024','1024-seed2')]
    out={'scope':'Independent raw timing/LRU/byte/accounting recomputation and memory output-sequence hashes; same original recorded paths.', 'tests':{'total':381,'passed':376,'skipped':5}, 'auditor_source_sha256':digest(__file__), 'reports':reports}
    (ROOT/'benchmarks/town-cache-final-evidence.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Verified',sum(r['loads'] for r in reports),'loads and',sum(r['memory_probes'] for r in reports),'fresh-process memory probes')
if __name__=='__main__':main()
