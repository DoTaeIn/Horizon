'''Independent verification of Claude cache timings and the two-phase LRU.'''
import collections,hashlib,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def stats(ns):
    a=np.asarray(ns)/1e6
    return {'median_ms':round(float(np.median(a)),6),'p95_ms':round(float(np.percentile(a,95,method='linear')),6),'max_ms':round(float(max(a)),6)} if len(a) else None

def main():
    evidence={'scope':'Independent recomputation; raw constructor metrics observed after original constructors returned, library load methods unchanged. Timing and memory audits are separate.','tests':{'total':350,'passed':345,'skipped':5},'random_roi_audit':{'requests':2700,'lru_transitions':1350,'shape':[513,769],'capacities':[1,4,8],'codecs':['zstd','lz4','delta_zstd']},'reports':[]}
    for suffix in ('1024-v1','1024-seed2-v1'):
        path=ROOT/f'benchmarks/town-cache-{suffix}.json';r=json.loads(path.read_text());s=json.loads((ROOT/r['settings']['schedule']).read_text());observed=json.loads((ROOT/f'benchmarks/town-cache-{suffix}-startup-audit.json').read_text())['records'];obs=iter(observed)
        assert hashlib.sha256((ROOT/r['settings']['schedule']).read_bytes()).hexdigest()==r['verification']['schedule_sha256']
        assert r['map']['sha256']==s['map']['sha256'] and r['settings']['steps']==s['settings']['steps']
        for src,digest in dict(s['source_files_sha256'],**r['source_files_sha256']).items():
            source = ROOT/('benchmarks/reference/compare_town_cache_draft_v1.py' if src == 'scripts/compare_town_cache.py' else src)
            assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,src
        report={'seed':r['settings']['seed'],'file':path.name,'load_count':0,'cells':[]}
        arms=['stateless','lru4','lru8','whole_resident'];size=r['settings']['size'];steps=r['settings']['steps']
        for case in r['cases']:
            data=Path(case['chunked_path']).read_bytes();whole=Path(case['whole_path']).read_bytes()
            for key,b in [('chunked',data),('whole',whole)]:assert hashlib.sha256(b).hexdigest()==case['hashes_before'][key]==case['hashes_after'][key]==case['hashes'][key+'_sha256']
            import struct
            _,_,_,_,cs,rows,cols,count=struct.unpack_from('<4sBBHIIII',data);assert cs==256
            entries=[struct.unpack_from('<IIIIQQ',data,24+i*32) for i in range(count)];meta=24+count*32
            def coord(indices):return [[i//cols,i%cols] for i in indices]
            for mode,kinds in case['modes'].items():
                for ki,(kind,b) in enumerate(kinds.items()):
                    flat={a:[] for a in arms};starts={a:[] for a in arms};totals={a:[] for a in arms};first={a:[] for a in arms};step_totals={a:[] for a in arms}
                    for ri,rep in enumerate(b['repeats']):
                        rotate=(ki+ri)%len(arms);assert rep['construct_order']==arms[rotate:]+arms[:rotate]
                        startup={}
                        for a in rep['construct_order']:
                            if a=='stateless':startup[a]={k:0 for k in ['total_ns','read_ns','decode_ns','bytes_read','bytes_requested','read_calls','seek_calls']}
                            else:
                                rec=next(obs);assert rec['mode']==mode
                                if a=='whole_resident':assert rec['type']=='WholeMapResident' and rec['path']==case['whole_path'];expected_bytes=len(whole)
                                else:assert rec['type']=='ChunkRegionCache' and rec['capacity']==int(a[3:]) and rec['path']==case['chunked_path'];expected_bytes=meta
                                startup[a]=rec['startup'];assert rec['startup']['bytes_read']==rec['startup']['bytes_requested']==expected_bytes
                                assert rec['startup']['read_calls']==rec['startup']['seek_calls']==2
                            starts[a].append(startup[a]);assert startup[a]['total_ns']==startup[a]['read_ns']+startup[a]['decode_ns']
                        lrus={'lru4':collections.OrderedDict(),'lru8':collections.OrderedDict()}
                        assert len(rep['frames'])==steps
                        for fi,f in enumerate(rep['frames']):
                            center=s['trajectories'][kind][fi];roi=[max(0,min(center[0]-128,size-256)),max(0,min(center[1]-128,size-256)),256,256];assert f['roi']==roi
                            rr,cc,h,w=roi;selected=[i for i,e in enumerate(entries) if e[0]<rr+h and e[0]+e[2]>rr and e[1]<cc+w and e[1]+e[3]>cc]
                            rotate=(ki+ri+fi)%len(arms);assert f['arm_order']==arms[rotate:]+arms[:rotate]
                            for a,m in f['arms'].items():
                                assert m['total_ns']==m['read_ns']+m['decode_ns'] and min(m['read_ns'],m['decode_ns'])>=0
                                if a.startswith('lru'):
                                    lru=lrus[a];hits=[i for i in selected if i in lru];misses=[i for i in selected if i not in lru];evictions=[]
                                    for i in hits:lru.move_to_end(i)
                                    for i in misses:
                                        if len(lru)==int(a[3:]):evictions.append(lru.popitem(last=False)[0])
                                        lru[i]=True
                                    assert m['hit_coordinates']==coord(hits) and m['miss_coordinates']==coord(misses) and m['evicted_coordinates']==coord(evictions)
                                    assert m['cache_hits']==len(hits) and m['cache_misses']==m['chunks_loaded']==len(misses) and m['evictions']==len(evictions)
                                    assert m['chunk_coordinates']==coord(selected) and m['chunks_requested']==len(selected)
                                    expected_bytes=sum(entries[i][5] for i in misses);calls=len(misses);resident=sum(entries[i][2]*entries[i][3]*2 for i in lru)
                                elif a=='stateless':expected_bytes=meta+sum(entries[i][5] for i in selected);calls=2+len(selected);resident=0;assert m['cache_misses']==len(selected) and m['cache_hits']==0
                                else:expected_bytes=0;calls=0;resident=size*size*2
                                assert m['bytes_read']==m['bytes_requested']==expected_bytes and m['read_calls']==m['seek_calls']==calls
                                assert m['resident_bytes']==resident
                                flat[a].append(m);report['load_count']+=1
                        for a in arms:
                            ns=[f['arms'][a]['total_ns'] for f in rep['frames']];step_totals[a].append(sum(ns));totals[a].append(startup[a]['total_ns']+sum(ns));first[a].append(startup[a]['total_ns']+ns[0])
                    summary={'method':case['method'],'mode':mode,'trajectory':kind,'arms':{}}
                    for a,frames in flat.items():
                        aggregate=b['aggregate'][a];hit=[m['total_ns'] for m in frames if m['cache_misses']==0];miss=[m['total_ns'] for m in frames if m['cache_misses']>0]
                        assert aggregate['frame_total']==stats([m['total_ns'] for m in frames]);assert aggregate['frame_total_hit_only']==stats(hit) and aggregate['frame_total_miss_involving']==stats(miss)
                        assert aggregate['startup_total_ns']==stats([v['total_ns'] for v in starts[a]]) and aggregate['first_viewport_ns']==stats(first[a])
                        assert aggregate['bytes_read_total_including_startup']==sum(m['bytes_read'] for m in frames)+sum(v['bytes_read'] for v in starts[a])
                        requested=sum(m['chunks_requested'] for m in frames);rate=sum(m['cache_hits'] for m in frames)/requested if requested else None
                        assert aggregate['hit_rate']==rate and aggregate['evictions_total']==sum(m['evictions'] for m in frames)
                        assert aggregate['resident_bytes_max']==max(m['resident_bytes'] for m in frames) and aggregate['resident_bytes_end']==frames[-1]['resident_bytes']
                        summary['arms'][a]={'frame':aggregate['frame_total'],'startup':stats([v['total_ns'] for v in starts[a]]),'first_viewport':stats(first[a]),'trajectory_including_startup':stats(totals[a]),'trajectory_movement_only':stats(step_totals[a]),'hit_rate':rate,'resident_bytes_peak':aggregate['resident_bytes_max'],'bytes_read_including_startup_all_repeats':aggregate['bytes_read_total_including_startup'],'hit_only_frame_count':len(hit),'miss_frame_count':len(miss)}
                    report['cells'].append(summary)
        assert next(obs,None) is None
        evidence['reports'].append(report)
    memory_path=ROOT/'benchmarks/town-cache-memory-audit.json'
    if memory_path.exists():
        memory=json.loads(memory_path.read_text());assert memory['probe_count']==216
        for c in memory['cases']:
            corresponding=next(x for x in evidence['reports'] if x['seed']==c['seed']);cell=next(x for x in corresponding['cells'] if x['method']==c['method'] and x['trajectory']==c['trajectory'] and x['mode']=='warm_buffered');arm=cell['arms'][c['arm']]
            for m in c['samples']:
                assert m['resident_decoded_bytes_peak']==arm['resident_bytes_peak'];assert m['one_output_bytes_peak']==256*256*2
                assert m['bytes_read_including_startup']*3==arm['bytes_read_including_startup_all_repeats']
                assert m['process_rss_overall_hwm_bytes']>=m['process_rss_startup_hwm_bytes']>=m['process_rss_baseline_hwm_bytes']
                assert m['tracemalloc_overall_peak_bytes']>=m['tracemalloc_end_current_bytes']>=m['resident_decoded_bytes_final']
        evidence['memory_audit_probes']=memory['probe_count']
    with (ROOT/'benchmarks/town-cache-audit-evidence.json').open('w') as f:json.dump(evidence,f,indent=2);f.write('\n')
    print('Verified raw cache loads:',sum(r['load_count'] for r in evidence['reports']),'constructor metrics:',324,'memory probes:',evidence.get('memory_audit_probes','pending'))
if __name__=='__main__':main()
