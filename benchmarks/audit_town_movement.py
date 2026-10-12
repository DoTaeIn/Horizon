import collections, hashlib, json, statistics
from pathlib import Path
import numpy as np
from horizon.town_map import make_town_map
from horizon.chunk_benchmark import _INDEX_HEADER,_INDEX_ENTRY

root=Path.cwd()
reports=[]
for name in ('town-movement-1024-v1.json','town-movement-1024-seed2-v1.json'):
 path=root/'benchmarks'/name
 report=json.loads(path.read_text())
 settings=report['settings']; size=settings['size'];steps=settings['steps'];repeats=settings['repeats']
 arr=make_town_map(size=size,seed=settings['seed'])
 assert hashlib.sha256(arr.tobytes()).hexdigest()==report['map']['sha256']
 for source,digest in report['source_files_sha256'].items():
  assert hashlib.sha256((root/source).read_bytes()).hexdigest()==digest,source
 result={'report':name,'seed':settings['seed'],'input_sha256':report['map']['sha256'],'load_count':0,'paths':{},'cells':[]}
 for kind,positions in report['trajectories'].items():
  coords=[]
  for r,c in positions:
   top=max(0,min(r-128,size-256));left=max(0,min(c-128,size-256))
   coords.append(tuple((y,x) for y in range(top//256,(top+255)//256+1) for x in range(left//256,(left+255)//256+1)))
  changes=sum(a!=b for a,b in zip(coords,coords[1:]))
  assert changes>0
  if kind=='boundary_backtrack':
   assert changes==steps-1
   assert max(max(abs(a[i]-b[i]) for i in (0,1)) for a,b in zip(positions,positions[1:]))<=28
  result['paths'][kind]={'distinct_chunk_sets':len(set(coords)),'chunk_set_changes':changes}
 def check_stats(actual,ns):
  ms=np.asarray(ns,dtype=np.float64)/1e6
  for key,value in [('median_ms',statistics.median(ms)),('p95_ms',np.percentile(ms,95,method='linear')),('max_ms',max(ms))]:
   assert abs(actual[key]-value)<=0.00000051,(key,actual[key],value)
 for case in report['cases']:
  data=Path(case['chunked_path']).read_bytes();wholedata=Path(case['whole_path']).read_bytes()
  assert hashlib.sha256(data).hexdigest()==case['hashes_before']['chunked']==case['hashes_after']['chunked']==case['hashes']['chunked_sha256']
  assert hashlib.sha256(wholedata).hexdigest()==case['hashes_before']['whole']==case['hashes_after']['whole']==case['hashes']['whole_sha256']
  _,_,_,_,cs,rows,cols,count=_INDEX_HEADER.unpack_from(data)
  assert cs==256
  entries=[_INDEX_ENTRY.unpack_from(data,_INDEX_HEADER.size+i*_INDEX_ENTRY.size) for i in range(count)]
  meta=_INDEX_HEADER.size+count*_INDEX_ENTRY.size
  for mode,kinds in case['modes'].items():
   for ki,(kind,block) in enumerate(kinds.items()):
    arms={'whole':[],'region':[]};sums={'whole':[],'region':[]};byte_sums={'whole':[],'region':[]}
    for ri,rep in enumerate(block['repeats']):
     assert rep['repeat']==ri and len(rep['frames'])==steps
     for fi,frame in enumerate(rep['frames']):
      center=report['trajectories'][kind][fi]
      assert [frame['center_row'],frame['center_col']]==center
      r,c,h,w=frame['roi'];assert frame['roi']==[max(0,min(center[0]-128,size-256)),max(0,min(center[1]-128,size-256)),256,256]
      assert frame['swapped']==bool((ki*repeats+ri+fi)%2)
      selected=[(i,e) for i,e in enumerate(entries) if e[0]<r+h and e[0]+e[2]>r and e[1]<c+w and e[1]+e[3]>c]
      expected_coords=[[i//cols,i%cols] for i,e in selected]
      for arm,m in [('whole',frame['whole']),('region',frame['region'])]:
       assert m['total_ns']==m['read_ns']+m['decode_ns'] and min(m['read_ns'],m['decode_ns'])>=0
       expected_bytes=len(wholedata) if arm=='whole' else meta+sum(e[5] for i,e in selected)
       expected_calls=2 if arm=='whole' else 2+len(selected)
       assert m['bytes_read']==m['bytes_requested']==expected_bytes
       assert m['read_calls']==m['seek_calls']==expected_calls
       if arm=='region':assert m['chunk_coordinates']==expected_coords and m['chunks_loaded']==len(selected)
       arms[arm].append(m);result['load_count']+=1
     for arm in arms:
      ms=[f[arm] for f in rep['frames']]
      sums[arm].append(sum(m['total_ns'] for m in ms));byte_sums[arm].append(sum(m['bytes_read'] for m in ms))
    for arm,metrics in arms.items():
     aggregate=block['aggregate'][arm]
     for stage in ('total','read','decode'):check_stats(aggregate[stage],[m[stage+'_ns'] for m in metrics])
     for key in ('bytes_read','bytes_requested','read_calls','seek_calls'):assert aggregate[key]==[m[key] for m in metrics]
     check_stats(block['aggregate']['trajectory_totals'][arm]['time_ms'],sums[arm])
     bs=block['aggregate']['trajectory_totals'][arm]['bytes']; values=byte_sums[arm]
     assert bs=={'median':round(statistics.median(values),3),'p95':round(float(np.percentile(values,95)),3),'max':round(max(values),3)}
    aggregate=block['aggregate']['region'];regionmetrics=arms['region']
    assert aggregate['chunks_loaded']==[m['chunks_loaded'] for m in regionmetrics]
    assert aggregate['chunk_coordinates']==[m['chunk_coordinates'] for m in regionmetrics]
    assert aggregate['chunk_count_distribution']==dict(sorted((str(k),v) for k,v in collections.Counter(m['chunks_loaded'] for m in regionmetrics).items()))
    result['cells'].append({'method':case['method'],'mode':mode,'trajectory':kind,'whole':block['aggregate']['whole']['total'],'region':aggregate['total'],'speedup':round(block['aggregate']['whole']['total']['median_ms']/aggregate['total']['median_ms'],3),'bytes_reduction_pct':round(100*(1-sum(m['bytes_read'] for m in arms['region'])/sum(m['bytes_read'] for m in arms['whole'])),3),'chunk_distribution':aggregate['chunk_count_distribution'],'whole_trajectory_ms':block['aggregate']['trajectory_totals']['whole']['time_ms'],'region_trajectory_ms':block['aggregate']['trajectory_totals']['region']['time_ms']})
 reports.append(result)
evidence={'audit':'Independent recomputation of schedules, chunk intersections, actual requested bytes, file/source/input hashes, all raw timing identities and aggregates. Random ROI correctness separately audited 450 requests across all3codecs on513x769.','tests':{'total':300,'passed':295,'skipped':5},'claude_sessions':{'roi_opus':'9325cb77-f939-4852-aece-1ef9efa48922','roi_tests_sonnet':'a26bea1d-f98e-479a-856d-9f3766d94196','movement_cli_sonnet':'cce0731e-6654-4117-80c5-043f587e6a03','movement_tests_opus':'9325cb77-f939-4852-aece-1ef9efa48922'},'reports':reports}
(root/'benchmarks'/'town-movement-audit-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence,indent=2))
