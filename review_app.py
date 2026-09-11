"""Local candidate-review UI. No extraction, model inference, or training endpoints.

Run with the project interpreter, or double-click Launch Candidate Review.cmd.
The HTTP server binds only to loopback. Plotly and candidate assets are served
locally. Human reviews, suggestion exposures, and cursor state are separate from
the immutable candidate batch and existing prediction/feature caches.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import object_candidates as existing
import review_batch

ROOT = Path(__file__).resolve().parent
APP_ID = 'ai4infra-candidate-review'
CROP_QUALITIES = ['usable single object', 'incomplete object', 'multiple objects', 'ground contamination', 'unclear']
DEFAULT_CLASSES = ['Vegetation', 'Utility pole', 'Streetlight', 'Traffic sign', 'Fence or railing',
                   'Building / wall', 'Vehicle', 'Bicycle', 'Debris']


def now(): return datetime.now(timezone.utc).isoformat()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic_json(path, value): existing._atomic_json(Path(path), value)


class ReviewStore:
    """Immediate atomic saves with revision history; optimistic revision checks."""
    def __init__(self, batch_path, data_dir):
        self.batch_path = Path(batch_path).resolve()
        self.batch = review_batch.load_batch(self.batch_path)
        self.candidates = self.batch['candidates']
        self.by_id = {c['id']:c for c in self.candidates}
        if not self.candidates: raise ValueError('The prepared candidate batch is empty')
        self.data_dir = Path(data_dir).resolve(); self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.session_path = self.data_dir/'session.json'
        self.session = read_json(self.session_path) if self.session_path.exists() else {}
        if self.session.get('current_id') not in self.by_id: self.session['current_id'] = self.candidates[0]['id']
        self.session.setdefault('reviewer_name','')
        self.classes_path = self.data_dir/'classes.json'
        self.classes = read_json(self.classes_path) if self.classes_path.exists() else DEFAULT_CLASSES.copy()
        if not self.classes_path.exists(): atomic_json(self.classes_path,self.classes)
        atomic_json(self.session_path,self.session)

    def candidate(self, identity):
        if identity not in self.by_id: raise ValueError('Candidate is not in this prepared batch')
        return self.by_id[identity]

    def document(self, identity):
        self.candidate(identity); path=self.data_dir/'reviews'/f'{identity}.json'
        return read_json(path) if path.exists() else {'candidate_id':identity,'history':[]}

    def latest(self, identity):
        history=self.document(identity)['history'];return history[-1] if history else None

    def exposures(self, identity):
        path=self.data_dir/'exposures'/f'{identity}.json'
        return read_json(path) if path.exists() else []

    def refs(self, candidate):
        predictions=candidate.get('predictions',{})
        refs={'prediction_path':candidate.get('prediction_path') or predictions.get('candidate'),
              'context_prediction_path':candidate.get('context_prediction_path') or predictions.get('context'),
              'feature_cache':candidate.get('feature_reference') or candidate.get('feature_cache'),
              'feature_cache_reference':candidate.get('feature_cache_reference')}
        refs['models']={}
        for role,key in [('candidate','prediction_path'),('context','context_prediction_path')]:
            if refs[key]:
                record=read_json(refs[key])
                refs['models'][role]={k:record.get(k) for k in ('checkpoint','resolved_revision','inference_key','prompts')}
        return refs

    def progress(self):
        return {'reviewed':sum(self.latest(c['id']) is not None for c in self.candidates),'total':len(self.candidates)}

    def public_candidate(self,c):
        reviewed=self.latest(c['id']) is not None
        display=c.get('original_parent_id') or c['id']
        if not c.get('is_existing_parent'):display+=f" / group {c['component_label']}"
        return {'id':c['id'],'display_id':display,
            'point_count':c['point_count'],'context_count':c['crop_point_count']-c['point_count'],
            'region_id':c.get('region_id',c.get('crop_id','cached region')),
            'shared_object_group':c['leakage_group_id'],
            'overlap_ids':c.get('possible_duplicate_ids',[]),'reviewed':reviewed,
            'boundary_warning':bool(c.get('near_crop_boundary'))}

    def state(self):
        with self.lock:
            return {'candidates':[self.public_candidate(c) for c in self.candidates],
                **self.session,'classes':self.classes,'crop_qualities':CROP_QUALITIES,
                'progress':self.progress(),'session_token':self.token}

    def position(self, identity, reviewer_name=''):
        with self.lock:
            self.candidate(identity)
            self.session.update(current_id=identity,reviewer_name=str(reviewer_name)[:160],updated_utc=now())
            atomic_json(self.session_path,self.session)
            return {'current_id':identity}

    def reveal(self, identity):
        with self.lock:
            candidate=self.candidate(identity);refs=self.refs(candidate)
            clip={}
            for role,key in [('candidate','prediction_path'),('context','context_prediction_path')]:
                path=refs.get(key)
                if path and Path(path).is_file():
                    record=read_json(path)
                    clip[role]={key:record.get(key) for key in ('ranking','top1_disagreement_fraction','checkpoint','resolved_revision')}
            geometry=None
            reference=refs.get('feature_cache') or refs.get('feature_cache_reference')
            if reference:
                if isinstance(reference,dict):
                    feature_path=reference.get('path') or reference.get('report_path')
                else: feature_path=reference
                if feature_path and Path(feature_path).is_file():
                    feature=read_json(feature_path)
                    if 'roles' in feature: geometry=feature['roles']
                    elif 'entries' in feature:
                        legacy=set(candidate.get('legacy_ids',[])+[candidate.get('original_parent_id')])
                        matched=next((e for e in feature['entries'] if e['id'] in legacy),None)
                        geometry=matched['roles'] if matched else None
            available=bool(clip or geometry)
            events=self.exposures(identity)
            events.append({'at_utc':now(),'reviewer_name':self.session.get('reviewer_name',''),
                           'suggestions_available':available,'references':refs})
            atomic_json(self.data_dir/'exposures'/f'{identity}.json',events)
            return {'clip':clip or None,'geometry':geometry,'available':available,
                'message':'Existing suggestions only; they are not verified labels.' if available else
                          'No saved model predictions or geometry features exist for this candidate. Nothing was computed.'}

    def evaluation_groups(self):
        """Conservative connected groups: automatic crop groups plus human join tags.

        An override only joins groups; it cannot split an overlapping crop group.
        Cross-scan grouping is prepared in batch provenance, never by point split.
        """
        parent={c['id']:c['id'] for c in self.candidates}
        def find(a):
            while parent[a]!=a: parent[a]=parent[parent[a]];a=parent[a]
            return a
        buckets={}
        for c in self.candidates:
            latest=self.latest(c['id']);tag=latest['human'].get('shared_object_group_override') if latest else None
            for key in [('automatic',c['leakage_group_id'])]+([('human',tag)] if tag else []):
                if key in buckets: parent[find(c['id'])]=find(buckets[key])
                else:buckets[key]=c['id']
        members={}
        for identity in parent:members.setdefault(find(identity),[]).append(identity)
        return {identity:'eval_'+min(group).removeprefix('candidate_') for group in members.values() for identity in group}

    def reference_rows(self):
        groups=self.evaluation_groups();rows=[]
        for c in self.candidates:
            record=self.latest(c['id']);human=record['human'] if record else {}
            rows.append({'candidate_id':c['id'],'original_parent_id':c.get('original_parent_id'),
                'region_id':c.get('region_id'),'evaluation_group':groups[c['id']],
                'automatic_group':c['leakage_group_id'],'possible_duplicate_ids':'|'.join(c.get('possible_duplicate_ids',[])),
                'point_count':c['point_count'],'reviewed':record is not None,
                'human_class':human.get('human_class'),'crop_quality':human.get('crop_quality'),
                'notes':human.get('notes',''),'reviewer_name':human.get('reviewer_name',''),
                'saved_utc':record.get('saved_utc') if record else None,'revision':record.get('revision',0) if record else 0,
                'suggestions_revealed':record.get('suggestions_revealed') if record else None,
                'eligible_for_later_classification':bool(record and human.get('human_class') and human.get('crop_quality')=='usable single object'),
                'membership_path':c['membership_path'],'crop_path':c['crop_path'],
                'prediction_reference':self.refs(c)['prediction_path'],
                'context_prediction_reference':self.refs(c)['context_prediction_path'],
                'feature_reference':json.dumps(self.refs(c).get('feature_cache'))})
        return rows

    def export(self):
        with self.lock:
            rows=self.reference_rows();buffer=io.StringIO(newline='')
            writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            csv_text=buffer.getvalue()
            folder=self.data_dir/'exports';folder.mkdir(exist_ok=True)
            temporary=folder/'reference_set.csv.tmp';temporary.write_text(csv_text,encoding='utf-8-sig',newline='')
            temporary.replace(folder/'reference_set.csv')
            manifest={'schema_version':1,'exported_utc':now(),'batch_path':str(self.batch_path),
                'data_dir':str(self.data_dir),'candidate_count':len(rows),'reviewed_count':sum(r['reviewed'] for r in rows),
                'evaluation_groups':len(set(r['evaluation_group'] for r in rows)),
                'label_scope':'candidate only; surrounding scene is not assigned its class',
                'split_policy':'Keep connected overlap groups, the same physical object, and corresponding overlapping left/right observations together. Human tags can join groups; they cannot split automatic groups. No train/test split or training was performed.',
                'limitations':'Small development reference set; preliminary experiments only. Prefer blind first-pass reviews for held-out evaluation; preserve suggestion-exposure flags.',
                'candidates':self.candidates,'rows':rows}
            atomic_json(folder/'manifest.json',manifest)
            return csv_text,manifest

    def save(self, payload):
        with self.lock:
            identity=payload['candidate_id'];candidate=self.candidate(identity)
            label=payload.get('human_class');quality=payload.get('crop_quality')
            if label is not None:
                label=str(label).strip()
                if label.lower() in ('','unknown','unknown/uncertain','unknown / uncertain'):label=None
                elif len(label)>160:raise ValueError('Class must be at most 160 characters')
            if quality not in CROP_QUALITIES:raise ValueError('Choose a crop quality')
            document=self.document(identity);revision=len(document['history'])
            if 'expected_revision' in payload and int(payload['expected_revision'])!=revision:
                raise ValueError('This review changed in another window. Reload before saving; its history is preserved.')
            human={'human_class':label,'crop_quality':quality,'notes':str(payload.get('notes',''))[:10000],
                'reviewer_name':str(payload.get('reviewer_name',''))[:160],
                'shared_object_group_override':str(payload.get('shared_object_group_override') or '').strip()[:160],
                'label_scope':'candidate_only'}
            events=self.exposures(identity)
            record={'schema_version':1,'candidate_id':identity,'revision':revision+1,'saved_utc':now(),
                'human':human,'suggestions_revealed':any(e['suggestions_available'] for e in events),
                'reveal_requested':bool(events),'reveal_event_count':len(events),
                'last_reveal_utc':events[-1]['at_utc'] if events else None,
                'exposure_note':'Conservative local-app exposure history; prior exposure outside this app is unknown.',
                'provenance':candidate,'model_and_feature_references':self.refs(candidate),
                'batch_path':str(self.batch_path)}
            document['history'].append(record)
            atomic_json(self.data_dir/'reviews'/f'{identity}.json',document)
            if label and label not in self.classes:
                self.classes.append(label);atomic_json(self.classes_path,self.classes)
            next_id=identity
            if payload.get('advance',True):
                ids=list(self.by_id);start=ids.index(identity)
                for step in range(1,len(ids)+1):
                    possible=ids[(start+step)%len(ids)]
                    if self.latest(possible) is None:next_id=possible;break
            self.position(next_id,human['reviewer_name'])
            self.export()
            return {'saved':{'candidate_id':identity,'revision':record['revision'],'at_utc':record['saved_utc']},
                    'current_id':next_id,'progress':self.progress(),'classes':self.classes}


@lru_cache(maxsize=3)
def figure_json(batch_path, identity):
    batch=review_batch.load_batch(batch_path)
    arrays=review_batch.load_candidate_arrays(batch,identity)
    points=np.asarray(arrays['xyz']);context=np.asarray(arrays['context_xyz']);origin=np.asarray(arrays['origin'])
    candidate_rows=np.asarray(arrays['raw_record_indices'])
    context_rows=arrays.get('context_raw_record_indices')
    keep=np.ones(len(context),dtype=bool) if context_rows is None else ~np.isin(context_rows,candidate_rows)
    def subset(n):return np.linspace(0,n-1,min(n,20000),dtype=int) if n else np.array([],dtype=int)
    fig=go.Figure()
    for xyz,rgb,name,size,opacity in [(points,arrays['rgb'],'Candidate — original RGB',3,1),
                                    (context[keep],np.asarray(arrays['context_rgb'])[keep],'Context — original RGB',2,.32)]:
        pick=subset(len(xyz));xyz=xyz[pick]-origin;colors=np.asarray(rgb)[pick]
        if colors.dtype==np.uint16:colors=colors/65535.
        elif colors.dtype==np.uint8:colors=colors/255.
        colors=np.rint(colors*255).astype(np.uint8)
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',name=name,
            marker={'size':size,'color':[f'rgb({r},{g},{b})' for r,g,b in colors],'opacity':opacity},
            hovertemplate='X %{x:.3f}<br>Y %{y:.3f}<br>Z %{z:.3f}<extra>'+name+'</extra>'))
    low,high=points.min(axis=0)-origin,points.max(axis=0)-origin
    corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
    lines=[]
    for i in range(8):
        for bit in (1,2,4):
            j=i^bit
            if i<j:lines.extend([corners[i].tolist(),corners[j].tolist(),[None]*3])
    fig.add_trace(go.Scatter3d(x=[p[0] for p in lines],y=[p[1] for p in lines],z=[p[2] for p in lines],
        mode='lines',name='Candidate bounds',line={'color':'#e44c86','width':4},hoverinfo='skip'))
    whole=context-origin;axis={}
    for dim,name in enumerate('xyz'):
        lo,hi=float(whole[:,dim].min()),float(whole[:,dim].max());padding=max((hi-lo)*.06,.05)
        axis[name+'axis']={'title':f'Local {name.upper()} · source units','range':[lo-padding,hi+padding]}
    fig.update_layout(height=560,margin={'l':0,'r':0,'t':10,'b':0},
        scene={'aspectmode':'data',**axis},legend={'orientation':'h'},uirevision=identity,
        paper_bgcolor='#ffffff')
    return json.loads(fig.to_json())


class Handler(BaseHTTPRequestHandler):
    server_version='LocalCandidateReview/1'
    def log_message(self,format,*args):pass
    @property
    def store(self):return self.server.store
    def origin_valid(self):
        host=self.headers.get('Host','').lower();allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if host not in allowed:return False
        origin=self.headers.get('Origin')
        return not origin or origin in {'http://'+h for h in allowed}
    def send(self,body,status=200,content_type='application/json',filename=None):
        if isinstance(body,(dict,list)):body=json.dumps(body,ensure_ascii=False).encode('utf-8')
        elif isinstance(body,str):body=body.encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if not self.origin_valid():return self.send({'error':'Local origin required'},403)
        parsed=urlparse(self.path);query=parse_qs(parsed.query)
        try:
            if parsed.path=='/health':return self.send({'app':APP_ID,'ready':True})
            if parsed.path=='/':return self.send((ROOT/'review_ui.html').read_bytes(),content_type='text/html; charset=utf-8')
            if parsed.path=='/plotly.js':return self.send(get_plotlyjs(),content_type='application/javascript; charset=utf-8')
            if parsed.path=='/api/state':return self.send(self.store.state())
            if parsed.path=='/api/candidate':
                identity=query['id'][0];c=self.store.candidate(identity);latest=self.store.latest(identity)
                images=[]
                for kind,key in [('candidate','static_views'),('context','context_views')]:
                    for index,view in enumerate(c.get(key,[])):
                        images.append({'url':f'/asset?id={identity}&kind={kind}&index={index}',
                            'label':('Candidate' if kind=='candidate' else 'Wider region')+' - '+view.get('name',f'View {index+1}'),'kind':kind})
                if c.get('locator_image'):
                    images.append({'url':f'/asset?id={identity}&kind=locator&index=0',
                                   'label':'Candidate location in context','kind':'locator'})
                warning='Candidate group; object completeness is unverified.'
                return self.send({**self.store.public_candidate(c),'plotly':figure_json(str(self.store.batch_path),identity),
                    'images':images,'review':latest['human'] if latest else None,'history_count':latest['revision'] if latest else 0,
                    'suggestions_revealed':latest['suggestions_revealed'] if latest else False,'warning':warning})
            if parsed.path=='/asset':
                c=self.store.candidate(query['id'][0]);kind=query['kind'][0]
                if kind=='locator':return self.send(Path(c['locator_image']).read_bytes(),content_type='image/png')
                if kind not in ('candidate','context'):raise ValueError('Invalid image kind')
                view=c['static_views' if kind=='candidate' else 'context_views'][int(query['index'][0])]
                return self.send(Path(view['path']).read_bytes(),content_type='image/png')
            if parsed.path in ('/api/export','/api/manifest'):
                csv_text,manifest=self.store.export()
                return self.send(csv_text,content_type='text/csv; charset=utf-8',filename='reference_set.csv') if parsed.path=='/api/export' else self.send(manifest,filename='manifest.json')
            return self.send({'error':'Not found'},404)
        except (ValueError,KeyError,IndexError,FileNotFoundError) as error:self.send({'error':str(error)},400)
    def do_POST(self):
        if not self.origin_valid() or not secrets.compare_digest(self.headers.get('X-Review-Token',''),self.store.token):
            return self.send({'error':'Local session token required'},403)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<=length<=100000:raise ValueError('Request too large')
            payload=json.loads(self.rfile.read(length) or b'{}')
            if self.path=='/api/position':return self.send(self.store.position(payload['candidate_id'],payload.get('reviewer_name','')))
            if self.path=='/api/reveal':return self.send(self.store.reveal(payload['candidate_id']))
            if self.path=='/api/review':return self.send(self.store.save(payload))
            if self.path=='/api/shutdown':
                self.send({'stopping':True});threading.Thread(target=self.server.shutdown,daemon=True).start();return
            return self.send({'error':'Not found'},404)
        except (ValueError,KeyError,TypeError) as error:
            self.send({'error':str(error)},409 if 'changed in another window' in str(error) else 400)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'outputs/objects/reference_set')
    parser.add_argument('--batch',type=Path)
    args=parser.parse_args()
    batch_path=args.batch or Path(read_json(ROOT/'outputs/objects/review_batches/active_batch.json')['batch_path'])
    store=ReviewStore(batch_path,args.data_dir)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.store=store
    store.export()
    print(f'Candidate review: http://127.0.0.1:{args.port} · {len(store.candidates)} candidates',flush=True)
    if not args.no_browser:threading.Timer(.3,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
