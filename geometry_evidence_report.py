"""Saved plots and offline HTML for descriptive geometry; reads cached evidence only."""
import base64
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
import numpy as np
import geometry_evidence as geo


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def fmt(value):return 'undefined' if value is None else f'{value:.3f}' if isinstance(value,float) else str(value)
def table(rows):
    if not rows:return ''
    names=list(rows[0]);return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in names)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(row[k]))+'</td>' for k in names)+'</tr>' for row in rows)+'</tbody></table></div>'


def summary_rows(report):
    rows=[]
    for entry in report['entries']:
        role=entry['roles']['candidate'];g=role['global'];p=role['local']['1.0']['proportions_of_valid_queries']
        rows.append({'candidate':entry['id'],'points':role['point_count'],
            'global L / P / S':' / '.join(fmt(g[k]) for k in ('linearity','planarity','scattering')),
            '|v1·Z|':g['axis_vertical_alignment'],
            'r=1 linear %':100*p['linear'],'r=1 planar %':100*p['planar'],
            'r=1 scattered %':100*p['scattered'],'r=1 vertical-linear %':100*p['vertical_linear']})
    return rows


def measured_description(entry):
    local=entry['roles']['candidate']['local'];a=local['0.3']['proportions_of_valid_queries'];b=local['1.0']['proportions_of_valid_queries']
    return (f"At radius 0.3, {a['linear']:.1%} of valid queries are locally linear and {a['planar']:.1%} locally planar. "
            f"At radius 1.0, {b['linear']:.1%} are linear, {b['planar']:.1%} planar and {b['scattered']:.1%} meet the scattering bin. "
            f"Locally linear regions with strong vertical alignment account for {b['vertical_linear']:.1%} at radius 1.0. "
            "These describe sampled point neighborhoods; semantic identity remains inconclusive.")


def make_figure(entry, run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    small=np.load(entry['roles']['candidate']['local']['0.3']['data_file'])
    large=np.load(entry['roles']['candidate']['local']['1.0']['data_file'])
    with np.load(entry['fingerprint']['crop_path'],allow_pickle=False) as data:
        with np.load(entry['fingerprint']['membership_path'],allow_pickle=False) as m:
            xyz=data['xyz'][m['selected_crop_indices']]
    origin=xyz.min(axis=0);local=xyz-origin
    fig,axes=plt.subplots(2,3,figsize=(13,9))
    axes[0,0].imshow(Image.open(entry['saved_rgb_view']));axes[0,0].axis('off')
    axes[0,0].set_title('Existing candidate RGB render\nunchanged original colors')
    background=local[np.linspace(0,len(local)-1,min(2000,len(local)),dtype=int)]
    def spatial(ax,data,feature,dims,title):
        points=data['query_xyz']-origin;values=data[feature];finite=np.isfinite(values)
        ax.scatter(background[:,dims[0]],background[:,dims[1]],s=1,c='#aaaaaa',alpha=.12)
        if (~finite).any():ax.scatter(points[~finite,dims[0]],points[~finite,dims[1]],marker='x',s=7,c='#777777')
        plotted=ax.scatter(points[finite,dims[0]],points[finite,dims[1]],c=values[finite],vmin=0,vmax=1,cmap='viridis',s=9,linewidths=0)
        ax.set(xlabel=f"Local {'XYZ'[dims[0]]} [source units]",ylabel=f"Local {'XYZ'[dims[1]]} [source units]",title=title,aspect='equal')
        ax.grid(alpha=.12);fig.colorbar(plotted,ax=ax,shrink=.7)
    spatial(axes[0,1],small,'linearity',(0,2),'Local linearity · radius 0.3 · XZ')
    spatial(axes[0,2],large,'linearity',(0,2),'Local linearity · radius 1.0 · XZ')
    spatial(axes[1,0],large,'axis_vertical_alignment',(0,2),'Principal-axis vertical alignment\n|v1·Z| · radius 1.0 · XZ')
    spatial(axes[1,1],large,'linearity',(1,2),'Local linearity · radius 1.0 · YZ')
    for data,radius in [(small,.3),(large,1.)]:
        vals=np.sort(data['linearity'][np.isfinite(data['linearity'])]);axes[1,2].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=f'radius {radius:g}')
    axes[1,2].set(xlim=(0,1),ylim=(0,1),xlabel='Local linearity',ylabel='Fraction of valid queries ≤ value',title='Linearity distributions (ECDF)')
    axes[1,2].legend();axes[1,2].grid(alpha=.15)
    fig.suptitle(f"{entry['id']} · {len(xyz):,} cached candidate points",fontsize=13)
    fig.text(.02,.012,'512 reproducible query points per cloud; neighbors searched in all candidate points. Gray crosses = undefined.\nColors on spatial plots encode measured geometry, not classes. Wider context is analyzed separately. All distances use unresolved source coordinate units.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.95));path=Path(run)/entry['id']/'geometry.png';fig.savefig(path,dpi=125);plt.close(fig)
    small.close();large.close();return path


def detail_tables(entry):
    output=''
    for name,role in entry['roles'].items():
        g=role['global'];output+=f'<h3>{name.title()} geometry — {role["point_count"]:,} points</h3>'
        output+=table([{'global L':g['linearity'],'global P':g['planarity'],'global S':g['scattering'],
            '|v1·Z|':g['axis_vertical_alignment'],'|normal·Z|':g['normal_vertical_alignment'],
            '1−|normal·Z|':g['surface_verticality'],'XYZ extents':' / '.join(fmt(v) for v in g['extents_xyz'])}])
        rows=[]
        for radius,s in role['local'].items():
            p=s['proportions_of_valid_queries']
            rows.append({'radius':radius,'valid / queries':f"{s['valid']} / {s['queries']}",
                'neighbors q10 / 50 / 90':' / '.join(fmt(v) for v in s['neighbor_count_quantiles']),
                'linear %':100*p['linear'],'planar %':100*p['planar'],
                'scattered %':100*p['scattered'],'vertical-linear %':100*p['vertical_linear'],
                'insufficient / zero variance':f"{s['insufficient_neighbors']} / {s['zero_variance']}"})
        output+=table(rows)
        output+='<details><summary>Descriptor distributions, defined-direction counts and full global eigenpairs</summary>'
        rows=[]
        for radius,s in role['local'].items():
            for feature in geo.NAMES:
                d=s[feature];rows.append({'radius':radius,'descriptor':feature,'defined':d['defined'],
                    'q10':d['q10_q50_q90'][0],'median':d['q10_q50_q90'][1],'q90':d['q10_q50_q90'][2]})
        output+=table(rows)+'<pre>'+html.escape(json.dumps({k:g[k] for k in ('covariance','eigenvalues','eigenvectors_columns')},indent=2))+'</pre></details>'
    for role in ('dense','context'):
        prediction=read(entry['predictions'][role]);title='Candidate-only CLIP' if role=='dense' else 'Wider-context CLIP (separate region)'
        output+=f'<h3>{title} — unchanged cached prediction</h3><p>View disagreement: {prediction["top1_disagreement_fraction"]:.0%}. Cosine similarity, not probability.</p>'
        output+=table(prediction['ranking'][:3])
        output+=table([{'view':v['view_label'],'leading description':v['top1']} for v in prediction['per_view']])
    return output


METHODS='''<h2>Method and interpretation</h2>
<p>Five existing dense candidates and their wider regions are analyzed separately. No raw LAS, extraction, regrouping, model download, inference, training or review writes are used. The user's visual pole assessment is development context, not independent test evidence. Only XYZ enters feature extraction; IDs name output files.</p>
<p>We center XYZ and use C = XᵀX/N. With descending λ1 ≥ λ2 ≥ λ3, linearity L=(λ1−λ2)/λ1, planarity P=(λ2−λ3)/λ1, and scattering S=λ3/λ1 (CloudCompare calls this sphericity). Ratios were checked against <a href="https://github.com/CloudCompare/CCCoreLib/blob/dc8c7d80f4ef8fd6a2c302c8d5f9c35099ba8432/src/Neighbourhood.cpp#L710">this pinned CloudCompare implementation</a>. The <a href="https://www.mdpi.com/1424-8220/23/3/1320">user's research reference</a> motivates multiple scales; its scalar normal notation is not used as a vector. We do not reproduce its improved Random Forest method.</p>
<p><b>Directions differ:</b> |v1·Z| measures principal-axis vertical alignment. |v3·Z| measures surface-normal vertical alignment: near 1 means a horizontal surface. CloudCompare's surface verticality is 1−|v3·Z|. Near-equal adjacent eigenvalues make the corresponding direction undefined (relative gap ≤10⁻⁶). Fewer than six neighbors or zero variance gives undefined features, not invented zeros. Population covariance normalization matches the implementation; eigenvector signs have no meaning.</p>
<p><b>Fixed scales:</b> radii 0.3 and 1.0 source coordinate units, roughly several to tens of prior measured dense-point spacings (~0.03–0.05). These test smaller surface patches and broader structure; they are not physically calibrated or tuned for a class. Up to 512 uniform point queries use RNG seed20260911, reused at both scales; neighbors are searched in the full corresponding candidate or context cloud, including the query itself.</p>
<p>Fixed descriptive bins: linear L≥0.6; planar P≥0.6; scattered S≥0.25; vertical-linear L≥0.6 and |v1·Z|≥0.9. These are provisional display summaries, identical across all inputs, not a semantic classifier. Bins can overlap. Fractions are among valid point queries and are density-weighted, not area, volume, object fractions or independent confidence estimates. Distributions and valid counts are provided alongside them.</p>
<p>No geometry/CLIP fusion, reranking, probabilities or class rules are added. Semantic consistency is <b>inconclusive</b>: geometry can reveal different structures behind similar CLIP descriptions but cannot establish their class from five unreviewed development examples. Boundaries, occlusion, RGB projection and attached parts remain confounders.</p>'''


def build_html(run):
    run=Path(run);report=read(run/'features_report.json');body='<h1>Geometry versus vision</h1><p>Cached data · descriptive experiment · human labeling remains paused</p>'
    body+='<h2>Measured contrasts</h2><p>The vertical candidate has global L=0.981 and |v1·Z|=0.998; global PCA already preserves its vertical structure. At radius1.0, 53.7% of valid local queries are vertical-linear, versus 8.4% at0.3 where many neighborhoods are planar surface patches. The planar candidate has 51.4% locally planar queries at1.0. The three development vegetation candidates have 33.6–47.1% scattered queries at1.0, versus 2.0–2.7% in the other two.</p><p>Candidate00019 has a high global vertical alignment (0.963) but only3.9% vertical-linear local neighborhoods. This directly shows why global orientation alone cannot identify a pole. Context changes these proportions substantially.</p><p><b>Outcome:</b> useful additional structural evidence, but no validated resolution of bicycle versus utility pole or other classes. CLIP rankings remain unchanged. <b>One next step:</b> collect independent human reviews of complete candidate boundaries and labels before testing whether these features can resolve class confusion.</p>'
    body+=table(summary_rows(report))+METHODS
    for entry in report['entries']:
        figure=make_figure(entry,run);entry['geometry_figure']=str(figure)
        body+=f'<section><h2>{html.escape(entry["id"])}</h2><p>{html.escape(measured_description(entry))}</p>'
        body+=f'<p>Crop-edge contact: {entry["crop_boundary"]}; original seed support: {entry["seed_support_fraction"]:.1%}. These are existing grouping limitations.</p>'
        body+='<img alt="RGB and local geometric evidence" src="data:image/png;base64,'+base64.b64encode(figure.read_bytes()).decode()+'">'
        if entry.get('interactive_html'):
            link=quote(os.path.relpath(entry['interactive_html'],run).replace('\\','/'))
            body+=f'<p><a href="{link}">Open existing local interactive view</a> (reused, unchanged).</p>'
        body+=detail_tables(entry)+'</section>'
    body+='<h2>Validation and files</h2><p>Separate synthetic tests passed for a line, horizontal/vertical planes, isotropic volume, translation, SVD eigenvalue agreement, degeneracies and the bounded full-neighbor query path. Synthetic points never enter real results.</p><p>Parameters: parameters.json. Full evidence: features_report.json; each candidate folder contains global covariance/eigenpairs and per-role/per-radius NPZ query membership, neighbor counts and descriptors. JSON null / NPZ NaN denotes undefined values. Previous CLIP JSON files and human reviews remain untouched.</p>'
    style='body{font:16px system-ui;margin:28px auto;max-width:1250px;padding:0 20px;color:#202630;line-height:1.5}h1,h2,h3{line-height:1.2}img{max-width:100%;height:auto}.table{overflow:auto;margin:15px 0}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px;border:1px solid #d5dae2;text-align:left}th{background:#edf2f8}section{border-top:3px solid #dbe4ed;margin-top:36px}details{background:#f6f8fa;padding:10px}pre{overflow:auto}'
    path=run/'report.html';path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Geometry versus vision</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
    geo.write_json(run/'features_report.json',report);return path


def show_summary(run):
    from IPython.display import HTML,display
    display(HTML(table(summary_rows(read(Path(run)/'features_report.json')))))


def show_candidate(run,index):
    from IPython.display import Image,HTML,display
    entry=read(Path(run)/'features_report.json')['entries'][index]
    print(entry['id'],measured_description(entry))
    display(Image(filename=entry['geometry_figure'],width=1000))
    display(HTML(detail_tables(entry)))


def show_candidate_compact(run,index):
    """Notebook version: spatial distributions plus unchanged top-three CLIP tables."""
    from IPython.display import Image,HTML,display
    entry=read(Path(run)/'features_report.json')['entries'][index]
    print(entry['id'],measured_description(entry))
    display(Image(filename=entry['geometry_figure'],width=1000))
    for role in ('dense','context'):
        prediction=read(entry['predictions'][role])
        label='Candidate only' if role=='dense' else 'Wider context — separate region'
        print(label, '| View disagreement:', f"{prediction['top1_disagreement_fraction']:.0%}", '| Cosine similarity, not probability')
        display(HTML(table(prediction['ranking'][:3])))
