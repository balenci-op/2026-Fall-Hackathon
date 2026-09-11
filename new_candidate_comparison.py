"""Small additions for the bounded new-region comparison; existing algorithms reused."""
from pathlib import Path
import hashlib
import json
import numpy as np
from IPython.display import FileLink, Image, Markdown, display
import object_quality as quality
from ground_experiment_views import show_table


def combine_extraction_sources(sources, candidates):
    """Unite cached sample records so the existing extractor can read one raw file.

    Each candidate retains its original ID, XYZ/RGB, source identity and sample
    row membership. Only its indices into the combined record array are rebased.
    This is a union of already selected crops, never a new large scouting region.
    """
    if len({str(s['catalog']['input_manifest']['sources'][s['region']['side']]['source_path']) for s in sources}) != 1:
        raise ValueError('One matching raw source is required for this bounded pass')
    combined = dict(sources[0])
    all_rows = np.concatenate([s['sample_record_indices'] for s in sources])
    rows, first = np.unique(all_rows, return_index=True)
    records = np.concatenate([s['region']['records'] for s in sources])[first]
    identities = [s['identity'] for s in sources]
    combined['identity'] = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
    combined['region'] = {**combined['region'], 'records': records}
    combined['sample_record_indices'] = rows
    rebased = []
    for candidate in candidates:
        rebased.append({**candidate, 'original_source_identity': candidate['source_identity'],
                       'source_identity': combined['identity'],
                       'indices': np.searchsorted(rows, candidate['sample_record_indices'])})
    return combined, rebased


def export_interactive(figure, path):
    """Always create a standalone browser view; no notebook renderer or CDN needed."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False,
                      config={'responsive': True, 'displaylogo': False})
    print('Interactive HTML:', path)
    print('Open this file in your browser (Explorer: double-click the HTML file).')
    try:
        display(FileLink(str(path.relative_to(Path.cwd()))))
    except ValueError:
        pass
    return path


def dense_interactive(sparse, grouping, path):
    """Use the existing Plotly figure builder on dense membership + full crop context."""
    import object_views
    data = quality.load_crop({'path': grouping['crop_path']})
    with np.load(grouping['membership_path'], allow_pickle=False) as membership:
        indices = membership['selected_crop_indices']
    lo, hi = data['xyz'].min(axis=0), data['xyz'].max(axis=0)
    source = {'xyz': data['xyz'], 'rgb': data['rgb'], 'identity': grouping['candidate_id'],
              'region': {'origin': lo, 'context_roi_xy': [lo[0], hi[0], lo[1], hi[1]]}}
    candidate = {'id': grouping['candidate_id'], 'indices': indices,
                 'touches_region_boundary': grouping['touches_crop_boundary']}
    figure = object_views.candidate_3d(source, candidate, context_padding=1e6, max_display_points=20000)
    figure.data[1].visible = True  # Make the full-region fallback immediately inspectable.
    for trace, name in zip(figure.data, ('Tentative dense candidate', 'Wider dense crop context')):
        trace.hovertemplate = trace.hovertemplate.replace(trace.name, name)
        trace.name = name
    # Keep the entire cached crop; no processing sample is changed by the display cap.
    figure.update_layout(title_text=f"{sparse['id']} | region containing potentially multiple objects | source coordinate units",
        annotations=[dict(x=0,y=-.12,xref='paper',yref='paper',showarrow=False,xanchor='left',
            text='Full cached context available in legend; display capped at 20,000 points per trace.<br>'
                 'Region may contain multiple objects. Distances are unresolved source coordinate units.')])
    return export_interactive(figure, path)


def show_saved(report, index):
    """Read static comparison, top-three scores and HTML link; never infer or regroup."""
    item = report['items'][index]
    display(Markdown(f"**{item['candidate_id']} — {item['geometry']}**\n\n{item['warning']}"))
    display(Image(filename=item['figure'], width=950))
    show_table([{'sparse_points': item['sparse_points'], 'dense_points': item['dense_points'],
                 'context_points': item['context_points'], 'original_seed_retention': item['seed_retention'],
                 'crop_boundary': item['crop_boundary']}])
    for role in ('dense', 'context'):
        result = quality.read_json(item['predictions'][role])
        name = 'Candidate only (unverified grouping)' if role == 'dense' else 'Wider region containing potentially multiple objects'
        display(Markdown(f"**{name}** — viewpoint disagreement {result['top1_disagreement_fraction']:.0%}. Cosine similarity, not probability."))
        show_table(result['ranking'][:3])
        show_table([{'view': v['view_label'], 'top_description': v['top1']} for v in result['per_view']])
    html = Path(item['interactive_html'])
    print('Browser view:', html)
    display(FileLink(str(html.relative_to(Path.cwd()))))
