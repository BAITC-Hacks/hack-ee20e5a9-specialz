#!/usr/bin/env python3
"""Read-only acceptance checks: python audit_project.py (after starter.py)."""
import json
import math
import random
import re
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    nodes = pd.read_csv(ROOT / 'out/nodes_roles.csv', dtype={'gid': str})
    clusters = pd.read_csv(ROOT / 'out/clusters.csv')
    top = pd.read_csv(ROOT / 'out/top_nodes.csv', dtype={'gid': str})
    raw = pd.read_parquet(ROOT / 'data/nodes.parquet')
    edges = pd.read_parquet(ROOT / 'data/edges.parquet')
    assert len(nodes) == nodes.gid.nunique() == 2248
    assert set(nodes.gid) == set(raw.gid.astype(str))
    for col in ['role', 'role_score', 'cluster_id', 'priority_score', 'evidence']:
        assert nodes[col].notna().all(), col
    assert nodes.evidence.str.len().between(1, 200).all()
    assert nodes.role_score.between(0, 1).all() and nodes.priority_score.between(0, 1).all()
    assert nodes.role.isin(['coordinator','consolidator','transit','distributor','terminal','peripheral']).all()
    assert clusters.cluster_id.is_unique
    assert set(clusters.cluster_id) == set(nodes.cluster_id)
    assert clusters.hypothesis.fillna('').str.strip().ne('').all()
    by_gid = nodes.set_index('gid')
    internal = {c: 0.0 for c in clusters.cluster_id}
    for e in edges.itertuples(index=False):
        a, b = by_gid.loc[str(e.src), 'cluster_id'], by_gid.loc[str(e.dst), 'cluster_id']
        if a == b:
            internal[a] += e.sum_kzt
    for c in clusters.itertuples(index=False):
        group = nodes[nodes.cluster_id == c.cluster_id]
        assert len(group) == c.n_nodes and group.is_seed.sum() == c.n_seed
        assert math.isclose(internal[c.cluster_id], c.sum_kzt_internal, abs_tol=.01)
        assert set(str(c.top_gids).split(';')) <= set(group.gid)
    assert len(top) >= 20 and top.gid.is_unique
    assert top['rank'].tolist() == list(range(1, len(top)+1))
    assert top.why.fillna('').str.strip().ne('').all()
    expected = nodes.sort_values(['priority_score','gid'], ascending=[False,True]).head(len(top))
    assert top.gid.tolist() == expected.gid.tolist()
    assert top.role.tolist() == expected.role.tolist()
    assert top.priority_score.tolist() == expected.priority_score.tolist()
    # Reconstruct every role using independent rule checks and exact centrality.
    graph = nx.DiGraph()
    graph.add_edges_from((str(e.src),str(e.dst)) for e in edges.itertuples(index=False))
    bc = nx.betweenness_centrality(graph, weight=None)
    threshold = nodes.gid.map(bc).fillna(0).quantile(.95)
    for n in nodes.itertuples(index=False):
        b = bc.get(n.gid, 0)
        if n.in_deg == n.out_deg == 0: role, score = 'peripheral', .9
        elif n.depth == 4 and n.out_deg == 0: role, score = 'terminal', .5
        elif n.out_deg == 0 and n.depth < 4: role, score = 'terminal', .9
        elif b > 0 and b >= threshold and n.in_deg >= 2 and n.out_deg >= 2: role, score = 'coordinator', .8
        elif not n.is_seed and n.in_deg >= 3 and n.pass_through < .3: role, score = 'consolidator', .8
        elif n.out_deg >= 10: role, score = 'distributor', .8
        elif not n.is_seed and .7 <= n.pass_through <= 1.3 and n.in_deg >= 1 and n.out_deg >= 1: role, score = 'transit', .8
        else: role, score = 'peripheral', .4
        assert (n.role, n.role_score) == (role, score), n.gid
    cut = nodes[nodes.truncated_by_depth]
    assert len(cut) == 444 and cut.evidence.str.contains('артефакт глубины',regex=False).all()
    assert not nodes[nodes.is_seed].role.isin(['consolidator','transit']).any()
    html = (ROOT / 'out/analyst.html').read_text()
    payload = json.loads(re.search(r'<script id="project-data" type="application/json">(.*?)</script>',html,re.S)[1])
    assert {n['gid'] for n in payload['nodes']} == set(nodes.gid)
    assert all(isinstance(e[k],str) for e in payload['edges'] for k in ['src','dst'])
    assert not re.search(r'<(?:script|link|iframe|img)[^>]+(?:src|href)=',html)
    print(f'PASS: {len(nodes)} unique nodes; evidence max={nodes.evidence.str.len().max()}; {len(clusters)} clusters; {len(top)} ranked nodes')
    print('PASS: all roles reconstructed; cluster sizes, seed counts and directed internal sums agree')
    print('PASS: HTML string gids; no external assets; 444 depth artifacts; seed protection')
    print('Three reproducible random examples (random seed 42):')
    for gid in random.Random(42).sample(nodes.gid.tolist(),3):
        print(gid, by_gid.loc[gid,'role'], by_gid.loc[gid,'evidence'])


if __name__ == '__main__':
    main()
