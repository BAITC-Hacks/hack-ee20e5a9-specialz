#!/usr/bin/env python3
"""
Стартовый код кейса «Граф денег» — HackAlem AI.

Что он делает:
  1. грузит три parquet-файла и проверяет их консистентность;
  2. собирает направленный взвешенный граф;
  3. считает метрики, назначает роли, Louvain-кластеры и приоритеты;
  4. пишет три выгрузки в требуемой ТЗ схеме.

Чего он НЕ делает — это ваша работа:
  * не рисует граф.

Запуск:
    python starter.py --data ./data --out ./out
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROLES = ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"]


# ---------------------------------------------------------------- загрузка

def load(data_dir: Path):
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    return edges, nodes, tx


def sanity_check(edges, nodes, tx):
    """Проверки, которые стоит пройти до того, как строить модель."""
    print("=" * 64)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 64)
    print(f"  узлов в nodes.parquet : {len(nodes):>6}")
    print(f"  рёбер                 : {len(edges):>6}")
    print(f"  транзакций            : {len(tx):>6}")
    print(f"  seed-клиентов         : {int(nodes.is_seed.sum()):>6}")
    print(f"  оборот, KZT           : {edges.sum_kzt.sum():>14,.0f}")
    print(f"  период                : {tx.date.min().date()} — {tx.date.max().date()}")

    # транзакции должны складываться в рёбра
    agg = tx.groupby(["src", "dst"]).agg(s=("sum_kzt", "sum"), c=("sum_kzt", "size")).reset_index()
    m = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    assert (m._merge == "both").all(), "edges и transactions не сходятся по парам"
    print("  edges == transactions : OK")

    # узлы без единого ребра
    in_edges = set(edges.src) | set(edges.dst)
    orphans = set(nodes.gid) - in_edges
    print(f"\n  ВНИМАНИЕ: {len(orphans)} узлов нет ни в одном ребре "
          f"(из них seed: {len(orphans & set(nodes[nodes.is_seed].gid))})")
    print("  → они всё равно должны попасть в nodes_roles.csv")
    print("=" * 64, "\n")
    return orphans


# ---------------------------------------------------------------- граф

def build_graph(edges) -> nx.DiGraph:
    """Направленный граф. sum_kzt — вес ребра, n_tx — количество переводов."""
    G = nx.DiGraph()
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G


def basic_features(G: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    """Базовые метрики. Это старт, а не финиш — добавляйте свои."""
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_kzt = dict(G.in_degree(weight="sum_kzt"))
    out_kzt = dict(G.out_degree(weight="sum_kzt"))
    in_tx = dict(G.in_degree(weight="n_tx"))
    out_tx = dict(G.out_degree(weight="n_tx"))
    pr = nx.pagerank(G, weight="sum_kzt")

    df = nodes[["gid", "depth", "is_seed"]].copy()
    df["in_deg"] = df.gid.map(in_deg).fillna(0).astype(int)
    df["out_deg"] = df.gid.map(out_deg).fillna(0).astype(int)
    df["in_kzt"] = df.gid.map(in_kzt).fillna(0.0)
    df["out_kzt"] = df.gid.map(out_kzt).fillna(0.0)
    df["in_tx"] = df.gid.map(in_tx).fillna(0).astype(int)
    df["out_tx"] = df.gid.map(out_tx).fillna(0).astype(int)
    df["pagerank"] = df.gid.map(pr).fillna(0.0)

    # доля полученного, которая ушла дальше. Около 1.0 — деньги не задерживаются.
    df["pass_through"] = np.where(df.in_kzt > 0, df.out_kzt / df.in_kzt.replace(0, np.nan), np.nan)

    # ЛОВУШКА КЕЙСА: узел на 4-м колене без исходящих может быть не «стоком»,
    # а просто местом, где закончился обход. Разберитесь с этим.
    df["truncated_by_depth"] = (df.depth == 4) & (df.out_deg == 0)
    return df


def assign_roles(df, G, is_seed_col='is_seed'):
    """Первое совпадение побеждает; возвращает копию df с ролями.

    Для правил 4–7 score = 0.8. Топ-5%: положительная betweenness
    не ниже 95-го перцентиля по узлам df (включая совпадения на границе).
    Считаем пути без весов: сумма перевода не является расстоянием.
    Для seed после структурных правил 1–3 доступны только правила 4 и 6.
    """
    df = df.copy()
    centrality = nx.betweenness_centrality(G, weight=None)
    betweenness = df['gid'].map(centrality).fillna(0.0)
    df['betweenness'] = betweenness
    threshold = betweenness.quantile(0.95)
    roles, scores, evidence = [], [], []

    for row, bc in zip(df.to_dict('records'), betweenness):
        in_deg, out_deg = row['in_deg'], row['out_deg']
        in_kzt, pass_through = row['in_kzt'], row['pass_through']
        depth, is_seed = row['depth'], bool(row[is_seed_col])

        if in_deg == 0 and out_deg == 0:
            role, score, reason = 'peripheral', 0.9, 'Нет связей'
        elif depth == 4 and out_deg == 0:
            role, score, reason = 'terminal', 0.5, 'depth=4: артефакт глубины'
        elif out_deg == 0 and depth < 4:
            role, score, reason = 'terminal', 0.9, f'depth={depth}: нет выхода'
        elif bc > 0 and bc >= threshold and in_deg >= 2 and out_deg >= 2:
            role, score = 'coordinator', 0.8
            reason = f'BC={bc:.3g} >= p95={threshold:.3g}'
        elif in_deg >= 3 and pass_through < 0.3 and not is_seed:
            role, score, reason = 'consolidator', 0.8, 'Средства накапливаются'
        elif out_deg >= 10:
            role, score, reason = 'distributor', 0.8, 'Рассылка на 10+ узлов'
        elif not is_seed and 0.7 <= pass_through <= 1.3 and in_deg >= 1 and out_deg >= 1:
            role, score, reason = 'transit', 0.8, 'Вход близок к выходу'
        else:
            role, score, reason = 'peripheral', 0.4, 'Нет совпадений'

        details = (f'in_deg={in_deg}, out_deg={out_deg}, '
                   f'in_kzt={in_kzt:.6g}, pass_through={pass_through:.4g}')
        seed_note = '; seed: in_kzt занижен по устройству выборки' if is_seed else ''
        # Сохраняем метрики и оговорку seed даже при длинном пояснении.
        prefix = f'{details}{seed_note}; '
        roles.append(role)
        scores.append(score)
        evidence.append(prefix + reason[:max(0, 200 - len(prefix))])

    df['role'] = pd.Series(roles, index=df.index, dtype=object)
    df['role_score'] = pd.Series(scores, index=df.index, dtype=float)
    df['evidence'] = pd.Series(evidence, index=df.index, dtype=object)
    return df


def assign_clusters_and_priority(df, G):
    """Louvain, min–max приоритет и сводка кластеров.

    Доля seed — среди уникальных соседей обоих направлений, без самого
    узла; для изолятов 0. Постоянные метрики нормализуются в 0.
    Внутренний оборот считаем по исходному направленному графу,
    сохраняя оба перевода для взаимных рёбер.
    """
    df = df.copy()
    UG = G.to_undirected()
    UG.add_nodes_from(df['gid'])
    if UG.size(weight='sum_kzt') > 0:
        communities = nx.community.louvain_communities(UG, weight='sum_kzt', seed=42)
    else:
        communities = [{gid} for gid in UG]
    communities = sorted(communities, key=lambda members: min(members))
    membership = {gid: cid for cid, members in enumerate(communities) for gid in members}
    df['cluster_id'] = df['gid'].map(membership).astype(int)

    def normalized(values):
        span = values.max() - values.min()
        return (values - values.min()) / span if span > 0 else values * 0.0

    seeds = set(df.loc[df['is_seed'], 'gid'])
    seed_fractions = {}
    for gid in df['gid']:
        neighbors = set(UG.neighbors(gid)) - {gid}
        seed_fractions[gid] = len(neighbors & seeds) / len(neighbors) if neighbors else 0.0
    df['seed_neighbor_fraction'] = df['gid'].map(seed_fractions).astype(float)
    df['pagerank_normalized'] = normalized(df['pagerank'])
    df['betweenness_normalized'] = normalized(df['betweenness'])
    df['role_priority'] = df['role'].map({
        'consolidator': 1.0, 'coordinator': 1.0, 'distributor': 0.5,
    }).fillna(0.0)
    df['priority_score'] = (
        0.4 * df['pagerank_normalized'] + 0.3 * df['betweenness_normalized']
        + 0.2 * df['role_priority'] + 0.1 * df['seed_neighbor_fraction']
    )
    df['why'] = pd.Series([
        f'PR_norm={r.pagerank_normalized:.4f} × 0.4; '
        f'BC_norm={r.betweenness_normalized:.4f} × 0.3; '
        f'роль={r.role} ({r.role_priority:g} × 0.2); '
        f'доля seed-соседей={r.seed_neighbor_fraction:.4f} × 0.1'
        for r in df.itertuples(index=False)
    ], index=df.index, dtype=object)

    internal = {cid: 0.0 for cid in range(len(communities))}
    for src, dst, data in G.edges(data=True):
        if membership[src] == membership[dst]:
            internal[membership[src]] += float(data['sum_kzt'])
    patterns = {
        'consolidator': 'Возможно накопление средств',
        'coordinator': 'Возможно координирование потоков',
        'distributor': 'Возможно распределение средств',
        'transit': 'Возможен транзит средств',
        'terminal': 'Возможно завершение потоков или обрезка по глубине',
        'peripheral': 'Преобладают периферийные узлы',
    }
    records = []
    for cid, group in df.groupby('cluster_id', sort=True):
        counts = group['role'].value_counts()
        dominant = sorted(counts[counts == counts.max()].index)[0]
        n_seed = int(group['is_seed'].sum())
        top = group.sort_values(['pagerank', 'gid'], ascending=[False, True]).head(5)
        records.append({
            'cluster_id': cid, 'n_nodes': len(group), 'n_seed': n_seed,
            'sum_kzt_internal': internal[cid],
            'top_gids': ';'.join(str(gid) for gid in top['gid']),
            'hypothesis': f'{patterns[dominant]}; доминирует {dominant} '
                          f'({counts[dominant]}/{len(group)}); seed={n_seed}.',
        })
    clusters = pd.DataFrame(records, columns=[
        'cluster_id', 'n_nodes', 'n_seed', 'sum_kzt_internal', 'top_gids', 'hypothesis',
    ])
    return df, clusters


# ---------------------------------------------------------------- выгрузки

def write_outputs(df: pd.DataFrame, out_dir: Path, clusters: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. nodes_roles.csv — схема из ТЗ, рассчитанные значения из df
    roles = df[["gid"]].copy()
    roles["role"] = df["role"]
    roles["role_score"] = df["role_score"]
    roles["cluster_id"] = df["cluster_id"]
    roles["priority_score"] = df["priority_score"]
    roles["evidence"] = df["evidence"]
    roles = roles.merge(
        df[["gid", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank",
            "pass_through", "depth", "is_seed", "truncated_by_depth"]],
        on="gid", how="left")
    roles.to_csv(out_dir / "nodes_roles.csv", index=False)

    # 2. clusters.csv — сводка Louvain-кластеров
    clusters.to_csv(out_dir / "clusters.csv", index=False)

    # 3. top_nodes.csv — до 25 узлов с максимальным приоритетом
    top = df.sort_values(['priority_score', 'gid'], ascending=[False, True]).head(25)
    top = top[['gid', 'role', 'priority_score', 'why']].copy()
    top.insert(0, 'rank', range(1, len(top) + 1))
    top.to_csv(out_dir / "top_nodes.csv", index=False)

    print(f"Выгрузки записаны в {out_dir}/")


# ---------------------------------------------------------------- подсказки

def hints(G: nx.DiGraph, df: pd.DataFrame):
    """Куда смотреть дальше. Ответов здесь нет — только направления."""
    print("\nС ЧЕГО НАЧАТЬ")
    print("-" * 64)
    print(f"  узлов, получающих от 3+ разных плательщиков : {(df.in_deg >= 3).sum()}")
    print(f"  узлов, рассылающих на 10+ получателей       : {(df.out_deg >= 10).sum()}")
    print(f"  узлов и с входом, и с выходом               : {((df.in_deg > 0) & (df.out_deg > 0)).sum()}")
    print(f"  узлов, обрезанных 4-м коленом               : {df.truncated_by_depth.sum()}  <- разберитесь")
    print(f"  слабосвязных компонент                      : {nx.number_weakly_connected_components(G)}")
    print("""
  Вопросы, на которые стоит ответить метриками:
    * чем «деньги пришли и остались» отличается от «пришли и ушли дальше»?
    * что важнее для роли — количество плательщиков или сумма?
    * узел собирает средства от нескольких SEED — это случайность или структура?
    * если убрать узел, сеть распадётся или переживёт?

  Полезное в networkx: pagerank, hits, betweenness_centrality,
  community.louvain_communities, simple_cycles, all_simple_paths.
  Не забудьте: граф НАПРАВЛЕННЫЙ и ВЗВЕШЕННЫЙ.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path(__file__).resolve().parent / "data",
                    help="папка с parquet-файлами (по умолчанию data рядом со скриптом)")
    ap.add_argument("--out", default="./out", help="куда писать выгрузки")
    a = ap.parse_args()

    edges, nodes, tx = load(Path(a.data))
    sanity_check(edges, nodes, tx)
    G = build_graph(edges)
    df = basic_features(G, nodes)
    df = assign_roles(df, G)
    df, clusters = assign_clusters_and_priority(df, G)
    write_outputs(df, Path(a.out), clusters)
    hints(G, df)


if __name__ == "__main__":
    main()
