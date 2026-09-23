#!/usr/bin/env python3
"""Build a standalone, offline analyst page from existing CSV and edge data."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def explain_node(node):
    """Explain the saved role without recalculating or changing classification."""
    incoming, outgoing = int(node['in_deg']), int(node['out_deg'])
    text = (f'Разных наблюдаемых отправителей: {incoming}; получателей: {outgoing}. '
            'Это число участников, а не число переводов. ')
    role = node['role']
    if int(node['depth']) == 4 and outgoing == 0:
        text += ('На 4-м колене обхода наблюдение обрывается: исходящие связи за его '
                 'границей неизвестны. Метка terminal здесь — артефакт глубины, '
                 'а не установленный конечный получатель или подтверждённый «сток».')
    elif role == 'coordinator':
        match = re.search(r'BC=([\d.eE+\-]+) >= p95=([\d.eE+\-]+)', node['evidence'])
        text += 'Узел оказывается между другими узлами на наблюдаемых кратчайших маршрутах. '
        if match:
            bc, threshold = (value.replace('.', ',') for value in match.groups())
            text += (f'Показатель посредничества {bc} достигает или превышает порог '
                     f'{threshold} (95-й перцентиль); есть минимум два отправителя и два получателя. ')
        text += ('Поэтому алгоритм отметил признаки координирующего узла. '
                 'Это гипотеза о положении в графе, а не доказательство организации переводов.')
    elif role == 'consolidator':
        text += ('Узел не является исходным клиентом; наблюдаются минимум три отправителя, '
                 'а исходящая сумма составляет менее 30% входящей. '
                 'Алгоритм отметил признаки накопления в пределах выборки; фактический остаток неизвестен.')
    elif role == 'distributor':
        text += ('Переводы направлены как минимум 10 разным получателям. '
                 'По этому правилу алгоритм отметил признаки распределения средств.')
    elif role == 'transit':
        text += ('Узел не является исходным клиентом; есть входящие и исходящие связи, '
                 'а отношение исходящей суммы к входящей находится в диапазоне от 0,7 до 1,3 включительно. '
                 'Алгоритм отметил признаки транзита. Это не доказывает, что далее переводились те же деньги.')
    elif role == 'terminal':
        text += ('На наблюдаемой глубине менее 4 нет исходящих связей. '
                 'Поэтому алгоритм присвоил метку terminal. Это отсутствие выхода в выборке, '
                 'а не доказательство окончательного получения средств.')
    elif role == 'peripheral':
        text += ('В доступной выборке нет связей. Сработало правило изолированного узла. '
                 if incoming == 0 and outgoing == 0 else
                 'Специальные правила ролей не сработали; назначена остаточная роль peripheral. ')
        text += ('Метка peripheral не означает безопасность: приоритет проверки может быть высоким '
                 'из-за других показателей графа.')
    else:
        text += 'Для этой роли нет текстового пояснения; см. технические показатели.'
    if node['is_seed'] == 'True':
        text += (' Это исходный клиент: входящая сумма занижена по устройству выборки. '
                 'Правила накопления и транзита по отношению сумм для него не применяются.')
    return text


def build(data_dir, out_dir):
    nodes = pd.read_csv(out_dir / 'nodes_roles.csv', dtype=str, keep_default_na=False)
    top = pd.read_csv(out_dir / 'top_nodes.csv', dtype=str, keep_default_na=False).head(25)
    edges = pd.read_parquet(data_dir / 'edges.parquet')
    # Convert identifiers before serializing: JavaScript numbers cannot hold these IDs.
    for col in ('src', 'dst'):
        edges[col] = edges[col].astype(str)
    assert nodes.gid.is_unique, 'Duplicate gid in nodes_roles.csv'
    known = set(nodes.gid)
    assert set(edges.src) | set(edges.dst) <= known, 'Edge references unknown gid'
    assert set(top.gid) <= known, 'Top references unknown gid'
    nodes['explanation'] = [explain_node(node) for node in nodes.to_dict('records')]
    payload = {
        'nodes': nodes.to_dict('records'),
        'top': top.to_dict('records'),
        'edges': edges[['src', 'dst', 'sum_kzt', 'n_tx']].to_dict('records'),
    }
    # Escape HTML delimiters even inside JSON so evidence cannot close the script tag.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template = (ROOT / 'analyst_template.html').read_text(encoding='utf-8')
    destination = out_dir / 'analyst.html'
    destination.write_text(template.replace('__PROJECT_DATA__', encoded), encoding='utf-8')
    print(f'{destination.resolve()} — {len(nodes)} узлов, {len(edges)} направленных связей, топ-{len(top)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, default=ROOT / 'out')
    args = parser.parse_args()
    build(args.data, args.out)
