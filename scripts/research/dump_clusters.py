import json
M = json.load(open('/opt/project/Model/artifacts/metrics.json'))
c = M['head_c_clustering']
print('best_k =', c['best_k'])
print('| # | Ukuran | Istilah dominan |')
print('|---|---|---|')
for p in c['profiles']:
    print('| %d | %d | %s |' % (p['cluster_id'], p['size'], ', '.join(p['top_terms'][:4])))
print()
print('silhouette per k:', c.get('silhouette_by_k', c.get('scores', 'n/a')))
