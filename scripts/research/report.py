import json, sys, os
for f in sys.argv[1:]:
    p = '/opt/project/Model/artifacts/_%s.json' % f
    if not os.path.exists(p):
        print('%-14s (belum ada)' % f); continue
    d = json.load(open(p))
    print('=== %s ===' % f)
    rows = d if isinstance(d, list) else [d]
    for r in sorted(rows, key=lambda r: -r.get('oof_f1_macro', 0)):
        print('  %-26s OOF %.4f | UJI makro %.4f mikro %.4f P@3 %.4f Ham %.4f'
              % (r.get('name', r.get('model', '?'))[:26], r.get('oof_f1_macro', 0),
                 r['f1_macro'], r['f1_micro'], r['precision_at_3'], r['hamming_loss']))
