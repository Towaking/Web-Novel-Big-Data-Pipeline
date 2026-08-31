import importlib, sys
print('python', sys.version.split()[0])
mods = ['sklearn','torch','numpy','scipy','pandas','joblib','transformers',
        'sentence_transformers','gensim','nltk','lightgbm','xgboost',
        'datasets','huggingface_hub','iterstrat','skmultilearn','optuna']
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, '__version__', '?')
        print('  OK   %-22s %s' % (m, v))
    except Exception:
        print('  MISS %-22s' % m)
try:
    import torch
    print('torch cuda:', torch.cuda.is_available(), '| threads:', torch.get_num_threads())
except Exception:
    pass
