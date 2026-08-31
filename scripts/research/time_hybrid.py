"""Latensi hybrid: panggilan pertama (memuat encoder) vs steady state."""
import sys, time
sys.path.insert(0, "/opt/project")
from src.common.model_registry import load_registry

reg = load_registry()
S = "A swordsman betrayed by his sect wakes ten years in the past and begins cultivating a forbidden technique."
T = "Time Travel, Cultivation, Revenge"

t0 = time.time(); reg.predict_genres(S, tags=T, model="hybrid"); first = time.time() - t0
times = []
for i in range(8):
    t0 = time.time(); reg.predict_genres(S + f" Variant {i}.", tags=T, model="hybrid")
    times.append(time.time() - t0)
print("panggilan pertama (muat encoder) : %.0f ms" % (1000 * first))
print("steady state (8 teks berbeda)    : median %.0f ms | maks %.0f ms"
      % (1000 * sorted(times)[len(times)//2], 1000 * max(times)))
t0 = time.time()
for i in range(8):
    reg.predict_genres(S + f" Other {i}.", tags=T, model="svm")
print("pembanding svm                   : median %.0f ms" % (1000 * (time.time()-t0) / 8))
