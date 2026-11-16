import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from review5 import ARCH, TARGETS, models, condm


def decomposition(a, P):
    P("== 1. BBSE nevus-prior error decomposed by true class, all three targets (mean over 3 seeds x 5 fold models) ==")
    P("contribution of class j = A[nv, j] * pi_T(j), A = C_val^-1 (C_T - C_val); points of the nevus prior")
    rows = []
    for tg, ood in TARGETS.items():
        for arch, base in ARCH.items():
            M = models(a.runs, ood, base)
            if M is None:
                continue
            pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]; K = len(cl); n = cl.index("nv")
            con, tot, est = [], [], []
            for V, T, O in M:
                Cv = condm(V["y"].values, V[pc].values.argmax(1), K); Ct = condm(O["y"].values, O[pc].values.argmax(1), K)
                A = np.linalg.pinv(Cv) @ (Ct - Cv)
                piT = np.bincount(O["y"].values, minlength=K) / len(O)
                c = A[n] * piT * 100
                con.append(c); tot.append(c.sum())
                qT = np.bincount(O[pc].values.argmax(1), minlength=K) / len(O)
                est.append(((np.linalg.pinv(Cv) @ qT)[n] - piT[n]) * 100)
            con = np.mean(con, 0)
            r = {"target": tg, "arch": arch, "total": f"{np.mean(tot):+.1f}", "BBSE unclipped": f"{np.mean(est):+.1f}"}
            r.update({cl[j]: f"{con[j]:+.1f}" for j in range(K)})
            rows.append(r)
    P(pd.DataFrame(rows).to_string(index=False))


def rates(a, P):
    P("\n== 2. per-class P(pred = nv | y) and P(pred != y | y = nv), source -> target, and target class shares (ResNet-50, mean of 15 models) ==")
    for tg, ood in TARGETS.items():
        M = models(a.runs, ood, ARCH["ResNet-50"])
        if M is None:
            continue
        pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]; K = len(cl); n = cl.index("nv")
        rows = []
        for j in range(K):
            s, t = [], []
            for V, T, O in M:
                for D, L in ((T, s), (O, t)):
                    mk = D["y"].values == j
                    L.append(np.mean(D[pc].values[mk].argmax(1) == n) * 100 if mk.any() else np.nan)
            share = np.mean(M[0][2]["y"].values == j) * 100
            lab = "nv -> not nv" if j == n else f"{cl[j]} -> nv"
            if j == n:
                s, t = [100 - v for v in s], [100 - v for v in t]
            rows.append({"class": cl[j], "rate": lab, "source %": f"{np.nanmean(s):.1f}",
                         "target %": f"{np.nanmean(t):.1f}" if share > 0 else "absent", "target share %": f"{share:.1f}"})
        P(f"-- {tg}"); P(pd.DataFrame(rows).to_string(index=False))


def simple_rule(a, P):
    P("\n== 3. melanoma-to-nevus rate and melanoma sensitivity under the simple rule argmax q/m_S (mean over 15 models; [min, max] of the 15) ==")
    rows = []
    for tg, ood in TARGETS.items():
        for arch, base in ARCH.items():
            M = models(a.runs, ood, base)
            if M is None:
                continue
            pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]
            m, n = cl.index("mel"), cl.index("nv")
            r = {k: [] for k in ("raw_s", "raw_t", "sim_s", "sim_t", "sens_s", "sens_t")}
            for V, T, O in M:
                mS = V[pc].values.mean(0)
                for D, sfx in ((T, "s"), (O, "t")):
                    y = D["y"].values == m; q = D[pc].values[y]
                    r["raw_" + sfx].append(np.mean(q.argmax(1) == n) * 100)
                    h = (q / mS).argmax(1)
                    r["sim_" + sfx].append(np.mean(h == n) * 100); r["sens_" + sfx].append(np.mean(h == m) * 100)
            inc_raw = np.array(r["raw_t"]) - np.array(r["raw_s"]); inc_sim = np.array(r["sim_t"]) - np.array(r["sim_s"])
            rows.append({"target": tg, "arch": arch,
                         "raw mel->nv s->t": f"{np.mean(r['raw_s']):.1f} -> {np.mean(r['raw_t']):.1f}",
                         "simple mel->nv s->t": f"{np.mean(r['sim_s']):.1f} -> {np.mean(r['sim_t']):.1f}",
                         "increase raw": f"{inc_raw.mean():+.1f}",
                         "increase simple": f"{inc_sim.mean():+.1f} [{inc_sim.min():+.1f}, {inc_sim.max():+.1f}]",
                         "simple mel sens s->t": f"{np.mean(r['sens_s']):.1f} -> {np.mean(r['sens_t']):.1f}"})
    P(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--out", default="review9_results.txt")
    a = ap.parse_args()
    L = []; P = L.append
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    for name, fn in (("decomposition", decomposition), ("rates", rates), ("simple rule", simple_rule)):
        try:
            fn(a, P); print(f"{name} done", flush=True)
        except Exception as ex:
            P(f"!! {name} failed: {type(ex).__name__}: {ex}"); print(f"!! {name} failed: {ex}", flush=True)
        open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
