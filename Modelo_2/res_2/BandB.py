from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass
class Modelo2:
    """Estrutura linear do Modelo 2 apos o pre-processamento (P1)-(P5)."""
    servidores: List[str]
    arch: List[str]
    servicos: List[str]
    C: List[List[float]]               # C[i][k], com C[i][0] = m[i]
    Dp_ef: List[List[List[float]]]     # Dp_ef[i][j][k]
    Dm_ef: List[List[List[float]]]     # Dm_ef[i][j][k]
    E: List[List[int]]                 # E[i][j] em {0, 1}
    p: List[float]
    tau: List[float]


def construir_modelo_2(
    servidores: List[str], arch: List[str], m: List[float], v: List[float],
    C_out: List[List[float]], servicos: List[str], p: List[float],
    Wp: List[float], Wm: List[float], phi: List[float], T1: List[float],
    L: List[float], q: List[float], e: List[Dict[str, float]],
    A: List[Dict[str, int]], Dp_out: List[List[float]], Dm_out: List[List[float]],
    tau: Optional[List[float]] = None, dedicados: bool = False,
) -> Modelo2:
    """Executa o pre-processamento (P1)-(P5) do Modelo 2 (ARM e x86)."""
    x, y = len(servidores), len(servicos)
    C = [[m[i], *C_out[i]] for i in range(x)]
    E = [[0] * y for _ in range(x)]
    Dp_ef = [[[0.0] * 4 for _ in range(y)] for _ in range(x)]
    Dm_ef = [[[0.0] * 4 for _ in range(y)] for _ in range(x)]

    for i in range(x):
        arq = arch[i]
        for j in range(y):
            g = (L[j] * v[i] / T1[j]) - (1.0 - phi[j])                 # (P1)
            c_lat = max(1.0, phi[j] / g) if g > 0.0 else math.inf      # (P2)
            if dedicados and math.isfinite(c_lat):
                c_lat = float(math.ceil(c_lat))
            if A[j][arq] == 1 and c_lat <= min(q[j], m[i]):            # (P3)
                E[i][j] = 1
                den = v[i] * e[j][arq]
                Dp_ef[i][j] = [max(c_lat, Wp[j] / den), *Dp_out[j]]    # (P4)-(P5)
                Dm_ef[i][j] = [Wm[j] / den, *Dm_out[j]]

    return Modelo2(servidores, arch, servicos, C, Dp_ef, Dm_ef, E, p, tau or [0.75] * 4)


def avaliar_solucao(m2: Modelo2, a: List[List[int]], modo: str) -> Dict[str, Any]:
    """Recalcula metricas e audita as restricoes (R1), (R2) e (R3)."""
    x, y, r = len(m2.servidores), len(m2.servicos), 4
    u = [
        [sum(m2.Dm_ef[i][j][k] * a[i][j] for j in range(y) if m2.E[i][j]) / m2.C[i][k]
         for k in range(r)]
        for i in range(x)
    ]
    ubar = [sum(u[i][k] for i in range(x)) / x for k in range(r)]
    incl = sum(m2.p[j] * a[i][j] for i in range(x) for j in range(y)) / sum(m2.p)
    S = sum(max(0.0, u[i][k] - m2.tau[k]) for i in range(x) for k in range(r)) / (x * r)
    u_max = max(u[i][k] for i in range(x) for k in range(r))
    D = sum(abs(u[i][k] - ubar[k]) for i in range(x) for k in range(r)) / (x * r)

    ok_r1 = all(
        (sum(a[i][j] for i in range(x)) == 1) if modo == "obrigatorio"
        else (sum(a[i][j] for i in range(x)) <= 1)
        for j in range(y)
    )
    ok_r2 = all(
        sum(m2.Dp_ef[i][j][k] * a[i][j] for j in range(y) if m2.E[i][j]) <= m2.C[i][k] + 1e-6
        for i in range(x) for k in range(r)
    )
    ok_r3 = all(a[i][j] <= m2.E[i][j] for i in range(x) for j in range(y))
    aloc = {m2.servicos[j]: m2.servidores[i] for i in range(x) for j in range(y) if a[i][j] == 1}
    fora = [m2.servicos[j] for j in range(y) if m2.servicos[j] not in aloc]
    arq_cnt = {
        "X86": sum(a[i][j] for i in range(x) for j in range(y) if m2.arch[i] == "X86"),
        "ARM": sum(a[i][j] for i in range(x) for j in range(y) if m2.arch[i] == "ARM"),
    }
    return {"Incl": incl, "S": S, "U_max": u_max, "D": D, "valido": ok_r1 and ok_r2 and ok_r3,
            "alocados": aloc, "nao_alocados": fora, "arq": arq_cnt}


def _montar_lp(
    m2: Modelo2, etapa: int, modo: str, eps: float,
    incl_s: float, s_s: float, umax_s: float,
) -> Tuple[Any, ...]:
    """Monta a relaxacao linear (min c^T z) da etapa lexicografica."""
    x, y, r = len(m2.servidores), len(m2.servicos), 4
    n_a, n_s = x * y, x * r
    n_vars = [0, n_a, n_a + n_s, n_a + n_s + 1, n_a + 2 * n_s + 1][etapa]
    idx_u, off_d, p_tot = n_a + n_s, n_a + n_s + 1, sum(m2.p)

    c = np.zeros(n_vars)
    if etapa == 1:
        for i in range(x):
            for j in range(y):
                c[i * y + j] = -m2.p[j] / p_tot
    elif etapa == 2:
        c[n_a:n_a + n_s] = 1.0 / (x * r)
    elif etapa == 3:
        c[idx_u] = 1.0
    else:
        c[off_d:] = 1.0 / (x * r)

    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for j in range(y):                                                 # (R1) Admissao
        row = np.zeros(n_vars)
        row[[i * y + j for i in range(x)]] = 1.0
        (A_eq if modo == "obrigatorio" else A_ub).append(row)
        (b_eq if modo == "obrigatorio" else b_ub).append(1.0)
    for i in range(x):                                                 # (R2) Capacidade pico
        for k in range(r):
            row = np.zeros(n_vars)
            for j in range(y):
                if m2.E[i][j]:
                    row[i * y + j] = m2.Dp_ef[i][j][k]
            A_ub.append(row); b_ub.append(m2.C[i][k])

    if etapa >= 2 and modo == "opcional":                              # Fixacao Etapa 1
        row = np.zeros(n_vars)
        for i in range(x):
            for j in range(y):
                row[i * y + j] = -m2.p[j] / p_tot
        A_ub.append(row); b_ub.append(-(incl_s - eps))
    if etapa >= 2:                                                     # (R4) Zona segura
        for i in range(x):
            for k in range(r):
                row = np.zeros(n_vars)
                for j in range(y):
                    if m2.E[i][j]:
                        row[i * y + j] = m2.Dm_ef[i][j][k] / m2.C[i][k]
                row[n_a + i * r + k] = -1.0
                A_ub.append(row); b_ub.append(m2.tau[k])
    if etapa >= 3:                                                     # Fixacao Etapa 2 + (R5)
        row = np.zeros(n_vars); row[n_a:n_a + n_s] = 1.0 / (x * r)
        A_ub.append(row); b_ub.append(s_s + eps)
        for i in range(x):
            for k in range(r):
                row = np.zeros(n_vars)
                for j in range(y):
                    if m2.E[i][j]:
                        row[i * y + j] = m2.Dm_ef[i][j][k] / m2.C[i][k]
                row[idx_u] = -1.0
                A_ub.append(row); b_ub.append(0.0)
    if etapa == 4:                                                     # Fixacao Etapa 3 + L1
        row = np.zeros(n_vars); row[idx_u] = 1.0
        A_ub.append(row); b_ub.append(umax_s + eps)
        for i in range(x):
            for k in range(r):
                rp, rn = np.zeros(n_vars), np.zeros(n_vars)
                for i2 in range(x):
                    f = (1.0 - 1.0 / x) if i2 == i else (-1.0 / x)
                    for j in range(y):
                        if m2.E[i2][j]:
                            val = f * m2.Dm_ef[i2][j][k] / m2.C[i2][k]
                            rp[i2 * y + j], rn[i2 * y + j] = val, -val
                rp[off_d + i * r + k] = rn[off_d + i * r + k] = -1.0
                A_ub.extend([rp, rn]); b_ub.extend([0.0, 0.0])

    b_cont = (
        ([(0.0, None)] * n_s if etapa >= 2 else [])
        + ([(0.0, 1.0)] if etapa >= 3 else [])
        + ([(0.0, None)] * n_s if etapa == 4 else [])
    )
    return (c, np.vstack(A_ub), np.array(b_ub),
            np.vstack(A_eq) if A_eq else None, np.array(b_eq) if b_eq else None, b_cont)


def _bnb_etapa(
    lp_data: Tuple[Any, ...], raiz: Tuple[int, ...], x: int, y: int,
    inc: Optional[Tuple[float, List[List[int]]]], t_lim: float, max_nos: int,
) -> Tuple[Optional[List[List[int]]], bool, int]:
    """Branch-and-Bound classico (Best-First) com Simplex Dual nos nos."""
    c, A_ub, b_ub, A_eq, b_eq, b_cont = lp_data
    n_bin, t0 = x * y, time.perf_counter()
    best_v, best_a = inc if inc is not None else (math.inf, None)
    nos = 0

    def resolver_no(fix: Tuple[int, ...]) -> Any:
        nonlocal nos
        nos += 1
        bnds = [((0.0, 0.0) if f == 0 else (1.0, 1.0) if f == 1 else (0.0, 1.0)) for f in fix] + b_cont
        return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs-ds")

    def frac_idx(xv: np.ndarray) -> Optional[int]:
        cand = [(abs(xv[k] - 0.5), k) for k in range(n_bin) if abs(xv[k] - round(xv[k])) > 1e-6]
        return min(cand)[1] if cand else None

    def propagar(fix: Tuple[int, ...], idx: int, val: int) -> Optional[Tuple[int, ...]]:
        if (val == 0 and fix[idx] == 1) or (val == 1 and fix[idx] == 0):
            return None
        nv = list(fix); nv[idx] = val
        if val == 1:
            j = idx % y
            for i2 in range(x):
                k2 = i2 * y + j
                if k2 != idx:
                    if nv[k2] == 1:
                        return None
                    nv[k2] = 0
        return tuple(nv)

    res0 = resolver_no(raiz)
    if not res0.success:
        return None, True, nos
    if res0.fun >= best_v - 1e-7:
        return best_a, True, nos
    f0 = frac_idx(res0.x)
    if f0 is None:
        return [[int(round(res0.x[i * y + j])) for j in range(y)] for i in range(x)], True, nos

    heap: List[Tuple[float, int, Tuple[int, ...], int]] = [(float(res0.fun), 0, raiz, f0)]
    cid, interrompido = 0, False
    while heap:
        if heap[0][0] >= best_v - 1e-7:
            break
        if (time.perf_counter() - t0) >= t_lim or nos >= max_nos:
            interrompido = True
            break
        _, _, fix, idx_b = heapq.heappop(heap)
        for val in (1, 0):
            nfix = propagar(fix, idx_b, val)
            if nfix is None:
                continue
            res = resolver_no(nfix)
            if not res.success or res.fun >= best_v - 1e-7:
                continue
            fi = frac_idx(res.x)
            if fi is None:
                if res.fun < best_v - 1e-9:
                    best_v = float(res.fun)
                    best_a = [[int(round(res.x[i * y + j])) for j in range(y)] for i in range(x)]
            else:
                cid += 1
                heapq.heappush(heap, (float(res.fun), cid, nfix, fi))
    return best_a, not interrompido, nos


def resolver_modelo_2_bnb(
    m2: Modelo2, modo: str = "obrigatorio", etapa4: bool = True,
    eps: float = 1e-6, t_lim: float = 15.0, max_nos: int = 20000,
) -> Dict[str, Any]:
    """Resolve as etapas lexicograficas do Modelo 2 via Branch-and-Bound."""
    x, y = len(m2.servidores), len(m2.servicos)
    raiz = tuple(0 if m2.E[i][j] == 0 else -1 for i in range(x) for j in range(y))
    t0, nos_tot, otimo_all = time.perf_counter(), 0, True
    a_cur: Optional[List[List[int]]] = None
    incl_s, s_s, umax_s = 1.0, 0.0, 1.0

    etapas = ([1] if modo == "opcional" else []) + [2, 3] + ([4] if etapa4 else [])
    chave_inc = {2: "S", 3: "U_max", 4: "D"}
    for et in etapas:
        inc = (avaliar_solucao(m2, a_cur, modo)[chave_inc[et]], a_cur) if (a_cur and et > 1) else None
        lp = _montar_lp(m2, et, modo, eps, incl_s, s_s, umax_s)
        a_cur, opt, nos = _bnb_etapa(lp, raiz, x, y, inc, t_lim, max_nos)
        nos_tot += nos; otimo_all = otimo_all and opt
        if a_cur is None:
            # So se afirma inviabilidade quando a arvore foi esgotada (opt=True);
            # caso contrario o limite de tempo/nos parou a busca sem provar nada.
            return {"viavel": False, "status": "INVIAVEL" if opt else "LIMITE_TEMPO",
                    "otimo": otimo_all, "nos": nos_tot, "tempo_s": time.perf_counter() - t0}
        m = avaliar_solucao(m2, a_cur, modo)
        if et == 1: incl_s = m["Incl"]
        elif et == 2: s_s = m["S"]
        elif et == 3: umax_s = m["U_max"]

    return {"viavel": True, "status": "OTIMO" if otimo_all else "LIMITE_TEMPO",
            "otimo": otimo_all, "nos": nos_tot, "tempo_s": time.perf_counter() - t0,
            **avaliar_solucao(m2, a_cur, modo)}


def mostrar(titulo: str, r: Dict[str, Any]) -> None:
    if not r["viavel"]:
        if r["status"] == "INVIAVEL":
            print(f"{titulo}: INVIAVEL (provado) | nos={r['nos']} | t={r['tempo_s']:.4f}s")
        else:
            print(f"{titulo}: {r['status']} (nada provado: nenhuma solucao viavel dentro do limite) "
                  f"| nos={r['nos']} | t={r['tempo_s']:.4f}s")
    else:
        st = "OTIMO" if r["otimo"] else "VIAVEL (limite atingido, nao provada otima)"
        print(f"{titulo}: {st} | OK={r['valido']} | Incl={r['Incl']:.4f} | S={r['S']:.5f} | "
              f"Umax={r['U_max']:.4f} | D={r['D']:.6f} | X86={r['arq']['X86']}, ARM={r['arq']['ARM']} | "
              f"nos={r['nos']} | t={r['tempo_s']:.4f}s")


if __name__ == "__main__":
    # Simulacao 1: Especializacao ARM vs x86 (2x3)
    c1 = construir_modelo_2(
        ["S1_x86", "S2_ARM"], ["X86", "ARM"], [32, 80], [1.0, 0.5],
        [[20, 128, 1000], [20, 128, 1000]], ["J1_Web1", "J2_Web2", "J3_Seq"],
        [1, 1, 1], [10, 10, 16], [8, 8, 12], [0.95, 0.95, 0.30], [40, 40, 40],
        [100, 100, 50], [64, 64, 64], [{"ARM": 1, "X86": 1}] * 3, [{"ARM": 1, "X86": 1}] * 3,
        [[4, 32, 200], [4, 32, 200], [6, 40, 250]], [[2, 16, 100], [2, 16, 100], [4, 32, 200]],
    )
    r1 = resolver_modelo_2_bnb(c1, "obrigatorio")
    mostrar("Sim 1 (ARM vs x86)", r1)
    assert r1["viavel"] and r1["alocados"]["J3_Seq"] == "S1_x86" and abs(r1["D"] - 0.003125) < 1e-6

    # Simulacao 2: Cluster so ARM com servico inelegivel (obrigatorio -> opcional)
    c2 = construir_modelo_2(
        ["S1_ARM", "S2_ARM"], ["ARM", "ARM"], [40, 40], [0.5, 0.5],
        [[20, 64, 500], [20, 64, 500]], ["J1_WebA", "J2_WebB", "J3_SemEleg"],
        [5, 3, 10], [10, 10, 2], [8, 8, 1], [0.95, 0.95, 0.20], [40, 40, 40],
        [100, 100, 50], [64, 64, 64], [{"ARM": 1, "X86": 1}] * 3, [{"ARM": 1, "X86": 1}] * 3,
        [[4, 16, 100], [4, 16, 100], [1, 4, 20]], [[2, 8, 50], [2, 8, 50], [1, 4, 20]],
    )
    r2a = resolver_modelo_2_bnb(c2, "obrigatorio")
    r2b = resolver_modelo_2_bnb(c2, "opcional")
    mostrar("Sim 2A (So ARM obrig.)", r2a)
    mostrar("Sim 2B (So ARM opc.)  ", r2b)
    assert not r2a["viavel"] and r2a["status"] == "INVIAVEL" and r2b["nao_alocados"] == ["J3_SemEleg"] and abs(r2b["Incl"] - 8 / 18) < 1e-6

    # Simulacao 3: Cluster misto 3x9 (Ice Lake-SP, Zen 3 e Ampere Altra)
    rng = random.Random(7)
    servs = [f"Srv_{j+1:02d}" for j in range(9)]
    Wm = [round(rng.uniform(2.0, 6.0), 2) for _ in range(9)]
    Wp = [round(w * 1.25, 2) for w in Wm]
    Dm_o = [[round(rng.uniform(1.5, 3.0), 2), round(rng.uniform(8.0, 20.0), 2), round(rng.uniform(50.0, 120.0), 2)] for _ in range(9)]
    Dp_o = [[round(d[0] * 1.3, 2), round(d[1] * 1.25, 2), round(d[2] * 1.15, 2)] for d in Dm_o]
    c3 = construir_modelo_2(
        ["S1_IceLake", "S2_Zen3", "S3_Altra"], ["X86", "X86", "ARM"],
        [32, 64, 80], [1.0, 0.87, 0.51], [[25, 128, 1000], [25, 192, 1200], [25, 192, 1200]],
        servs, [float(rng.choice([1, 2, 3, 5])) for _ in range(9)], Wp, Wm,
        [0.92, 0.92, 0.92, 0.60, 0.60, 0.60, 0.25, 0.25, 0.25], [40] * 9,
        [100, 100, 100, 65, 65, 65, 52, 52, 52], [32] * 9,
        [{"ARM": 0.90, "X86": 1.0}] * 9, [{"ARM": 0 if j == 8 else 1, "X86": 1} for j in range(9)],
        Dp_o, Dm_o,
    )
    mostrar("Sim 3 (Misto 3x9)     ", resolver_modelo_2_bnb(c3, "obrigatorio"))