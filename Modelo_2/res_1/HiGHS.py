from __future__ import annotations

import inspect
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pulp


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


def criar_var(
    prob: pulp.LpProblem,
    nome: str,
    lowBound: Optional[float] = None,
    upBound: Optional[float] = None,
    cat: str = pulp.LpContinuous,
) -> pulp.LpVariable:
    """Cria variavel compativel com PuLP 4.x (prob.add_variable) e PuLP 3.x."""
    if hasattr(prob, "add_variable"):
        return prob.add_variable(nome, lowBound=lowBound, upBound=upBound, cat=cat)
    return pulp.LpVariable(nome, lowBound=lowBound, upBound=upBound, cat=cat)


def resolver_lp(
    prob: pulp.LpProblem, solver: pulp.LpSolver
) -> Tuple[str, str, Optional[float]]:
    """Resolve o problema e classifica o resultado em PuLP 4.x e PuLP 3.x.

    Retorna (estado, status_str, gap_rel), com estado em:
      "otimo"                    : otimalidade provada;
      "inviavel"                 : inviabilidade provada;
      "limite_tempo"             : limite de tempo, mas ha solucao viavel (nao provada otima);
      "limite_tempo_sem_solucao" : limite de tempo sem nenhuma solucao viavel (nada foi provado);
      "erro"                     : qualquer outro status (ilimitado, falha do solver etc.).
    """
    stats = prob.solve(solver)
    if hasattr(pulp, "LpSolveStatus"):
        status_str = str(stats.status_str)
        if stats.status == pulp.LpSolveStatus.Optimal:
            return "otimo", status_str, 0.0
        baixo = status_str.lower()
        if "infeasible" in baixo:
            return "inviavel", status_str, None
        if "timelimit" in baixo.replace(" ", ""):
            gap = getattr(stats, "gap_rel", None)
            if getattr(stats, "has_solution", False):
                return "limite_tempo", status_str, gap
            return "limite_tempo_sem_solucao", status_str, None
        return "erro", status_str, None
    # PuLP 3.x
    status_str = pulp.LpStatus[prob.status]
    if prob.status == pulp.LpStatusOptimal:
        return "otimo", status_str, 0.0
    if prob.status == pulp.LpStatusInfeasible:
        return "inviavel", status_str, None
    return "erro", status_str, None


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


def _solver_highs(time_limit: Optional[float] = None, warm_start: bool = True) -> pulp.LpSolver:
    for cls in (getattr(pulp, "HiGHS", None), getattr(pulp, "HiGHS_CMD", None)):
        if cls is None:
            continue
        sig = inspect.signature(cls.__init__).parameters
        kw: Dict[str, Any] = {"msg": False}
        if time_limit is not None and "timeLimit" in sig:
            kw["timeLimit"] = time_limit
        if warm_start and "warmStart" in sig:
            kw["warmStart"] = True
        s = cls(**kw)
        if s.available():
            return s
    raise RuntimeError("HiGHS indisponivel. Instale com: pip install pulp[highs]")


def resolver_modelo_2_highs(
    m2: Modelo2, modo: str = "obrigatorio", etapa4: bool = True,
    eps: float = 1e-6, time_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve as etapas lexicograficas do Modelo 2 reaproveitando o mesmo LpProblem."""
    x, y, r = len(m2.servidores), len(m2.servicos), 4
    t0 = time.perf_counter()
    # "otimo": todas as etapas provadas otimas; "falha": motivo de parada sem solucao
    estado_global: Dict[str, Any] = {"otimo": True, "falha": None, "tem_inicial": False}

    mdl = pulp.LpProblem("Modelo2_HiGHS", pulp.LpMinimize)
    a = [[criar_var(mdl, f"a_{i}_{j}", cat=pulp.LpBinary) for j in range(y)] for i in range(x)]

    for j in range(y):                                                 # (R1) Admissao
        soma = pulp.lpSum(a[i][j] for i in range(x))
        mdl += (soma == 1) if modo == "obrigatorio" else (soma <= 1)
    for i in range(x):                                                 # (R3) Elegibilidade
        for j in range(y):
            mdl += a[i][j] <= m2.E[i][j]
    for i in range(x):                                                 # (R2) Capacidade pico
        for k in range(r):
            mdl += pulp.lpSum(m2.Dp_ef[i][j][k] * a[i][j] for j in range(y) if m2.E[i][j]) <= m2.C[i][k]

    u = [[pulp.lpSum(m2.Dm_ef[i][j][k] * a[i][j] for j in range(y) if m2.E[i][j]) / m2.C[i][k]
          for k in range(r)] for i in range(x)]
    incl_expr = pulp.lpSum(m2.p[j] * a[i][j] for i in range(x) for j in range(y)) / sum(m2.p)

    def rodar(obj: Any) -> Optional[List[List[int]]]:
        mdl.setObjective(obj)
        solver = _solver_highs(time_limit, warm_start=estado_global["tem_inicial"])
        estado, _, _ = resolver_lp(mdl, solver)
        if estado == "inviavel":
            estado_global["falha"] = "INVIAVEL"
            return None
        if estado in ("limite_tempo_sem_solucao", "erro"):
            # Nao se afirma inviabilidade: o solver apenas nao concluiu.
            estado_global["falha"] = "LIMITE_TEMPO" if estado != "erro" else "ERRO"
            return None
        if estado == "limite_tempo":
            estado_global["otimo"] = False
        sol = [[int(round(pulp.value(a[i][j]) or 0.0)) for j in range(y)] for i in range(x)]
        for i in range(x):
            for j in range(y):
                a[i][j].setInitialValue(sol[i][j])
        estado_global["tem_inicial"] = True
        return sol

    if modo == "opcional":                                             # Etapa 1: max Incl == min(-Incl)
        sol = rodar(-1.0 * incl_expr)
        if sol is None:
            return {"viavel": False, "status": estado_global["falha"], "tempo_s": time.perf_counter() - t0}
        mdl += incl_expr >= avaliar_solucao(m2, sol, modo)["Incl"] - eps

    s = [[criar_var(mdl, f"s_{i}_{k}", lowBound=0.0) for k in range(r)] for i in range(x)]
    for i in range(x):                                                 # Etapa 2: QoS / folga S
        for k in range(r):
            mdl += s[i][k] >= u[i][k] - m2.tau[k]
    S_expr = pulp.lpSum(s[i][k] for i in range(x) for k in range(r)) / (x * r)
    sol = rodar(S_expr)
    if sol is None:
        return {"viavel": False, "status": estado_global["falha"], "tempo_s": time.perf_counter() - t0}
    mdl += S_expr <= avaliar_solucao(m2, sol, modo)["S"] + eps

    U_max = criar_var(mdl, "U_max", lowBound=0.0, upBound=1.0)         # Etapa 3: Gargalo Min-Max
    for i in range(x):
        for k in range(r):
            mdl += U_max >= u[i][k]
    sol = rodar(U_max)
    if sol is None:
        return {"viavel": False, "status": estado_global["falha"], "tempo_s": time.perf_counter() - t0}

    if etapa4:                                                         # Etapa 4: Equilibrio L1
        mdl += U_max <= avaliar_solucao(m2, sol, modo)["U_max"] + eps
        ubar = [pulp.lpSum(u[i][k] for i in range(x)) / x for k in range(r)]
        d = [[criar_var(mdl, f"d_{i}_{k}", lowBound=0.0) for k in range(r)] for i in range(x)]
        for i in range(x):
            for k in range(r):
                mdl += d[i][k] >= u[i][k] - ubar[k]
                mdl += d[i][k] >= ubar[k] - u[i][k]
        sol = rodar(pulp.lpSum(d[i][k] for i in range(x) for k in range(r)) / (x * r))
        if sol is None:
            return {"viavel": False, "status": estado_global["falha"], "tempo_s": time.perf_counter() - t0}

    return {"viavel": True, "otimo": estado_global["otimo"],
            "status": "OTIMO" if estado_global["otimo"] else "LIMITE_TEMPO",
            "tempo_s": time.perf_counter() - t0, **avaliar_solucao(m2, sol, modo)}


def mostrar(titulo: str, m2: Modelo2, r: Dict[str, Any]) -> None:
    if not r["viavel"]:
        sem_eleg = [m2.servicos[j] for j in range(len(m2.servicos))
                    if sum(m2.E[i][j] for i in range(len(m2.servidores))) == 0]
        extra = f" | sem elegivel={sem_eleg}" if sem_eleg else ""
        if r["status"] == "INVIAVEL":
            print(f"{titulo}: INVIAVEL (provado){extra} | t={r['tempo_s']:.4f}s")
        else:
            print(f"{titulo}: {r['status']} (nada provado: nenhuma solucao viavel dentro do limite) | t={r['tempo_s']:.4f}s")
    else:
        fora = f" | fora={r['nao_alocados']}" if r["nao_alocados"] else ""
        st = "OTIMO" if r["otimo"] else "VIAVEL (limite de tempo, nao provada otima)"
        print(f"{titulo}: {st} | OK={r['valido']} | Incl={r['Incl']:.4f} | S={r['S']:.6f} | "
              f"Umax={r['U_max']:.4f} | D={r['D']:.6f} | X86={r['arq']['X86']}, ARM={r['arq']['ARM']}"
              f"{fora} | t={r['tempo_s']:.4f}s")


def cenario_4_misto_6x24(seed: int = 42, v_arm: float = 0.51) -> Modelo2:
    """Cluster misto 6x24 (Ice Lake-SP, Zen 3 e Ampere Altra - Ansari et al., 2024)."""
    rng = random.Random(seed)
    perfis = [
        ("WebPar",   (3.0, 6.5), (1.20, 1.40), (0.85, 0.96), (30.0, 50.0), (80.0, 120.0), 64.0, (0.85, 1.00), (0.95, 1.10)),
        ("ApiMista", (2.5, 5.0), (1.20, 1.40), (0.45, 0.72), (30.0, 42.0), (52.0,  72.0), 32.0, (0.75, 0.95), (0.95, 1.05)),
        ("SeqLat",   (1.5, 3.8), (1.15, 1.35), (0.15, 0.35), (35.0, 44.0), (48.0,  60.0), 16.0, (0.70, 0.90), (0.95, 1.05)),
    ]
    servs, p, Wp, Wm, phi, T1, L, q, e, A, Dp_o, Dm_o = ([] for _ in range(12))

    for j in range(24):
        nome, r_wm, r_wp, r_phi, r_t1, r_l, q_j, r_ea, r_ex = perfis[j // 8]
        servs.append(f"{nome}_{(j % 8) + 1:02d}")
        wm_j = round(rng.uniform(*r_wm), 2)
        Wm.append(wm_j); Wp.append(round(wm_j * rng.uniform(*r_wp), 2))
        phi.append(round(rng.uniform(*r_phi), 2))
        T1.append(round(rng.uniform(*r_t1), 1)); L.append(round(rng.uniform(*r_l), 1))
        q.append(q_j)
        e.append({"ARM": round(rng.uniform(*r_ea), 2), "X86": round(rng.uniform(*r_ex), 2)})
        A.append({"ARM": 0 if j >= 22 else 1, "X86": 1})
        rm, ram, dsk = round(rng.uniform(1.0, 3.2), 2), round(rng.uniform(6.0, 22.0), 2), round(rng.uniform(40.0, 150.0), 2)
        Dm_o.append([rm, ram, dsk])
        Dp_o.append([round(rm * rng.uniform(1.20, 1.40), 2), round(ram * rng.uniform(1.15, 1.35), 2),
                     round(dsk * rng.uniform(1.10, 1.25), 2)])
        p.append(float(rng.choice([1, 2, 3, 5])))

    return construir_modelo_2(
        ["S1_IceLake", "S2_IceLake", "S3_Zen3", "S4_Altra", "S5_Altra", "S6_Altra"],
        ["X86", "X86", "X86", "ARM", "ARM", "ARM"],
        [32.0, 32.0, 64.0, 80.0, 80.0, 80.0],
        [1.00, 1.00, 0.87, v_arm, v_arm, v_arm],
        [[25, 128, 1000], [25, 128, 1000], [25, 256, 1500],
         [25, 256, 1500], [25, 256, 1500], [25, 256, 1500]],
        servs, p, Wp, Wm, phi, T1, L, q, e, A, Dp_o, Dm_o,
    )


if __name__ == "__main__":
    # Simulacao 1: Especializacao natural ARM vs x86 (2x3)
    c1 = construir_modelo_2(
        ["S1_x86", "S2_ARM"], ["X86", "ARM"], [32, 80], [1.0, 0.5],
        [[20, 128, 1000], [20, 128, 1000]], ["J1_Web1", "J2_Web2", "J3_SeqRigido"],
        [1, 1, 1], [10, 10, 16], [8, 8, 12], [0.95, 0.95, 0.30], [40, 40, 40],
        [100, 100, 50], [64, 64, 64], [{"ARM": 1, "X86": 1}] * 3, [{"ARM": 1, "X86": 1}] * 3,
        [[4, 32, 200], [4, 32, 200], [6, 40, 250]], [[2, 16, 100], [2, 16, 100], [4, 32, 200]],
    )
    r1 = resolver_modelo_2_highs(c1, "obrigatorio")
    mostrar("Sim 1  (ARM vs x86)", c1, r1)
    assert r1["viavel"] and r1["valido"] and c1.E[1][2] == 0
    assert r1["alocados"] == {"J1_Web1": "S2_ARM", "J2_Web2": "S2_ARM", "J3_SeqRigido": "S1_x86"}
    assert abs(r1["S"]) < 1e-6 and abs(r1["U_max"] - 0.40) < 1e-6 and abs(r1["D"] - 0.003125) < 1e-6

    # Simulacao 2: Regras de pre-processamento (A[j,t], q[j], c_lat no pico e e[j,t])
    c2 = construir_modelo_2(
        ["S1_x86", "S2_ARM"], ["X86", "ARM"], [16, 40], [1.0, 0.5],
        [[20, 64, 500], [20, 64, 500]],
        ["J1_LegadoX86", "J2_LimiteQ", "J3_LatenciaDomina", "J4_EscalaBaixaARM"],
        [1, 1, 1, 1], [6, 4, 2, 8], [4, 3, 2, 6], [0.80, 0.30, 0.30, 0.90],
        [20, 40, 40, 20], [50, 60, 60, 60], [16, 4, 8, 32],
        [{"ARM": 1, "X86": 1}, {"ARM": 1, "X86": 1}, {"ARM": 1, "X86": 1}, {"ARM": 0.5, "X86": 1}],
        [{"ARM": 0, "X86": 1}, {"ARM": 1, "X86": 1}, {"ARM": 1, "X86": 1}, {"ARM": 1, "X86": 1}],
        [[4, 16, 100], [2, 8, 50], [2, 8, 50], [4, 16, 100]],
        [[2, 8, 50], [1, 4, 25], [1, 4, 25], [2, 8, 50]],
    )
    r2 = resolver_modelo_2_highs(c2, "obrigatorio")
    mostrar("Sim 2  (Pre-proc.) ", c2, r2)
    assert r2["viavel"] and r2["valido"] and c2.E[1] == [0, 0, 1, 1]
    assert abs(c2.Dp_ef[1][2][0] - 6.0) < 1e-6 and abs(c2.Dm_ef[1][2][0] - 4.0) < 1e-6
    assert abs(c2.Dp_ef[1][3][0] - 32.0) < 1e-6
    assert r2["alocados"]["J3_LatenciaDomina"] == "S1_x86" and r2["alocados"]["J4_EscalaBaixaARM"] == "S2_ARM"
    assert abs(r2["S"]) < 1e-6 and abs(r2["U_max"] - 0.60) < 1e-6 and abs(r2["D"] - 0.0453125) < 1e-6

    # Simulacao 3A e 3B: Cluster so ARM (obrigatorio inviavel -> diagnostico opcional)
    c3 = construir_modelo_2(
        ["S1_ARM", "S2_ARM"], ["ARM", "ARM"], [40, 40], [0.5, 0.5],
        [[20, 64, 500], [20, 64, 500]], ["J1_WebAlto", "J2_WebMedio", "J3_SemElegivel"],
        [5, 3, 10], [10, 10, 2], [8, 8, 1], [0.95, 0.95, 0.20], [40, 40, 40],
        [100, 100, 50], [64, 64, 64], [{"ARM": 1, "X86": 1}] * 3, [{"ARM": 1, "X86": 1}] * 3,
        [[4, 16, 100], [4, 16, 100], [1, 4, 20]], [[2, 8, 50], [2, 8, 50], [1, 4, 20]],
    )
    r3a = resolver_modelo_2_highs(c3, "obrigatorio")
    r3b = resolver_modelo_2_highs(c3, "opcional")
    mostrar("Sim 3A (So ARM obr)", c3, r3a)
    mostrar("Sim 3B (So ARM opc)", c3, r3b)
    assert not r3a["viavel"] and r3a["status"] == "INVIAVEL" and r3b["viavel"] and r3b["valido"]
    assert abs(r3b["Incl"] - 8 / 18) < 1e-6 and abs(r3b["U_max"] - 0.40) < 1e-6
    assert abs(r3b["D"]) < 1e-6 and r3b["nao_alocados"] == ["J3_SemElegivel"]

    # Simulacao 4: Cluster misto 6x24 (144 binarias)
    c4 = cenario_4_misto_6x24(seed=42, v_arm=0.51)
    r4 = resolver_modelo_2_highs(c4, "obrigatorio", time_limit=30.0)
    mostrar("Sim 4  (Misto 6x24)", c4, r4)

    # Simulacao 5: Analise de sensibilidade em v_ARM
    print("\nSim 5 (Sensibilidade v_ARM no cluster 6x24):")
    print(f"  {'v_ARM':>5} {'Eleg_ARM':>8} {'X86':>5} {'ARM':>5} {'S (QoS)':>10} {'U_max':>8} {'Desvio D':>10} {'Tempo(s)':>9}")
    for v_t in (0.45, 0.51, 0.65, 0.80):
        cs = cenario_4_misto_6x24(seed=42, v_arm=v_t)
        rs = resolver_modelo_2_highs(cs, "obrigatorio", time_limit=30.0)
        eleg_arm = sum(cs.E[3][j] for j in range(len(cs.servicos)))
        if rs["viavel"]:
            print(f"  {v_t:>5.2f} {eleg_arm:>8d} {rs['arq']['X86']:>5d} {rs['arq']['ARM']:>5d} "
                  f"{rs['S']:>10.6f} {rs['U_max']:>8.4f} {rs['D']:>10.6f} {rs['tempo_s']:>9.4f}"
                  + ("" if rs["otimo"] else "  (limite de tempo)"))
        else:
            print(f"  {v_t:>5.2f} {eleg_arm:>8d} {rs['status']:>25}")

    print("\nTodas as verificacoes analiticas do Modelo 2 passaram.")