from __future__ import annotations

import inspect
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pulp


@dataclass
class InstanciaModelo1:
    """Estrutura de entrada do Modelo 1 (alocacao multidimensional base)."""

    servidores: List[str]
    servicos: List[str]
    C: List[List[float]]   # C[i][k]: capacidade fisica do recurso k no servidor i
    Dp: List[List[float]]  # Dp[j][k]: demanda de pico do recurso k pelo servico j
    Dm: List[List[float]]  # Dm[j][k]: demanda media do recurso k pelo servico j
    p: List[float]         # p[j]: peso de importancia do servico j
    tau: List[float] = field(default_factory=lambda: [0.75, 0.75, 0.75, 0.75])
    recursos: List[str] = field(
        default_factory=lambda: ["CPU", "Rede", "RAM", "Armazenamento"]
    )


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


def obter_solver_highs(
    time_limit: Optional[float] = None,
    msg: bool = False,
    warm_start: bool = True,
) -> pulp.LpSolver:
    """Seleciona a interface HiGHS disponivel no PuLP (highspy ou binario)."""
    for classe_solver in (getattr(pulp, "HiGHS", None), getattr(pulp, "HiGHS_CMD", None)):
        if classe_solver is None:
            continue
        params = inspect.signature(classe_solver.__init__).parameters
        kwargs: Dict[str, Any] = {"msg": msg}
        if time_limit is not None and "timeLimit" in params:
            kwargs["timeLimit"] = time_limit
        if warm_start and "warmStart" in params:
            kwargs["warmStart"] = True
        solver = classe_solver(**kwargs)
        if solver.available():
            return solver
    raise RuntimeError(
        "Solver HiGHS nao encontrado pelo PuLP. Instale com: pip install pulp[highs]"
    )


def calcular_metricas(
    inst: InstanciaModelo1,
    alocacao: List[List[int]],
) -> Dict[str, Any]:
    """Recalcula de forma independente todas as metricas a partir da matriz a[i][j]."""
    x, y, r = len(inst.servidores), len(inst.servicos), len(inst.recursos)

    u = [
        [sum(inst.Dm[j][k] * alocacao[i][j] for j in range(y)) / inst.C[i][k] for k in range(r)]
        for i in range(x)
    ]
    uso_pico = [
        [sum(inst.Dp[j][k] * alocacao[i][j] for j in range(y)) / inst.C[i][k] for k in range(r)]
        for i in range(x)
    ]
    ubar = [sum(u[i][k] for i in range(x)) / x for k in range(r)]
    incl = sum(inst.p[j] * alocacao[i][j] for i in range(x) for j in range(y)) / sum(inst.p)
    S = sum(max(0.0, u[i][k] - inst.tau[k]) for i in range(x) for k in range(r)) / (x * r)
    u_max = max(u[i][k] for i in range(x) for k in range(r))
    D = sum(abs(u[i][k] - ubar[k]) for i in range(x) for k in range(r)) / (x * r)

    alocados = {
        inst.servicos[j]: inst.servidores[i]
        for i in range(x)
        for j in range(y)
        if alocacao[i][j] == 1
    }
    nao_alocados = [s for s in inst.servicos if s not in alocados]

    return {
        "Incl": incl,
        "S": S,
        "U_max": u_max,
        "D": D,
        "u": u,
        "ubar": ubar,
        "uso_pico": uso_pico,
        "alocados": alocados,
        "nao_alocados": nao_alocados,
    }


def validar_solucao(
    inst: InstanciaModelo1,
    alocacao: List[List[int]],
    modo: str,
    tol: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """Audita as restricoes (R1), (R2) e dominio binario da alocacao obtida."""
    x, y, r = len(inst.servidores), len(inst.servicos), len(inst.recursos)
    erros: List[str] = []

    for j in range(y):
        soma_j = sum(alocacao[i][j] for i in range(x))
        if modo == "obrigatorio" and soma_j != 1:
            erros.append(f"Violacao (R1) obrigatoria em {inst.servicos[j]}: soma={soma_j}")
        if modo == "opcional" and soma_j not in (0, 1):
            erros.append(f"Violacao (R1) opcional em {inst.servicos[j]}: soma={soma_j}")

    for i in range(x):
        for k in range(r):
            pico = sum(inst.Dp[j][k] * alocacao[i][j] for j in range(y))
            if pico > inst.C[i][k] + tol:
                erros.append(
                    f"Violacao (R2) em ({inst.servidores[i]}, {inst.recursos[k]}): "
                    f"pico={pico:.4f} > C={inst.C[i][k]:.4f}"
                )

    return len(erros) == 0, erros


def resolver_modelo_1(
    inst: InstanciaModelo1,
    modo: str = "obrigatorio",
    executar_etapa_4: bool = True,
    eps: float = 1e-6,
    time_limit: Optional[float] = None,
    msg: bool = False,
) -> Dict[str, Any]:
    """Resolve o Modelo 1 em etapas lexicograficas usando PuLP + HiGHS."""
    if modo not in ("obrigatorio", "opcional"):
        raise ValueError("modo deve ser 'obrigatorio' ou 'opcional'.")

    x, y, r = len(inst.servidores), len(inst.servicos), len(inst.recursos)
    peso_total = sum(inst.p)

    historico: List[Dict[str, Any]] = []
    a_atual: Optional[List[List[int]]] = None
    incl_star, s_star, umax_star = 1.0, 0.0, 1.0

    etapas = ([1] if modo == "opcional" else []) + [2, 3] + ([4] if executar_etapa_4 else [])
    nomes_etapa = {
        1: "Etapa 1 (Inclusao)",
        2: "Etapa 2 (QoS / S)",
        3: "Etapa 3 (Gargalo)",
        4: "Etapa 4 (Equilibrio)",
    }

    for etapa in etapas:
        sentido = pulp.LpMaximize if etapa == 1 else pulp.LpMinimize
        prob = pulp.LpProblem(f"Modelo1_Etapa{etapa}", sentido)

        a = [
            [criar_var(prob, f"a_{i}_{j}", cat=pulp.LpBinary) for j in range(y)]
            for i in range(x)
        ]
        if a_atual is not None:
            for i in range(x):
                for j in range(y):
                    a[i][j].setInitialValue(a_atual[i][j])

        # (R1) Admissao
        for j in range(y):
            soma = pulp.lpSum(a[i][j] for i in range(x))
            prob += (soma == 1, f"R1_j{j}") if modo == "obrigatorio" else (soma <= 1, f"R1_j{j}")

        # (R2) Capacidade multidimensional no pico
        for i in range(x):
            for k in range(r):
                prob += (
                    pulp.lpSum(inst.Dp[j][k] * a[i][j] for j in range(y)) <= inst.C[i][k],
                    f"R2_i{i}_k{k}",
                )

        u = [
            [pulp.lpSum(inst.Dm[j][k] * a[i][j] for j in range(y)) / inst.C[i][k] for k in range(r)]
            for i in range(x)
        ]
        incl_expr = pulp.lpSum(inst.p[j] * a[i][j] for i in range(x) for j in range(y)) / peso_total

        if etapa == 1:
            prob.setObjective(incl_expr)
        else:
            if modo == "opcional":
                prob += (incl_expr >= incl_star - eps, "Fix_Incl")
            s = [[criar_var(prob, f"s_{i}_{k}", lowBound=0.0) for k in range(r)] for i in range(x)]
            for i in range(x):
                for k in range(r):
                    prob += (s[i][k] >= u[i][k] - inst.tau[k], f"R3_i{i}_k{k}")
            S_expr = pulp.lpSum(s[i][k] for i in range(x) for k in range(r)) / (x * r)

            if etapa == 2:
                prob.setObjective(S_expr)
            else:
                prob += (S_expr <= s_star + eps, "Fix_S")
                U_max = criar_var(prob, "U_max", lowBound=0.0, upBound=1.0)
                for i in range(x):
                    for k in range(r):
                        prob += (U_max >= u[i][k], f"R4_i{i}_k{k}")

                if etapa == 3:
                    prob.setObjective(U_max)
                else:
                    prob += (U_max <= umax_star + eps, "Fix_Umax")
                    ubar = [pulp.lpSum(u[i][k] for i in range(x)) / x for k in range(r)]
                    d = [
                        [criar_var(prob, f"d_{i}_{k}", lowBound=0.0) for k in range(r)]
                        for i in range(x)
                    ]
                    for i in range(x):
                        for k in range(r):
                            prob += (d[i][k] >= u[i][k] - ubar[k], f"Desv_pos_i{i}_k{k}")
                            prob += (d[i][k] >= ubar[k] - u[i][k], f"Desv_neg_i{i}_k{k}")
                    prob.setObjective(pulp.lpSum(d[i][k] for i in range(x) for k in range(r)) / (x * r))

        # Warm start so quando ha solucao anterior (evita aviso do PuLP na primeira etapa)
        solver = obter_solver_highs(
            time_limit=time_limit, msg=msg, warm_start=a_atual is not None
        )
        t0 = time.perf_counter()
        estado, status_str, gap = resolver_lp(prob, solver)
        dt = time.perf_counter() - t0
        if estado == "inviavel":
            return {
                "viavel": False,
                "etapa_falha": etapa,
                "status": "INVIAVEL",
                "status_solver": status_str,
                "historico": historico,
            }
        if estado in ("limite_tempo_sem_solucao", "erro"):
            # Nao se afirma inviabilidade: o solver apenas nao concluiu.
            return {
                "viavel": False,
                "etapa_falha": etapa,
                "status": "LIMITE_TEMPO" if estado != "erro" else "ERRO",
                "status_solver": status_str,
                "historico": historico,
            }
        # estado "otimo" ou "limite_tempo" (com solucao viavel, nao provada otima)

        a_atual = [
            [int(round(pulp.value(a[i][j]) or 0.0)) for j in range(y)]
            for i in range(x)
        ]
        m = calcular_metricas(inst, a_atual)
        if etapa == 1:
            incl_star = m["Incl"]
        elif etapa == 2:
            s_star = m["S"]
        elif etapa == 3:
            umax_star = m["U_max"]
        historico.append({
            "etapa": nomes_etapa[etapa],
            "tempo_s": dt,
            "status": "OTIMO" if estado == "otimo" else "LIMITE_TEMPO",
            "gap_rel": gap,
            **m,
        })

    assert a_atual is not None
    ok, erros = validar_solucao(inst, a_atual, modo=modo)
    return {
        "viavel": True,
        "otimo_global": all(h["status"] == "OTIMO" for h in historico),
        "valido": ok,
        "erros_validacao": erros,
        "alocacao": a_atual,
        "metricas": historico[-1],
        "historico": historico,
        "tempo_total_s": sum(h["tempo_s"] for h in historico),
    }


def imprimir_relatorio(titulo: str, inst: InstanciaModelo1, res: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(titulo)
    print("=" * 78)
    if not res["viavel"]:
        if res["status"] == "INVIAVEL":
            print(f"Resultado: INVIAVEL (provado) na Etapa {res['etapa_falha']} "
                  f"(status HiGHS: {res['status_solver']}).")
        else:
            print(f"Resultado: {res['status']} na Etapa {res['etapa_falha']} "
                  f"(status HiGHS: {res['status_solver']}). Nada foi provado: "
                  f"nenhuma solucao viavel foi encontrada dentro do limite.")
        return

    rotulo = "OTIMO" if res["otimo_global"] else "SOLUCAO VIAVEL (limite de tempo em alguma etapa)"
    print(
        f"Status: {rotulo} | Auditoria de restricoes: {'OK' if res['valido'] else 'FALHOU'} | "
        f"Tempo total: {res['tempo_total_s']:.4f} s"
    )
    for erro in res["erros_validacao"]:
        print(f"  [ERRO] {erro}")

    print("\nEvolucao por etapa lexicografica:")
    print(f"  {'Etapa':<22} {'Status':<13} {'Incl':>8} {'S (QoS)':>10} {'U_max':>9} {'Desvio D':>10} {'Gap%':>7} {'Tempo (s)':>10}")
    for h in res["historico"]:
        gap_pct = (h["gap_rel"] or 0.0) * 100.0
        print(
            f"  {h['etapa']:<22} {h['status']:<13} {h['Incl']:>8.4f} {h['S']:>10.6f} "
            f"{h['U_max']:>9.4f} {h['D']:>10.6f} {gap_pct:>7.2f} {h['tempo_s']:>10.4f}"
        )

    m = res["metricas"]
    print("\nUtilizacao media (u) e pico (p) por servidor:")
    for i, srv in enumerate(inst.servidores):
        u_fmt = ", ".join(
            f"{inst.recursos[k]}={m['u'][i][k]*100:4.1f}% (pico {m['uso_pico'][i][k]*100:4.1f}%)"
            for k in range(len(inst.recursos))
        )
        print(f"  {srv}: {u_fmt}")

    if m["nao_alocados"]:
        print(f"\nServicos nao alocados: {', '.join(m['nao_alocados'])}")


# ==========================================================================
# SIMULACOES DE VALIDACAO
# ==========================================================================

def cenario_1_desempate_etapa_4() -> InstanciaModelo1:
    return InstanciaModelo1(
        servidores=["S1_Grande", "S2_Medio", "S3_Medio"],
        servicos=["J1_Pesado", "J2_Web", "J3_Web", "J4_Leve", "J5_Leve"],
        C=[[20, 20, 64, 200], [10, 10, 32, 100], [10, 10, 32, 100]],
        Dm=[[12, 8, 24, 80], [4, 2, 8, 20], [4, 2, 8, 20], [1, 1, 4, 10], [1, 1, 4, 10]],
        Dp=[[16, 10, 32, 100], [5, 3, 10, 25], [5, 3, 10, 25], [2, 2, 6, 15], [2, 2, 6, 15]],
        p=[1, 1, 1, 1, 1],
    )


def cenario_2_sobrecarga_diagnostico() -> InstanciaModelo1:
    return InstanciaModelo1(
        servidores=["S1", "S2"],
        servicos=["J1_Critico", "J2_Alto", "J3_Medio", "J4_Baixo"],
        C=[[8, 10, 32, 100], [8, 10, 32, 100]],
        Dm=[[5, 3, 12, 30], [5, 3, 12, 30], [2, 2, 8, 20], [4, 2, 8, 20]],
        Dp=[[6, 4, 16, 40], [6, 4, 16, 40], [2, 3, 10, 25], [5, 3, 10, 25]],
        p=[5, 4, 3, 1],
    )


def cenario_3_separacao_pico_media() -> InstanciaModelo1:
    return InstanciaModelo1(
        servidores=["S1", "S2"],
        servicos=["J1_PicoAlto", "J2_PicoAlto", "J3_Auxiliar"],
        C=[[10, 10, 32, 100], [10, 10, 32, 100]],
        Dm=[[3, 2, 8, 20], [3, 2, 8, 20], [2, 2, 6, 15]],
        Dp=[[6, 4, 16, 40], [6, 4, 16, 40], [4, 3, 12, 30]],
        p=[1, 1, 1],
    )


def cenario_4_cluster_medio_aleatorio(seed: int = 42) -> InstanciaModelo1:
    rng = random.Random(seed)
    servidores = ["S1_G", "S2_G", "S3_M", "S4_M", "S5_P", "S6_P"]
    C = [
        [32.0, 25.0, 128.0, 1000.0], [32.0, 25.0, 128.0, 1000.0],
        [16.0, 10.0, 64.0, 500.0],   [16.0, 10.0, 64.0, 500.0],
        [8.0, 10.0, 32.0, 250.0],    [8.0, 10.0, 32.0, 250.0],
    ]
    Dm, Dp, p = [], [], []
    for _ in range(24):
        cm, rm, ram, dsk = (
            round(rng.uniform(1.0, 4.0), 2), round(rng.uniform(0.8, 3.0), 2),
            round(rng.uniform(4.0, 16.0), 2), round(rng.uniform(30.0, 120.0), 2),
        )
        Dm.append([cm, rm, ram, dsk])
        Dp.append([
            round(cm * rng.uniform(1.20, 1.45), 2), round(rm * rng.uniform(1.20, 1.45), 2),
            round(ram * rng.uniform(1.15, 1.35), 2), round(dsk * rng.uniform(1.10, 1.25), 2),
        ])
        p.append(float(rng.choice([1, 2, 3, 5])))
    return InstanciaModelo1(
        servidores=servidores,
        servicos=[f"Srv_{j+1:02d}" for j in range(24)],
        C=C, Dm=Dm, Dp=Dp, p=p,
    )


if __name__ == "__main__":
    inst1 = cenario_1_desempate_etapa_4()
    res1 = resolver_modelo_1(inst1, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio("SIMULACAO 1: Heterogeneo viavel e efeito da Etapa 4", inst1, res1)
    assert res1["viavel"] and res1["valido"]
    assert abs(res1["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res1["metricas"]["U_max"] - 0.60) < 1e-6
    assert abs(res1["metricas"]["D"] - (1.0 / 30.0)) < 1e-5

    inst2 = cenario_2_sobrecarga_diagnostico()
    res2a = resolver_modelo_1(inst2, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio("SIMULACAO 2A: Sobrecarga no modo obrigatorio", inst2, res2a)
    assert not res2a["viavel"] and res2a["status"] == "INVIAVEL"

    res2b = resolver_modelo_1(inst2, modo="opcional", executar_etapa_4=True)
    imprimir_relatorio("SIMULACAO 2B: Diagnostico no modo opcional com pesos", inst2, res2b)
    assert res2b["viavel"] and res2b["valido"]
    assert abs(res2b["metricas"]["Incl"] - (12.0 / 13.0)) < 1e-6
    assert abs(res2b["metricas"]["S"] - 0.015625) < 1e-6
    assert abs(res2b["metricas"]["U_max"] - 0.875) < 1e-6
    assert res2b["metricas"]["nao_alocados"] == ["J4_Baixo"]

    inst3 = cenario_3_separacao_pico_media()
    res3 = resolver_modelo_1(inst3, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio("SIMULACAO 3: Separacao entre pico e media", inst3, res3)
    assert res3["viavel"] and res3["valido"]
    assert abs(res3["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res3["metricas"]["U_max"] - 0.50) < 1e-6

    inst4 = cenario_4_cluster_medio_aleatorio(seed=42)
    res4 = resolver_modelo_1(inst4, modo="obrigatorio", executar_etapa_4=True, time_limit=30.0)
    imprimir_relatorio("SIMULACAO 4: Cluster heterogeneo medio (6x24)", inst4, res4)

    print("\nTodas as verificacoes analiticas dos Cenarios 1, 2 e 3 passaram.")