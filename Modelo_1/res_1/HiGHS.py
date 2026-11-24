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

    def __post_init__(self) -> None:
        x = len(self.servidores)
        y = len(self.servicos)
        r = len(self.recursos)
        if len(self.C) != x or any(len(linha) != r for linha in self.C):
            raise ValueError("Dimensao de C deve ser (x, |K|).")
        if len(self.Dp) != y or any(len(linha) != r for linha in self.Dp):
            raise ValueError("Dimensao de Dp deve ser (y, |K|).")
        if len(self.Dm) != y or any(len(linha) != r for linha in self.Dm):
            raise ValueError("Dimensao de Dm deve ser (y, |K|).")
        if len(self.p) != y or any(peso <= 0 for peso in self.p):
            raise ValueError("Todos os pesos p[j] devem ser estritamente positivos.")
        if len(self.tau) != r or any(not (0.0 < t < 1.0) for t in self.tau):
            raise ValueError("Todos os limiares tau[k] devem estar em (0, 1).")
        for i in range(x):
            for k in range(r):
                if self.C[i][k] <= 0:
                    raise ValueError(f"C[{i}][{k}] deve ser > 0.")
        for j in range(y):
            for k in range(r):
                if self.Dm[j][k] < 0 or self.Dm[j][k] > self.Dp[j][k]:
                    raise ValueError(
                        f"Exige-se 0 <= Dm[{j}][{k}] <= Dp[{j}][{k}]."
                    )


def obter_solver_highs(
    time_limit: Optional[float] = None,
    msg: bool = False,
    warm_start: bool = True,
) -> pulp.LpSolver:
    """Seleciona a interface HiGHS disponivel no PuLP (highspy ou binario)."""
    candidatos = [
        getattr(pulp, "HiGHS", None),
        getattr(pulp, "HiGHS_CMD", None),
    ]
    for classe_solver in candidatos:
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
        "Solver HiGHS nao encontrado pelo PuLP. Instale com: pip install pulp highspy"
    )


def calcular_metricas(
    inst: InstanciaModelo1,
    alocacao: List[List[int]],
) -> Dict[str, Any]:
    """Recalcula de forma independente todas as metricas a partir da matriz a[i][j]."""
    x = len(inst.servidores)
    y = len(inst.servicos)
    r = len(inst.recursos)

    u = [[0.0 for _ in range(r)] for _ in range(x)]
    uso_pico = [[0.0 for _ in range(r)] for _ in range(x)]
    for i in range(x):
        for k in range(r):
            soma_m = sum(inst.Dm[j][k] * alocacao[i][j] for j in range(y))
            soma_p = sum(inst.Dp[j][k] * alocacao[i][j] for j in range(y))
            u[i][k] = soma_m / inst.C[i][k]
            uso_pico[i][k] = soma_p / inst.C[i][k]

    ubar = [sum(u[i][k] for i in range(x)) / x for k in range(r)]
    peso_total = sum(inst.p)
    peso_admitido = sum(
        inst.p[j] * alocacao[i][j] for i in range(x) for j in range(y)
    )
    incl = peso_admitido / peso_total
    s_matriz = [
        [max(0.0, u[i][k] - inst.tau[k]) for k in range(r)] for i in range(x)
    ]
    S = sum(s_matriz[i][k] for i in range(x) for k in range(r)) / (x * r)
    u_max = max(u[i][k] for i in range(x) for k in range(r))
    D = sum(abs(u[i][k] - ubar[k]) for i in range(x) for k in range(r)) / (x * r)

    alocados: Dict[str, str] = {}
    nao_alocados: List[str] = []
    for j in range(y):
        servs = [inst.servidores[i] for i in range(x) if alocacao[i][j] == 1]
        if servs:
            alocados[inst.servicos[j]] = servs[0]
        else:
            nao_alocados.append(inst.servicos[j])

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
    x = len(inst.servidores)
    y = len(inst.servicos)
    r = len(inst.recursos)
    erros: List[str] = []

    for j in range(y):
        soma_j = sum(alocacao[i][j] for i in range(x))
        if modo == "obrigatorio" and soma_j != 1:
            erros.append(
                f"Violacao (R1) obrigatoria no servico {inst.servicos[j]}: soma={soma_j}"
            )
        if modo == "opcional" and soma_j not in (0, 1):
            erros.append(
                f"Violacao (R1) opcional no servico {inst.servicos[j]}: soma={soma_j}"
            )

    for i in range(x):
        for k in range(r):
            demanda_pico = sum(inst.Dp[j][k] * alocacao[i][j] for j in range(y))
            if demanda_pico > inst.C[i][k] + tol:
                erros.append(
                    f"Violacao (R2) em ({inst.servidores[i]}, {inst.recursos[k]}): "
                    f"pico={demanda_pico:.4f} > C={inst.C[i][k]:.4f}"
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

    x = len(inst.servidores)
    y = len(inst.servicos)
    r = len(inst.recursos)
    peso_total = sum(inst.p)
    solver = obter_solver_highs(time_limit=time_limit, msg=msg, warm_start=True)

    def montar_base(
        nome: str,
        sentido: int,
        a_inicial: Optional[List[List[int]]] = None,
    ) -> Tuple[
        pulp.LpProblem,
        List[List[pulp.LpVariable]],
        List[List[pulp.LpAffineExpression]],
        List[pulp.LpAffineExpression],
        pulp.LpAffineExpression,
    ]:
        prob = pulp.LpProblem(nome, sentido)
        a = [
            [pulp.LpVariable(f"a_{i}_{j}", cat=pulp.LpBinary) for j in range(y)]
            for i in range(x)
        ]
        if a_inicial is not None:
            for i in range(x):
                for j in range(y):
                    a[i][j].setInitialValue(a_inicial[i][j])

        # (R1) Admissao
        for j in range(y):
            soma_serv = pulp.lpSum(a[i][j] for i in range(x))
            if modo == "obrigatorio":
                prob += (soma_serv == 1, f"R1_obrig_j{j}")
            else:
                prob += (soma_serv <= 1, f"R1_opc_j{j}")

        # (R2) Capacidade multidimensional no pico
        for i in range(x):
            for k in range(r):
                prob += (
                    pulp.lpSum(inst.Dp[j][k] * a[i][j] for j in range(y))
                    <= inst.C[i][k],
                    f"R2_cap_i{i}_k{k}",
                )

        # Expressoes lineares auxiliares u[i,k], ubar[k] e Incl
        u = [
            [
                pulp.lpSum(inst.Dm[j][k] * a[i][j] for j in range(y)) / inst.C[i][k]
                for k in range(r)
            ]
            for i in range(x)
        ]
        ubar = [pulp.lpSum(u[i][k] for i in range(x)) / x for k in range(r)]
        incl_expr = (
            pulp.lpSum(
                inst.p[j] * a[i][j] for i in range(x) for j in range(y)
            )
            / peso_total
        )
        return prob, a, u, ubar, incl_expr

    def extrair_a(a: List[List[pulp.LpVariable]]) -> List[List[int]]:
        return [
            [int(round(pulp.value(a[i][j]) or 0.0)) for j in range(y)]
            for i in range(x)
        ]

    historico: List[Dict[str, Any]] = []
    a_atual: Optional[List[List[int]]] = None
    incl_star = 1.0

    # ------------------------------------------------------------------
    # ETAPA 1: INCLUSAO (apenas no modo opcional)
    # ------------------------------------------------------------------
    if modo == "opcional":
        prob1, a1, _, _, incl_expr1 = montar_base("Modelo1_Etapa1", pulp.LpMaximize)
        prob1 += incl_expr1, "Obj_Inclusao"
        t0 = time.perf_counter()
        prob1.solve(solver)
        dt = time.perf_counter() - t0
        if prob1.status != pulp.LpStatusOptimal:
            return {
                "viavel": False,
                "etapa_falha": 1,
                "status": pulp.LpStatus[prob1.status],
                "historico": historico,
            }
        a_atual = extrair_a(a1)
        m1 = calcular_metricas(inst, a_atual)
        incl_star = m1["Incl"]
        historico.append(
            {"etapa": "Etapa 1 (Inclusao)", "tempo_s": dt, **m1}
        )

    # ------------------------------------------------------------------
    # ETAPA 2: QUALIDADE DE SERVICO (minimizar folga media S)
    # ------------------------------------------------------------------
    prob2, a2, u2, _, incl_expr2 = montar_base(
        "Modelo1_Etapa2", pulp.LpMinimize, a_inicial=a_atual
    )
    s2 = [
        [pulp.LpVariable(f"s_{i}_{k}", lowBound=0.0) for k in range(r)]
        for i in range(x)
    ]
    for i in range(x):
        for k in range(r):
            prob2 += (s2[i][k] >= u2[i][k] - inst.tau[k], f"R3_folga_i{i}_k{k}")
    if modo == "opcional":
        prob2 += (incl_expr2 >= incl_star - eps, "Fix_Incl")

    S_expr2 = pulp.lpSum(s2[i][k] for i in range(x) for k in range(r)) / (x * r)
    prob2 += S_expr2, "Obj_QoS"

    t0 = time.perf_counter()
    prob2.solve(solver)
    dt = time.perf_counter() - t0
    if prob2.status != pulp.LpStatusOptimal:
        return {
            "viavel": False,
            "etapa_falha": 2,
            "status": pulp.LpStatus[prob2.status],
            "historico": historico,
        }
    a_atual = extrair_a(a2)
    m2 = calcular_metricas(inst, a_atual)
    s_star = m2["S"]
    historico.append({"etapa": "Etapa 2 (QoS / S)", "tempo_s": dt, **m2})

    # ------------------------------------------------------------------
    # ETAPA 3: GARGALO (minimizar U_max)
    # ------------------------------------------------------------------
    prob3, a3, u3, _, incl_expr3 = montar_base(
        "Modelo1_Etapa3", pulp.LpMinimize, a_inicial=a_atual
    )
    s3 = [
        [pulp.LpVariable(f"s_{i}_{k}", lowBound=0.0) for k in range(r)]
        for i in range(x)
    ]
    U_max3 = pulp.LpVariable("U_max", lowBound=0.0, upBound=1.0)
    for i in range(x):
        for k in range(r):
            prob3 += (s3[i][k] >= u3[i][k] - inst.tau[k], f"R3_folga_i{i}_k{k}")
            prob3 += (U_max3 >= u3[i][k], f"R4_gargalo_i{i}_k{k}")
    if modo == "opcional":
        prob3 += (incl_expr3 >= incl_star - eps, "Fix_Incl")
    S_expr3 = pulp.lpSum(s3[i][k] for i in range(x) for k in range(r)) / (x * r)
    prob3 += (S_expr3 <= s_star + eps, "Fix_S")
    prob3 += U_max3, "Obj_Gargalo"

    t0 = time.perf_counter()
    prob3.solve(solver)
    dt = time.perf_counter() - t0
    if prob3.status != pulp.LpStatusOptimal:
        return {
            "viavel": False,
            "etapa_falha": 3,
            "status": pulp.LpStatus[prob3.status],
            "historico": historico,
        }
    a_atual = extrair_a(a3)
    m3 = calcular_metricas(inst, a_atual)
    umax_star = m3["U_max"]
    historico.append({"etapa": "Etapa 3 (Gargalo)", "tempo_s": dt, **m3})

    # ------------------------------------------------------------------
    # ETAPA 4: EQUILIBRIO ENTRE SERVIDORES (opcional, desvio absoluto L1)
    # ------------------------------------------------------------------
    if executar_etapa_4:
        prob4, a4, u4, ubar4, incl_expr4 = montar_base(
            "Modelo1_Etapa4", pulp.LpMinimize, a_inicial=a_atual
        )
        s4 = [
            [pulp.LpVariable(f"s_{i}_{k}", lowBound=0.0) for k in range(r)]
            for i in range(x)
        ]
        U_max4 = pulp.LpVariable("U_max", lowBound=0.0, upBound=1.0)
        d4 = [
            [pulp.LpVariable(f"d_{i}_{k}", lowBound=0.0) for k in range(r)]
            for i in range(x)
        ]
        for i in range(x):
            for k in range(r):
                prob4 += (s4[i][k] >= u4[i][k] - inst.tau[k], f"R3_folga_i{i}_k{k}")
                prob4 += (U_max4 >= u4[i][k], f"R4_gargalo_i{i}_k{k}")
                prob4 += (d4[i][k] >= u4[i][k] - ubar4[k], f"Desv_pos_i{i}_k{k}")
                prob4 += (d4[i][k] >= ubar4[k] - u4[i][k], f"Desv_neg_i{i}_k{k}")

        if modo == "opcional":
            prob4 += (incl_expr4 >= incl_star - eps, "Fix_Incl")
        S_expr4 = pulp.lpSum(s4[i][k] for i in range(x) for k in range(r)) / (x * r)
        prob4 += (S_expr4 <= s_star + eps, "Fix_S")
        prob4 += (U_max4 <= umax_star + eps, "Fix_Umax")

        D_expr4 = pulp.lpSum(d4[i][k] for i in range(x) for k in range(r)) / (x * r)
        prob4 += D_expr4, "Obj_Equilibrio"

        t0 = time.perf_counter()
        prob4.solve(solver)
        dt = time.perf_counter() - t0
        if prob4.status != pulp.LpStatusOptimal:
            return {
                "viavel": False,
                "etapa_falha": 4,
                "status": pulp.LpStatus[prob4.status],
                "historico": historico,
            }
        a_atual = extrair_a(a4)
        m4 = calcular_metricas(inst, a_atual)
        historico.append({"etapa": "Etapa 4 (Equilibrio)", "tempo_s": dt, **m4})

    assert a_atual is not None
    ok, erros = validar_solucao(inst, a_atual, modo=modo)
    metricas_finais = historico[-1]
    tempo_total = sum(h["tempo_s"] for h in historico)

    return {
        "viavel": True,
        "valido": ok,
        "erros_validacao": erros,
        "alocacao": a_atual,
        "metricas": metricas_finais,
        "historico": historico,
        "tempo_total_s": tempo_total,
    }


def imprimir_relatorio(titulo: str, inst: InstanciaModelo1, res: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(titulo)
    print("=" * 78)
    if not res["viavel"]:
        print(
            f"Resultado: INVIAVEL na Etapa {res['etapa_falha']} "
            f"(status HiGHS: {res['status']})."
        )
        return

    print(
        f"Status: OTIMO | Auditoria de restricoes: "
        f"{'OK' if res['valido'] else 'FALHOU'} | "
        f"Tempo total: {res['tempo_total_s']:.4f} s"
    )
    if res["erros_validacao"]:
        for erro in res["erros_validacao"]:
            print(f"  [ERRO] {erro}")

    print("\nEvolucao por etapa lexicografica:")
    print(
        f"  {'Etapa':<22} {'Incl':>8} {'S (QoS)':>10} {'U_max':>9} "
        f"{'Desvio D':>10} {'Tempo (s)':>10}"
    )
    for h in res["historico"]:
        print(
            f"  {h['etapa']:<22} {h['Incl']:>8.4f} {h['S']:>10.6f} "
            f"{h['U_max']:>9.4f} {h['D']:>10.6f} {h['tempo_s']:>10.4f}"
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
    """Cluster heterogeneo viavel com empate em U_max=0.60 quebrado pela Etapa 4."""
    return InstanciaModelo1(
        servidores=["S1_Grande", "S2_Medio", "S3_Medio"],
        servicos=["J1_Pesado", "J2_Web", "J3_Web", "J4_Leve", "J5_Leve"],
        C=[
            [20, 20, 64, 200],
            [10, 10, 32, 100],
            [10, 10, 32, 100],
        ],
        Dm=[
            [12, 8, 24, 80],
            [4, 2, 8, 20],
            [4, 2, 8, 20],
            [1, 1, 4, 10],
            [1, 1, 4, 10],
        ],
        Dp=[
            [16, 10, 32, 100],
            [5, 3, 10, 25],
            [5, 3, 10, 25],
            [2, 2, 6, 15],
            [2, 2, 6, 15],
        ],
        p=[1, 1, 1, 1, 1],
        tau=[0.75, 0.75, 0.75, 0.75],
    )


def cenario_2_sobrecarga_diagnostico() -> InstanciaModelo1:
    """Cluster sobrecarregado: inviavel no obrigatorio, prioriza pesos no opcional."""
    return InstanciaModelo1(
        servidores=["S1", "S2"],
        servicos=["J1_Critico", "J2_Alto", "J3_Medio", "J4_Baixo"],
        C=[
            [8, 10, 32, 100],
            [8, 10, 32, 100],
        ],
        Dm=[
            [5, 3, 12, 30],
            [5, 3, 12, 30],
            [2, 2, 8, 20],
            [4, 2, 8, 20],
        ],
        Dp=[
            [6, 4, 16, 40],
            [6, 4, 16, 40],
            [2, 3, 10, 25],
            [5, 3, 10, 25],
        ],
        p=[5, 4, 3, 1],
        tau=[0.75, 0.75, 0.75, 0.75],
    )


def cenario_3_separacao_pico_media() -> InstanciaModelo1:
    """Servicos com picos altos: R2 impede sobrecarga fisica mantendo media segura."""
    return InstanciaModelo1(
        servidores=["S1", "S2"],
        servicos=["J1_PicoAlto", "J2_PicoAlto", "J3_Auxiliar"],
        C=[
            [10, 10, 32, 100],
            [10, 10, 32, 100],
        ],
        Dm=[
            [3, 2, 8, 20],
            [3, 2, 8, 20],
            [2, 2, 6, 15],
        ],
        Dp=[
            [6, 4, 16, 40],
            [6, 4, 16, 40],
            [4, 3, 12, 30],
        ],
        p=[1, 1, 1],
        tau=[0.75, 0.75, 0.75, 0.75],
    )


def cenario_4_cluster_medio_aleatorio(seed: int = 42) -> InstanciaModelo1:
    """Instancia sintetica heterogenea (6 servidores, 24 servicos) com semente fixa."""
    rng = random.Random(seed)
    servidores = [
        "S1_G", "S2_G", "S3_M", "S4_M", "S5_P", "S6_P"
    ]
    C = [
        [32.0, 25.0, 128.0, 1000.0],
        [32.0, 25.0, 128.0, 1000.0],
        [16.0, 10.0, 64.0, 500.0],
        [16.0, 10.0, 64.0, 500.0],
        [8.0, 10.0, 32.0, 250.0],
        [8.0, 10.0, 32.0, 250.0],
    ]
    y = 24
    servicos = [f"Srv_{j+1:02d}" for j in range(y)]
    Dm: List[List[float]] = []
    Dp: List[List[float]] = []
    p: List[float] = []

    for _ in range(y):
        cpu_m = round(rng.uniform(1.0, 4.0), 2)
        rede_m = round(rng.uniform(0.8, 3.0), 2)
        ram_m = round(rng.uniform(4.0, 16.0), 2)
        arm_m = round(rng.uniform(30.0, 120.0), 2)

        cpu_p = round(cpu_m * rng.uniform(1.20, 1.45), 2)
        rede_p = round(rede_m * rng.uniform(1.20, 1.45), 2)
        ram_p = round(ram_m * rng.uniform(1.15, 1.35), 2)
        arm_p = round(arm_m * rng.uniform(1.10, 1.25), 2)

        Dm.append([cpu_m, rede_m, ram_m, arm_m])
        Dp.append([cpu_p, rede_p, ram_p, arm_p])
        p.append(float(rng.choice([1, 2, 3, 5])))

    return InstanciaModelo1(
        servidores=servidores,
        servicos=servicos,
        C=C,
        Dm=Dm,
        Dp=Dp,
        p=p,
        tau=[0.75, 0.75, 0.75, 0.75],
    )


if __name__ == "__main__":
    # Simulacao 1: Modo obrigatorio com desempate da Etapa 4
    inst1 = cenario_1_desempate_etapa_4()
    res1 = resolver_modelo_1(inst1, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio("SIMULACAO 1: Heterogeneo viavel e efeito da Etapa 4", inst1, res1)

    # Verificacao automatica contra o gabarito analitico do Cenario 1
    assert res1["viavel"] and res1["valido"]
    assert abs(res1["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res1["metricas"]["U_max"] - 0.60) < 1e-6
    assert abs(res1["metricas"]["D"] - (1.0 / 30.0)) < 1e-5

    # Simulacao 2A e 2B: Sobrecarga (obrigatorio inviavel -> diagnostico opcional)
    inst2 = cenario_2_sobrecarga_diagnostico()
    res2a = resolver_modelo_1(inst2, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio("SIMULACAO 2A: Sobrecarga no modo obrigatorio", inst2, res2a)
    assert not res2a["viavel"]

    res2b = resolver_modelo_1(inst2, modo="opcional", executar_etapa_4=True)
    imprimir_relatorio("SIMULACAO 2B: Diagnostico no modo opcional com pesos", inst2, res2b)
    assert res2b["viavel"] and res2b["valido"]
    assert abs(res2b["metricas"]["Incl"] - (12.0 / 13.0)) < 1e-6
    assert abs(res2b["metricas"]["S"] - 0.015625) < 1e-6
    assert abs(res2b["metricas"]["U_max"] - 0.875) < 1e-6
    assert res2b["metricas"]["nao_alocados"] == ["J4_Baixo"]

    # Simulacao 3: Separacao entre demanda de pico e media
    inst3 = cenario_3_separacao_pico_media()
    res3 = resolver_modelo_1(inst3, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio("SIMULACAO 3: Separacao entre pico e media", inst3, res3)
    assert res3["viavel"] and res3["valido"]
    assert abs(res3["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res3["metricas"]["U_max"] - 0.50) < 1e-6

    # Simulacao 4: Instancia heterogenea de medio porte (6 servidores x 24 servicos)
    inst4 = cenario_4_cluster_medio_aleatorio(seed=42)
    res4 = resolver_modelo_1(inst4, modo="obrigatorio", executar_etapa_4=True, time_limit=30.0)
    imprimir_relatorio("SIMULACAO 4: Cluster heterogeneo medio (6x24)", inst4, res4)

    print("\nTodas as verificacoes analiticas dos Cenarios 1, 2 e 3 passaram.")