from __future__ import annotations

import inspect
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pulp


@dataclass
class InstanciaModelo2:
    """Estrutura de entrada do Modelo 2 (heterogeneidade de arquiteturas ARM e x86)."""

    servidores: List[str]
    arch: List[str]                 # arch[i] em {"ARM", "X86"}
    m: List[float]                  # m[i] > 0: nucleos fisicos (C[i, 1] = m[i])
    v: List[float]                  # v[i] > 0: desempenho por nucleo relativo ao x86 ref
    C_outros: List[List[float]]     # [Rede, RAM, Armazenamento] de cada servidor i

    servicos: List[str]
    p: List[float]                  # p[j] > 0: peso de importancia
    Wp: List[float]                 # Wp[j]: carga CPU de pico (nucleos x86 ref)
    Wm: List[float]                 # Wm[j]: carga CPU media (nucleos x86 ref)
    phi: List[float]                # phi[j] em [0, 1]: fracao paralelizavel (Amdahl)
    T1: List[float]                 # T1[j] > 0: tempo de servico em 1 nucleo x86 ref
    L: List[float]                  # L[j] > 0: SLO de latencia (mesma unidade de T1)
    q: List[float]                  # q[j] >= 1: limite de paralelismo por requisicao
    e: List[Dict[str, float]]       # e[j][t] > 0: fator de escalabilidade na arq t
    A: List[Dict[str, int]]         # A[j][t] em {0, 1}: compatibilidade binaria na arq t
    Dp_outros: List[List[float]]    # demanda de pico de [Rede, RAM, Armazenamento]
    Dm_outros: List[List[float]]    # demanda media de [Rede, RAM, Armazenamento]

    tau: List[float] = field(default_factory=lambda: [0.75, 0.75, 0.75, 0.75])
    recursos: List[str] = field(
        default_factory=lambda: ["CPU", "Rede", "RAM", "Armazenamento"]
    )
    nucleos_dedicados: bool = False

    def __post_init__(self) -> None:
        x = len(self.servidores)
        y = len(self.servicos)
        arqs_validas = {"ARM", "X86"}

        if len(self.arch) != x or any(t not in arqs_validas for t in self.arch):
            raise ValueError("arch[i] deve conter 'ARM' ou 'X86' para cada servidor.")
        if len(self.m) != x or any(val <= 0 for val in self.m):
            raise ValueError("Todos os m[i] devem ser estritamente positivos.")
        if len(self.v) != x or any(val <= 0 for val in self.v):
            raise ValueError("Todos os v[i] devem ser estritamente positivos.")
        if len(self.C_outros) != x or any(
            len(linha) != 3 or any(c <= 0 for c in linha) for linha in self.C_outros
        ):
            raise ValueError("C_outros deve ter dimensao (x, 3) com valores > 0.")

        if len(self.p) != y or any(val <= 0 for val in self.p):
            raise ValueError("Todos os pesos p[j] devem ser > 0.")
        if len(self.Wp) != y or len(self.Wm) != y:
            raise ValueError("Wp e Wm devem ter tamanho y.")
        for j in range(y):
            if self.Wm[j] < 0 or self.Wm[j] > self.Wp[j]:
                raise ValueError(f"Exige-se 0 <= Wm[{j}] <= Wp[{j}].")
            if not (0.0 <= self.phi[j] <= 1.0):
                raise ValueError(f"phi[{j}] deve estar em [0, 1].")
            if self.T1[j] <= 0 or self.L[j] <= 0 or self.q[j] < 1:
                raise ValueError(f"Exige-se T1[{j}]>0, L[{j}]>0 e q[{j}]>=1.")
            for t in arqs_validas:
                if t not in self.e[j] or self.e[j][t] <= 0:
                    raise ValueError(f"e[{j}]['{t}'] deve ser > 0.")
                if t not in self.A[j] or self.A[j][t] not in (0, 1):
                    raise ValueError(f"A[{j}]['{t}'] deve ser 0 ou 1.")
            for idx_k in range(3):
                if (
                    self.Dm_outros[j][idx_k] < 0
                    or self.Dm_outros[j][idx_k] > self.Dp_outros[j][idx_k]
                ):
                    raise ValueError(
                        f"Exige-se 0 <= Dm_outros[{j}][{idx_k}] <= Dp_outros[{j}][{idx_k}]."
                    )

        if len(self.tau) != 4 or any(not (0.0 < val < 1.0) for val in self.tau):
            raise ValueError("tau deve ter 4 valores em (0, 1).")


@dataclass
class PreProcessamentoModelo2:
    """Coeficientes constantes derivados antes da otimizacao MILP (P1 a P5)."""

    C: List[List[float]]               # C[i][k] com C[i][0] = m[i]
    g: List[List[float]]               # g[i][j]: folga de latencia
    c_lat: List[List[float]]           # c_lat[i][j]: nucleos minimos para latencia
    E: List[List[int]]                 # E[i][j] em {0, 1}: elegibilidade
    Dp_ef: List[List[List[float]]]     # Dp_ef[i][j][k] (0.0 se E[i][j] == 0)
    Dm_ef: List[List[List[float]]]     # Dm_ef[i][j][k] (0.0 se E[i][j] == 0)


def pre_processar_modelo_2(inst: InstanciaModelo2) -> PreProcessamentoModelo2:
    """Executa as equacoes (P1) a (P5) para cada par servidor-servico (i, j)."""
    x = len(inst.servidores)
    y = len(inst.servicos)

    C = [[inst.m[i], *inst.C_outros[i]] for i in range(x)]
    g = [[0.0 for _ in range(y)] for _ in range(x)]
    c_lat = [[math.inf for _ in range(y)] for _ in range(x)]
    E = [[0 for _ in range(y)] for _ in range(x)]
    Dp_ef = [[[0.0 for _ in range(4)] for _ in range(y)] for _ in range(x)]
    Dm_ef = [[[0.0 for _ in range(4)] for _ in range(y)] for _ in range(x)]

    for i in range(x):
        arq_i = inst.arch[i]
        for j in range(y):
            # (P1) Folga de latencia (Lei de Amdahl)
            g_ij = (inst.L[j] * inst.v[i] / inst.T1[j]) - (1.0 - inst.phi[j])
            g[i][j] = g_ij

            # (P2) Nucleos minimos para cumprir o SLO de latencia
            if g_ij > 0.0:
                c_min = max(1.0, inst.phi[j] / g_ij)
                if inst.nucleos_dedicados:
                    c_min = float(math.ceil(c_min))
                c_lat[i][j] = c_min
            else:
                c_lat[i][j] = math.inf

            # (P3) Elegibilidade (compatibilidade binaria, paralelismo q[j] e nucleos m[i])
            if (
                inst.A[j][arq_i] == 1
                and c_lat[i][j] <= min(inst.q[j], inst.m[i])
            ):
                E[i][j] = 1
                denom = inst.v[i] * inst.e[j][arq_i]
                # (P4) Demanda efetiva de CPU (k = 0)
                Dp_ef[i][j][0] = max(c_lat[i][j], inst.Wp[j] / denom)
                Dm_ef[i][j][0] = inst.Wm[j] / denom
                # (P5) Demais recursos (Rede, RAM, Armazenamento)
                for k in range(1, 4):
                    Dp_ef[i][j][k] = inst.Dp_outros[j][k - 1]
                    Dm_ef[i][j][k] = inst.Dm_outros[j][k - 1]
            else:
                E[i][j] = 0

    return PreProcessamentoModelo2(
        C=C, g=g, c_lat=c_lat, E=E, Dp_ef=Dp_ef, Dm_ef=Dm_ef
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


def calcular_metricas_modelo_2(
    inst: InstanciaModelo2,
    pre: PreProcessamentoModelo2,
    alocacao: List[List[int]],
) -> Dict[str, Any]:
    """Recalcula de forma independente todas as metricas a partir de a[i][j]."""
    x = len(inst.servidores)
    y = len(inst.servicos)
    r = 4

    u = [[0.0 for _ in range(r)] for _ in range(x)]
    uso_pico = [[0.0 for _ in range(r)] for _ in range(x)]
    for i in range(x):
        for k in range(r):
            soma_m = sum(
                pre.Dm_ef[i][j][k] * alocacao[i][j]
                for j in range(y)
                if pre.E[i][j] == 1
            )
            soma_p = sum(
                pre.Dp_ef[i][j][k] * alocacao[i][j]
                for j in range(y)
                if pre.E[i][j] == 1
            )
            u[i][k] = soma_m / pre.C[i][k]
            uso_pico[i][k] = soma_p / pre.C[i][k]

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
    contagem_arq = {"ARM": 0, "X86": 0}
    for j in range(y):
        servs = [i for i in range(x) if alocacao[i][j] == 1]
        if servs:
            idx_srv = servs[0]
            alocados[inst.servicos[j]] = inst.servidores[idx_srv]
            contagem_arq[inst.arch[idx_srv]] += 1
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
        "contagem_arq": contagem_arq,
    }


def validar_solucao_modelo_2(
    inst: InstanciaModelo2,
    pre: PreProcessamentoModelo2,
    alocacao: List[List[int]],
    modo: str,
    tol: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """Audita as restricoes (R1), (R2), (R3) e o dominio binario."""
    x = len(inst.servidores)
    y = len(inst.servicos)
    r = 4
    erros: List[str] = []

    for j in range(y):
        soma_j = sum(alocacao[i][j] for i in range(x))
        if modo == "obrigatorio" and soma_j != 1:
            erros.append(
                f"Violacao (R1) obrigatoria em {inst.servicos[j]}: soma={soma_j}"
            )
        if modo == "opcional" and soma_j not in (0, 1):
            erros.append(
                f"Violacao (R1) opcional em {inst.servicos[j]}: soma={soma_j}"
            )

    for i in range(x):
        for j in range(y):
            if alocacao[i][j] == 1 and pre.E[i][j] == 0:
                erros.append(
                    f"Violacao (R3) elegibilidade: {inst.servicos[j]} em {inst.servidores[i]}"
                )

    for i in range(x):
        for k in range(r):
            demanda_pico = sum(
                pre.Dp_ef[i][j][k] * alocacao[i][j]
                for j in range(y)
                if pre.E[i][j] == 1
            )
            if demanda_pico > pre.C[i][k] + tol:
                erros.append(
                    f"Violacao (R2) em ({inst.servidores[i]}, {inst.recursos[k]}): "
                    f"pico={demanda_pico:.4f} > C={pre.C[i][k]:.4f}"
                )

    return len(erros) == 0, erros


def resolver_modelo_2(
    inst: InstanciaModelo2,
    modo: str = "obrigatorio",
    executar_etapa_4: bool = True,
    eps: float = 1e-6,
    time_limit: Optional[float] = None,
    msg: bool = False,
) -> Dict[str, Any]:
    """Executa o pre-processamento (P1)-(P5) e resolve o Modelo 2 via PuLP + HiGHS."""
    if modo not in ("obrigatorio", "opcional"):
        raise ValueError("modo deve ser 'obrigatorio' ou 'opcional'.")

    t_pre0 = time.perf_counter()
    pre = pre_processar_modelo_2(inst)
    tempo_pre = time.perf_counter() - t_pre0

    x = len(inst.servidores)
    y = len(inst.servicos)
    r = 4
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

        # (R3) Elegibilidade (compatibilidade binaria, latencia, q[j] e m[i])
        for i in range(x):
            for j in range(y):
                prob += (a[i][j] <= pre.E[i][j], f"R3_eleg_i{i}_j{j}")

        # (R2) Capacidade multidimensional efetiva no pico (apenas pares com E[i][j]=1)
        for i in range(x):
            for k in range(r):
                prob += (
                    pulp.lpSum(
                        pre.Dp_ef[i][j][k] * a[i][j]
                        for j in range(y)
                        if pre.E[i][j] == 1
                    )
                    <= pre.C[i][k],
                    f"R2_cap_i{i}_k{k}",
                )

        # Expressoes lineares auxiliares u[i,k], ubar[k] e Incl
        u = [
            [
                pulp.lpSum(
                    pre.Dm_ef[i][j][k] * a[i][j]
                    for j in range(y)
                    if pre.E[i][j] == 1
                )
                / pre.C[i][k]
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
        prob1, a1, _, _, incl_expr1 = montar_base("Modelo2_Etapa1", pulp.LpMaximize)
        prob1 += incl_expr1, "Obj_Inclusao"
        t0 = time.perf_counter()
        prob1.solve(solver)
        dt = time.perf_counter() - t0
        if prob1.status != pulp.LpStatusOptimal:
            return {
                "viavel": False,
                "etapa_falha": 1,
                "status": pulp.LpStatus[prob1.status],
                "pre": pre,
                "tempo_pre_s": tempo_pre,
                "historico": historico,
            }
        a_atual = extrair_a(a1)
        m1 = calcular_metricas_modelo_2(inst, pre, a_atual)
        incl_star = m1["Incl"]
        historico.append({"etapa": "Etapa 1 (Inclusao)", "tempo_s": dt, **m1})

    # ------------------------------------------------------------------
    # ETAPA 2: QUALIDADE DE SERVICO (minimizar folga media S)
    # ------------------------------------------------------------------
    prob2, a2, u2, _, incl_expr2 = montar_base(
        "Modelo2_Etapa2", pulp.LpMinimize, a_inicial=a_atual
    )
    s2 = [
        [pulp.LpVariable(f"s_{i}_{k}", lowBound=0.0) for k in range(r)]
        for i in range(x)
    ]
    for i in range(x):
        for k in range(r):
            prob2 += (s2[i][k] >= u2[i][k] - inst.tau[k], f"R4_folga_i{i}_k{k}")
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
            "pre": pre,
            "tempo_pre_s": tempo_pre,
            "historico": historico,
        }
    a_atual = extrair_a(a2)
    m2 = calcular_metricas_modelo_2(inst, pre, a_atual)
    s_star = m2["S"]
    historico.append({"etapa": "Etapa 2 (QoS / S)", "tempo_s": dt, **m2})

    # ------------------------------------------------------------------
    # ETAPA 3: GARGALO (minimizar U_max)
    # ------------------------------------------------------------------
    prob3, a3, u3, _, incl_expr3 = montar_base(
        "Modelo2_Etapa3", pulp.LpMinimize, a_inicial=a_atual
    )
    s3 = [
        [pulp.LpVariable(f"s_{i}_{k}", lowBound=0.0) for k in range(r)]
        for i in range(x)
    ]
    U_max3 = pulp.LpVariable("U_max", lowBound=0.0, upBound=1.0)
    for i in range(x):
        for k in range(r):
            prob3 += (s3[i][k] >= u3[i][k] - inst.tau[k], f"R4_folga_i{i}_k{k}")
            prob3 += (U_max3 >= u3[i][k], f"R5_gargalo_i{i}_k{k}")
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
            "pre": pre,
            "tempo_pre_s": tempo_pre,
            "historico": historico,
        }
    a_atual = extrair_a(a3)
    m3 = calcular_metricas_modelo_2(inst, pre, a_atual)
    umax_star = m3["U_max"]
    historico.append({"etapa": "Etapa 3 (Gargalo)", "tempo_s": dt, **m3})

    # ------------------------------------------------------------------
    # ETAPA 4: EQUILIBRIO ENTRE SERVIDORES (opcional, desvio absoluto L1)
    # ------------------------------------------------------------------
    if executar_etapa_4:
        prob4, a4, u4, ubar4, incl_expr4 = montar_base(
            "Modelo2_Etapa4", pulp.LpMinimize, a_inicial=a_atual
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
                prob4 += (s4[i][k] >= u4[i][k] - inst.tau[k], f"R4_folga_i{i}_k{k}")
                prob4 += (U_max4 >= u4[i][k], f"R5_gargalo_i{i}_k{k}")
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
                "pre": pre,
                "tempo_pre_s": tempo_pre,
                "historico": historico,
            }
        a_atual = extrair_a(a4)
        m4 = calcular_metricas_modelo_2(inst, pre, a_atual)
        historico.append({"etapa": "Etapa 4 (Equilibrio)", "tempo_s": dt, **m4})

    assert a_atual is not None
    ok, erros = validar_solucao_modelo_2(inst, pre, a_atual, modo=modo)
    tempo_solver = sum(h["tempo_s"] for h in historico)

    return {
        "viavel": True,
        "valido": ok,
        "erros_validacao": erros,
        "pre": pre,
        "alocacao": a_atual,
        "metricas": historico[-1],
        "historico": historico,
        "tempo_pre_s": tempo_pre,
        "tempo_total_s": tempo_pre + tempo_solver,
    }


def imprimir_relatorio_modelo_2(
    titulo: str,
    inst: InstanciaModelo2,
    res: Dict[str, Any],
    mostrar_alocacao: bool = True,
) -> None:
    print("\n" + "=" * 86)
    print(titulo)
    print("=" * 86)
    if not res["viavel"]:
        print(
            f"Resultado: INVIAVEL na Etapa {res['etapa_falha']} "
            f"(status HiGHS: {res['status']})."
        )
        pre = res["pre"]
        sem_eleg = [
            inst.servicos[j]
            for j in range(len(inst.servicos))
            if sum(pre.E[i][j] for i in range(len(inst.servidores))) == 0
        ]
        if sem_eleg:
            print(f"Servicos sem nenhum servidor elegivel: {', '.join(sem_eleg)}")
        return

    m = res["metricas"]
    print(
        f"Status: OTIMO | Auditoria (R1-R3): {'OK' if res['valido'] else 'FALHOU'} | "
        f"Pre-proc: {res['tempo_pre_s']*1000:.2f} ms | "
        f"Tempo total: {res['tempo_total_s']:.4f} s"
    )
    print(
        f"Distribuicao por arquitetura -> X86: {m['contagem_arq']['X86']} servicos | "
        f"ARM: {m['contagem_arq']['ARM']} servicos"
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

    print("\nUtilizacao media (u) e pico (p) por servidor:")
    for i, srv in enumerate(inst.servidores):
        u_fmt = ", ".join(
            f"{inst.recursos[k]}={m['u'][i][k]*100:4.1f}% (pico {m['uso_pico'][i][k]*100:4.1f}%)"
            for k in range(4)
        )
        print(f"  {srv} [{inst.arch[i]}, m={int(inst.m[i])}, v={inst.v[i]:.2f}]: {u_fmt}")

    if mostrar_alocacao:
        print("\nAlocacao final dos servicos:")
        for serv, srv in m["alocados"].items():
            idx_i = inst.servidores.index(srv)
            idx_j = inst.servicos.index(serv)
            dp_cpu = res["pre"].Dp_ef[idx_i][idx_j][0]
            dm_cpu = res["pre"].Dm_ef[idx_i][idx_j][0]
            c_lat = res["pre"].c_lat[idx_i][idx_j]
            print(
                f"  {serv:<18} -> {srv:<16} ({inst.arch[idx_i]}) | "
                f"c_lat={c_lat:4.2f}, Dp_ef_cpu={dp_cpu:5.2f}, Dm_ef_cpu={dm_cpu:5.2f}"
            )

    if m["nao_alocados"]:
        print(f"\nServicos nao alocados: {', '.join(m['nao_alocados'])}")


# ==========================================================================
# SIMULACOES DE VALIDACAO E SENSIBILIDADE DO MODELO 2
# ==========================================================================

def cenario_1_especializacao_arm_x86() -> InstanciaModelo2:
    """
    Inspirado na Secao 10 do Modelo 2: servico sequencial rigido (phi=0.30, L=50)
    fica inelegivel no ARM e vai ao x86, enquanto servicos paralelizaveis (phi=0.95)
    aproveitam os 80 nucleos ARM ocupando menor fracao de CPU.
    """
    return InstanciaModelo2(
        servidores=["S1_x86", "S2_ARM"],
        arch=["X86", "ARM"],
        m=[32.0, 80.0],
        v=[1.0, 0.5],
        C_outros=[
            [20.0, 128.0, 1000.0],
            [20.0, 128.0, 1000.0],
        ],
        servicos=["J1_Web1", "J2_Web2", "J3_SeqRigido"],
        p=[1.0, 1.0, 1.0],
        Wp=[10.0, 10.0, 16.0],
        Wm=[8.0, 8.0, 12.0],
        phi=[0.95, 0.95, 0.30],
        T1=[40.0, 40.0, 40.0],
        L=[100.0, 100.0, 50.0],
        q=[64.0, 64.0, 64.0],
        e=[
            {"ARM": 1.0, "X86": 1.0},
            {"ARM": 1.0, "X86": 1.0},
            {"ARM": 1.0, "X86": 1.0},
        ],
        A=[
            {"ARM": 1, "X86": 1},
            {"ARM": 1, "X86": 1},
            {"ARM": 1, "X86": 1},
        ],
        Dp_outros=[
            [4.0, 32.0, 200.0],
            [4.0, 32.0, 200.0],
            [6.0, 40.0, 250.0],
        ],
        Dm_outros=[
            [2.0, 16.0, 100.0],
            [2.0, 16.0, 100.0],
            [4.0, 32.0, 200.0],
        ],
        tau=[0.75, 0.75, 0.75, 0.75],
    )


def cenario_2_regras_preprocessamento() -> InstanciaModelo2:
    """
    Valida simultaneamente:
      - J1_LegadoX86: A[ARM]=0 (incompatibilidade binaria)
      - J2_LimiteQ: c_lat=6 > q=4 no ARM (limite de paralelismo do software)
      - J3_LatenciaDomina: no ARM, c_lat=6 domina Wp/(v*e)=4 no pico, mas media usa vazao
      - J4_EscalaBaixaARM: fator e[ARM]=0.5 reduz vazao efetiva no ARM (contencao)
    """
    return InstanciaModelo2(
        servidores=["S1_x86", "S2_ARM"],
        arch=["X86", "ARM"],
        m=[16.0, 40.0],
        v=[1.0, 0.5],
        C_outros=[
            [20.0, 64.0, 500.0],
            [20.0, 64.0, 500.0],
        ],
        servicos=["J1_LegadoX86", "J2_LimiteQ", "J3_LatenciaDomina", "J4_EscalaBaixaARM"],
        p=[1.0, 1.0, 1.0, 1.0],
        Wp=[6.0, 4.0, 2.0, 8.0],
        Wm=[4.0, 3.0, 2.0, 6.0],
        phi=[0.80, 0.30, 0.30, 0.90],
        T1=[20.0, 40.0, 40.0, 20.0],
        L=[50.0, 60.0, 60.0, 60.0],
        q=[16.0, 4.0, 8.0, 32.0],
        e=[
            {"ARM": 1.0, "X86": 1.0},
            {"ARM": 1.0, "X86": 1.0},
            {"ARM": 1.0, "X86": 1.0},
            {"ARM": 0.5, "X86": 1.0},
        ],
        A=[
            {"ARM": 0, "X86": 1},
            {"ARM": 1, "X86": 1},
            {"ARM": 1, "X86": 1},
            {"ARM": 1, "X86": 1},
        ],
        Dp_outros=[
            [4.0, 16.0, 100.0],
            [2.0, 8.0, 50.0],
            [2.0, 8.0, 50.0],
            [4.0, 16.0, 100.0],
        ],
        Dm_outros=[
            [2.0, 8.0, 50.0],
            [1.0, 4.0, 25.0],
            [1.0, 4.0, 25.0],
            [2.0, 8.0, 50.0],
        ],
        tau=[0.75, 0.75, 0.75, 0.75],
    )


def cenario_3_diagnostico_inelegibilidade() -> InstanciaModelo2:
    """
    Cluster exclusivamente ARM com um servico de alto peso (p=10) cuja latencia
    e impossivel em v=0.5: obrigatorio e inviavel; opcional identifica o bloqueio.
    """
    return InstanciaModelo2(
        servidores=["S1_ARM", "S2_ARM"],
        arch=["ARM", "ARM"],
        m=[40.0, 40.0],
        v=[0.5, 0.5],
        C_outros=[
            [20.0, 64.0, 500.0],
            [20.0, 64.0, 500.0],
        ],
        servicos=["J1_WebAlto", "J2_WebMedio", "J3_SemElegivel"],
        p=[5.0, 3.0, 10.0],
        Wp=[10.0, 10.0, 2.0],
        Wm=[8.0, 8.0, 1.0],
        phi=[0.95, 0.95, 0.20],
        T1=[40.0, 40.0, 40.0],
        L=[100.0, 100.0, 50.0],
        q=[64.0, 64.0, 64.0],
        e=[
            {"ARM": 1.0, "X86": 1.0},
            {"ARM": 1.0, "X86": 1.0},
            {"ARM": 1.0, "X86": 1.0},
        ],
        A=[
            {"ARM": 1, "X86": 1},
            {"ARM": 1, "X86": 1},
            {"ARM": 1, "X86": 1},
        ],
        Dp_outros=[
            [4.0, 16.0, 100.0],
            [4.0, 16.0, 100.0],
            [1.0, 4.0, 20.0],
        ],
        Dm_outros=[
            [2.0, 8.0, 50.0],
            [2.0, 8.0, 50.0],
            [1.0, 4.0, 20.0],
        ],
        tau=[0.75, 0.75, 0.75, 0.75],
    )


def cenario_4_cluster_misto_medio(
    seed: int = 42,
    v_arm: float = 0.51,
) -> InstanciaModelo2:
    """
    Cluster misto de 6 servidores (Ice Lake-SP, Zen 3 e Ampere Altra, inspirados
    em Ansari et al., 2024) e 24 servicos com tres perfis de paralelismo.
    """
    rng = random.Random(seed)
    servidores = [
        "S1_IceLake_x86",
        "S2_IceLake_x86",
        "S3_Zen3_x86",
        "S4_Altra_ARM",
        "S5_Altra_ARM",
        "S6_Altra_ARM",
    ]
    arch = ["X86", "X86", "X86", "ARM", "ARM", "ARM"]
    m = [32.0, 32.0, 64.0, 80.0, 80.0, 80.0]
    v = [1.00, 1.00, 0.87, v_arm, v_arm, v_arm]
    C_outros = [
        [25.0, 128.0, 1000.0],
        [25.0, 128.0, 1000.0],
        [25.0, 256.0, 1500.0],
        [25.0, 256.0, 1500.0],
        [25.0, 256.0, 1500.0],
        [25.0, 256.0, 1500.0],
    ]

    servicos: List[str] = []
    p: List[float] = []
    Wp: List[float] = []
    Wm: List[float] = []
    phi: List[float] = []
    T1: List[float] = []
    L: List[float] = []
    q: List[float] = []
    e: List[Dict[str, float]] = []
    A: List[Dict[str, int]] = []
    Dp_outros: List[List[float]] = []
    Dm_outros: List[List[float]] = []

    for j in range(24):
        if j < 8:
            # Perfil 1: Web/Scale-out altamente paralelizavel
            servicos.append(f"WebPar_{j+1:02d}")
            wm_j = round(rng.uniform(3.0, 6.5), 2)
            wp_j = round(wm_j * rng.uniform(1.20, 1.40), 2)
            phi_j = round(rng.uniform(0.85, 0.96), 2)
            t1_j = round(rng.uniform(30.0, 50.0), 1)
            l_j = round(rng.uniform(80.0, 120.0), 1)
            q_j = 64.0
            e_j = {
                "ARM": round(rng.uniform(0.85, 1.00), 2),
                "X86": round(rng.uniform(0.95, 1.10), 2),
            }
            a_j = {"ARM": 1, "X86": 1}
        elif j < 16:
            # Perfil 2: APIs mistas (sensibilidade a v_arm e escalabilidade)
            servicos.append(f"ApiMista_{j-7:02d}")
            wm_j = round(rng.uniform(2.5, 5.0), 2)
            wp_j = round(wm_j * rng.uniform(1.20, 1.40), 2)
            phi_j = round(rng.uniform(0.45, 0.72), 2)
            t1_j = round(rng.uniform(30.0, 42.0), 1)
            l_j = round(rng.uniform(52.0, 72.0), 1)
            q_j = 32.0
            e_j = {
                "ARM": round(rng.uniform(0.75, 0.95), 2),
                "X86": round(rng.uniform(0.95, 1.05), 2),
            }
            a_j = {"ARM": 1, "X86": 1}
        else:
            # Perfil 3: Baixo paralelismo e latencia apertada (2 sem binario ARM)
            servicos.append(f"SeqLat_{j-15:02d}")
            wm_j = round(rng.uniform(1.5, 3.8), 2)
            wp_j = round(wm_j * rng.uniform(1.15, 1.35), 2)
            phi_j = round(rng.uniform(0.15, 0.35), 2)
            t1_j = round(rng.uniform(35.0, 44.0), 1)
            l_j = round(rng.uniform(48.0, 60.0), 1)
            q_j = 16.0
            e_j = {
                "ARM": round(rng.uniform(0.70, 0.90), 2),
                "X86": round(rng.uniform(0.95, 1.05), 2),
            }
            a_j = {"ARM": 0 if j >= 22 else 1, "X86": 1}

        rede_m = round(rng.uniform(1.0, 3.2), 2)
        ram_m = round(rng.uniform(6.0, 22.0), 2)
        arm_m = round(rng.uniform(40.0, 150.0), 2)

        rede_p = round(rede_m * rng.uniform(1.20, 1.40), 2)
        ram_p = round(ram_m * rng.uniform(1.15, 1.35), 2)
        arm_p = round(arm_m * rng.uniform(1.10, 1.25), 2)

        p.append(float(rng.choice([1, 2, 3, 5])))
        Wp.append(wp_j)
        Wm.append(wm_j)
        phi.append(phi_j)
        T1.append(t1_j)
        L.append(l_j)
        q.append(q_j)
        e.append(e_j)
        A.append(a_j)
        Dm_outros.append([rede_m, ram_m, arm_m])
        Dp_outros.append([rede_p, ram_p, arm_p])

    return InstanciaModelo2(
        servidores=servidores,
        arch=arch,
        m=m,
        v=v,
        C_outros=C_outros,
        servicos=servicos,
        p=p,
        Wp=Wp,
        Wm=Wm,
        phi=phi,
        T1=T1,
        L=L,
        q=q,
        e=e,
        A=A,
        Dp_outros=Dp_outros,
        Dm_outros=Dm_outros,
        tau=[0.75, 0.75, 0.75, 0.75],
    )


if __name__ == "__main__":
    # Simulacao 1: Especializacao natural ARM vs x86
    inst1 = cenario_1_especializacao_arm_x86()
    res1 = resolver_modelo_2(inst1, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio_modelo_2(
        "SIMULACAO 1 (Modelo 2): Especializacao ARM vs x86 por paralelismo e latencia",
        inst1,
        res1,
    )
    assert res1["viavel"] and res1["valido"]
    assert res1["pre"].E[1][2] == 0  # J3_SeqRigido inelegivel no ARM
    assert res1["metricas"]["alocados"]["J3_SeqRigido"] == "S1_x86"
    assert res1["metricas"]["alocados"]["J1_Web1"] == "S2_ARM"
    assert res1["metricas"]["alocados"]["J2_Web2"] == "S2_ARM"
    assert abs(res1["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res1["metricas"]["U_max"] - 0.40) < 1e-6
    assert abs(res1["metricas"]["D"] - 0.003125) < 1e-6

    # Simulacao 2: Compatibilidade A, limite q[j], c_lat dominando pico e e[j,t]
    inst2 = cenario_2_regras_preprocessamento()
    res2 = resolver_modelo_2(inst2, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio_modelo_2(
        "SIMULACAO 2 (Modelo 2): Validacao de A[j,t], q[j], c_lat e escalabilidade e[j,t]",
        inst2,
        res2,
    )
    assert res2["viavel"] and res2["valido"]
    assert res2["pre"].E[1][0] == 0  # J1_LegadoX86 sem binario ARM
    assert res2["pre"].E[1][1] == 0  # J2_LimiteQ barrado por q=4 < c_lat=6
    assert res2["pre"].E[1][2] == 1  # J3_LatenciaDomina elegivel no ARM
    assert abs(res2["pre"].Dp_ef[1][2][0] - 6.0) < 1e-6  # c_lat=6 domina Wp/v=4 no pico
    assert abs(res2["pre"].Dm_ef[1][2][0] - 4.0) < 1e-6  # media usa somente vazao
    assert abs(res2["pre"].Dp_ef[1][3][0] - 32.0) < 1e-6  # e[ARM]=0.5 dobra nucleos
    assert res2["metricas"]["alocados"]["J3_LatenciaDomina"] == "S1_x86"
    assert res2["metricas"]["alocados"]["J4_EscalaBaixaARM"] == "S2_ARM"
    assert abs(res2["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res2["metricas"]["U_max"] - 0.60) < 1e-6
    assert abs(res2["metricas"]["D"] - 0.0453125) < 1e-6

    # Simulacao 3A e 3B: Inelegibilidade total no obrigatorio -> diagnostico opcional
    inst3 = cenario_3_diagnostico_inelegibilidade()
    res3a = resolver_modelo_2(inst3, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio_modelo_2(
        "SIMULACAO 3A (Modelo 2): Cluster so ARM no modo obrigatorio",
        inst3,
        res3a,
    )
    assert not res3a["viavel"]

    res3b = resolver_modelo_2(inst3, modo="opcional", executar_etapa_4=True)
    imprimir_relatorio_modelo_2(
        "SIMULACAO 3B (Modelo 2): Diagnostico de inelegibilidade no modo opcional",
        inst3,
        res3b,
    )
    assert res3b["viavel"] and res3b["valido"]
    assert abs(res3b["metricas"]["Incl"] - (8.0 / 18.0)) < 1e-6
    assert abs(res3b["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res3b["metricas"]["U_max"] - 0.40) < 1e-6
    assert abs(res3b["metricas"]["D"] - 0.0) < 1e-6
    assert res3b["metricas"]["nao_alocados"] == ["J3_SemElegivel"]

    # Simulacao 4: Cluster misto de medio porte (6 servidores x 24 servicos = 144 binarias)
    inst4 = cenario_4_cluster_misto_medio(seed=42, v_arm=0.51)
    res4 = resolver_modelo_2(
        inst4, modo="obrigatorio", executar_etapa_4=True, time_limit=30.0
    )
    imprimir_relatorio_modelo_2(
        "SIMULACAO 4 (Modelo 2): Cluster misto 6x24 (Ice Lake / Zen 3 / Ampere Altra)",
        inst4,
        res4,
        mostrar_alocacao=False,
    )

    # Simulacao 5: Analise de sensibilidade em v_ARM (Secao 12 do Modelo 2)
    print("\n" + "=" * 86)
    print("SIMULACAO 5 (Modelo 2): Analise de sensibilidade no cluster 6x24 variando v_ARM")
    print("=" * 86)
    print(
        f"  {'v_ARM':>6} {'Eleg_ARM':>9} {'Aloc_X86':>9} {'Aloc_ARM':>9} "
        f"{'S (QoS)':>10} {'U_max':>8} {'Desvio D':>10} {'Tempo(s)':>9}"
    )
    for v_teste in (0.45, 0.51, 0.65, 0.80):
        inst_s = cenario_4_cluster_misto_medio(seed=42, v_arm=v_teste)
        res_s = resolver_modelo_2(
            inst_s, modo="obrigatorio", executar_etapa_4=True, time_limit=30.0
        )
        eleg_arm = sum(
            1 for j in range(len(inst_s.servicos)) if res_s["pre"].E[3][j] == 1
        )
        if res_s["viavel"]:
            ms = res_s["metricas"]
            print(
                f"  {v_teste:>6.2f} {eleg_arm:>9d} {ms['contagem_arq']['X86']:>9d} "
                f"{ms['contagem_arq']['ARM']:>9d} {ms['S']:>10.6f} "
                f"{ms['U_max']:>8.4f} {ms['D']:>10.6f} {res_s['tempo_total_s']:>9.4f}"
            )
        else:
            print(f"  {v_teste:>6.2f} {eleg_arm:>9d} {'INVIAVEL':>30}")

    print("\nTodas as verificacoes analiticas do Modelo 2 passaram.")