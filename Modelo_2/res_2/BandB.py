from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


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
    """Coeficientes constantes derivados antes da otimizacao (P1 a P5)."""

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


def montar_lp_etapa_modelo_2(
    inst: InstanciaModelo2,
    pre: PreProcessamentoModelo2,
    etapa: int,
    modo: str,
    eps: float = 1e-6,
    incl_star: Optional[float] = None,
    s_star: Optional[float] = None,
    umax_star: Optional[float] = None,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    List[Tuple[float, Optional[float]]],
]:
    """
    Constroi a formulacao matricial da relaxacao linear (min c^T z) do Modelo 2
    para a etapa lexicografica especificada (1, 2, 3 ou 4).
    A restricao (R3) de elegibilidade e aplicada diretamente nas fixacoes iniciais.
    """
    x = len(inst.servidores)
    y = len(inst.servicos)
    r = 4
    n_a = x * y
    n_s = x * r
    n_d = x * r
    peso_total = sum(inst.p)

    if etapa == 1:
        n_vars = n_a
    elif etapa == 2:
        n_vars = n_a + n_s
    elif etapa == 3:
        n_vars = n_a + n_s + 1
    elif etapa == 4:
        n_vars = n_a + n_s + 1 + n_d
    else:
        raise ValueError("Etapa deve ser 1, 2, 3 ou 4.")

    idx_umax = n_a + n_s
    offset_d = n_a + n_s + 1

    c = np.zeros(n_vars, dtype=float)
    if etapa == 1:
        for i in range(x):
            for j in range(y):
                c[i * y + j] = -inst.p[j] / peso_total
    elif etapa == 2:
        for i in range(x):
            for k in range(r):
                c[n_a + i * r + k] = 1.0 / (x * r)
    elif etapa == 3:
        c[idx_umax] = 1.0
    elif etapa == 4:
        for i in range(x):
            for k in range(r):
                c[offset_d + i * r + k] = 1.0 / (x * r)

    A_ub_rows: List[np.ndarray] = []
    b_ub_vals: List[float] = []
    A_eq_rows: List[np.ndarray] = []
    b_eq_vals: List[float] = []

    # (R1) Admissao
    for j in range(y):
        row = np.zeros(n_vars, dtype=float)
        for i in range(x):
            row[i * y + j] = 1.0
        if modo == "obrigatorio":
            A_eq_rows.append(row)
            b_eq_vals.append(1.0)
        else:
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)

    # (R2) Capacidade multidimensional efetiva no pico (somente pares elegiveis)
    for i in range(x):
        for k in range(r):
            row = np.zeros(n_vars, dtype=float)
            for j in range(y):
                if pre.E[i][j] == 1:
                    row[i * y + j] = pre.Dp_ef[i][j][k]
            A_ub_rows.append(row)
            b_ub_vals.append(pre.C[i][k])

    # Fixacao da Etapa 1 (modo opcional)
    if etapa >= 2 and modo == "opcional":
        assert incl_star is not None
        row = np.zeros(n_vars, dtype=float)
        for i in range(x):
            for j in range(y):
                row[i * y + j] = -inst.p[j] / peso_total
        A_ub_rows.append(row)
        b_ub_vals.append(-(incl_star - eps))

    # (R4) Excesso acima da zona segura: u[i,k] - s[i,k] <= tau[k]
    if etapa >= 2:
        for i in range(x):
            for k in range(r):
                row = np.zeros(n_vars, dtype=float)
                for j in range(y):
                    if pre.E[i][j] == 1:
                        row[i * y + j] = pre.Dm_ef[i][j][k] / pre.C[i][k]
                row[n_a + i * r + k] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(inst.tau[k])

    # Fixacao da Etapa 2 e (R5) Gargalo Min-Max
    if etapa >= 3:
        assert s_star is not None
        row_s = np.zeros(n_vars, dtype=float)
        for i in range(x):
            for k in range(r):
                row_s[n_a + i * r + k] = 1.0 / (x * r)
        A_ub_rows.append(row_s)
        b_ub_vals.append(s_star + eps)

        for i in range(x):
            for k in range(r):
                row = np.zeros(n_vars, dtype=float)
                for j in range(y):
                    if pre.E[i][j] == 1:
                        row[i * y + j] = pre.Dm_ef[i][j][k] / pre.C[i][k]
                row[idx_umax] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(0.0)

    # Fixacao da Etapa 3 e linearizacao do desvio absoluto L1
    if etapa == 4:
        assert umax_star is not None
        row_u = np.zeros(n_vars, dtype=float)
        row_u[idx_umax] = 1.0
        A_ub_rows.append(row_u)
        b_ub_vals.append(umax_star + eps)

        for i in range(x):
            for k in range(r):
                row_pos = np.zeros(n_vars, dtype=float)
                row_neg = np.zeros(n_vars, dtype=float)
                for i2 in range(x):
                    fator = (1.0 - 1.0 / x) if i2 == i else (-1.0 / x)
                    for j in range(y):
                        if pre.E[i2][j] == 1:
                            coef = fator * (pre.Dm_ef[i2][j][k] / pre.C[i2][k])
                            row_pos[i2 * y + j] = coef
                            row_neg[i2 * y + j] = -coef
                idx_d = offset_d + i * r + k
                row_pos[idx_d] = -1.0
                row_neg[idx_d] = -1.0
                A_ub_rows.append(row_pos)
                b_ub_vals.append(0.0)
                A_ub_rows.append(row_neg)
                b_ub_vals.append(0.0)

    bounds_cont: List[Tuple[float, Optional[float]]] = []
    if etapa >= 2:
        bounds_cont.extend([(0.0, None)] * n_s)
    if etapa >= 3:
        bounds_cont.append((0.0, 1.0))
    if etapa == 4:
        bounds_cont.extend([(0.0, None)] * n_d)

    A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals, dtype=float) if b_ub_vals else None
    A_eq = np.vstack(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_vals, dtype=float) if b_eq_vals else None

    return c, A_ub, b_ub, A_eq, b_eq, bounds_cont


def _escolher_variavel_fracionaria(
    x_vec: np.ndarray,
    n_bin: int,
    tol_int: float,
) -> Optional[int]:
    """Seleciona a variavel binaria mais proxima de 0.5."""
    melhor_idx: Optional[int] = None
    melhor_dist = float("inf")
    for idx in range(n_bin):
        val = float(x_vec[idx])
        if abs(val - round(val)) > tol_int:
            dist = abs(val - 0.5)
            if dist < melhor_dist:
                melhor_dist = dist
                melhor_idx = idx
    return melhor_idx


def _vetor_para_matriz(x_vec: np.ndarray, x: int, y: int) -> List[List[int]]:
    return [
        [int(round(float(x_vec[i * y + j]))) for j in range(y)]
        for i in range(x)
    ]


def _propagar_fixacao(
    fixacoes: Tuple[int, ...],
    idx_alvo: int,
    valor: int,
    x: int,
    y: int,
) -> Optional[Tuple[int, ...]]:
    """Aplica ramificacao em a[i,j] e propaga exclusividade de (R1) quando a[i,j]=1."""
    if valor == 0:
        if fixacoes[idx_alvo] == 1:
            return None
        novo = list(fixacoes)
        novo[idx_alvo] = 0
        return tuple(novo)

    if fixacoes[idx_alvo] == 0:
        return None
    novo = list(fixacoes)
    i_alvo = idx_alvo // y
    j_alvo = idx_alvo % y
    for i2 in range(x):
        idx2 = i2 * y + j_alvo
        if i2 == i_alvo:
            novo[idx2] = 1
        else:
            if novo[idx2] == 1:
                return None
            novo[idx2] = 0
    return tuple(novo)


def branch_and_bound_lp(
    c: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    bounds_cont: List[Tuple[float, Optional[float]]],
    raiz_fix: Tuple[int, ...],
    x: int,
    y: int,
    incumbente_inicial: Optional[Tuple[float, List[List[int]]]] = None,
    time_limit: float = 30.0,
    max_nos: int = 50000,
    tol_int: float = 1e-6,
    tol_bound: float = 1e-7,
) -> Dict[str, Any]:
    """
    Branch-and-Bound classico (busca Best-First) com relaxacao linear
    resolvida pelo metodo Simplex Dual (method='highs-ds').
    """
    t0 = time.perf_counter()
    n_bin = x * y

    def construir_bounds(
        fix: Tuple[int, ...],
    ) -> List[Tuple[float, Optional[float]]]:
        b_bin: List[Tuple[float, Optional[float]]] = []
        for f in fix:
            if f == 0:
                b_bin.append((0.0, 0.0))
            elif f == 1:
                b_bin.append((1.0, 1.0))
            else:
                b_bin.append((0.0, 1.0))
        return b_bin + bounds_cont

    best_val = float("inf")
    best_a: Optional[List[List[int]]] = None
    if incumbente_inicial is not None:
        best_val, best_a = incumbente_inicial

    nos_resolvidos = 0
    podas_inviabilidade = 0
    podas_limitante = 0
    podas_integralidade = 0

    res_raiz = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=construir_bounds(raiz_fix),
        method="highs-ds",
    )
    nos_resolvidos += 1

    if not res_raiz.success:
        return {
            "viavel": False,
            "otimo_provado": True,
            "status": "INVIAVEL",
            "best_val": None,
            "best_a": None,
            "dual_lb": None,
            "gap_rel": None,
            "nos_resolvidos": nos_resolvidos,
            "podas_inviabilidade": 1,
            "podas_limitante": 0,
            "podas_integralidade": 0,
            "tempo_s": time.perf_counter() - t0,
        }

    root_lb = float(res_raiz.fun)
    if root_lb >= best_val - tol_bound:
        podas_limitante += 1
        return {
            "viavel": True,
            "otimo_provado": True,
            "status": "OTIMO",
            "best_val": best_val,
            "best_a": best_a,
            "dual_lb": best_val,
            "gap_rel": 0.0,
            "nos_resolvidos": nos_resolvidos,
            "podas_inviabilidade": podas_inviabilidade,
            "podas_limitante": podas_limitante,
            "podas_integralidade": podas_integralidade,
            "tempo_s": time.perf_counter() - t0,
        }

    idx_frac_raiz = _escolher_variavel_fracionaria(res_raiz.x, n_bin, tol_int)
    if idx_frac_raiz is None:
        podas_integralidade += 1
        if root_lb < best_val - 1e-9:
            best_val = root_lb
            best_a = _vetor_para_matriz(res_raiz.x, x, y)
        return {
            "viavel": True,
            "otimo_provado": True,
            "status": "OTIMO",
            "best_val": best_val,
            "best_a": best_a,
            "dual_lb": best_val,
            "gap_rel": 0.0,
            "nos_resolvidos": nos_resolvidos,
            "podas_inviabilidade": podas_inviabilidade,
            "podas_limitante": podas_limitante,
            "podas_integralidade": podas_integralidade,
            "tempo_s": time.perf_counter() - t0,
        }

    heap: List[Tuple[float, int, Tuple[int, ...], int]] = []
    contador_id = 0
    heapq.heappush(heap, (root_lb, contador_id, raiz_fix, idx_frac_raiz))

    status = "OTIMO"
    while heap:
        if heap[0][0] >= best_val - tol_bound:
            podas_limitante += len(heap)
            heap.clear()
            break
        if (time.perf_counter() - t0) >= time_limit:
            status = "LIMITE_TEMPO"
            break
        if nos_resolvidos >= max_nos:
            status = "LIMITE_NOS"
            break

        _, _, node_fix, idx_branch = heapq.heappop(heap)

        for ramo_val in (1, 0):
            child_fix = _propagar_fixacao(node_fix, idx_branch, ramo_val, x, y)
            if child_fix is None:
                podas_inviabilidade += 1
                continue

            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=construir_bounds(child_fix),
                method="highs-ds",
            )
            nos_resolvidos += 1

            if not res.success:
                podas_inviabilidade += 1
                continue

            child_lb = float(res.fun)
            if child_lb >= best_val - tol_bound:
                podas_limitante += 1
                continue

            idx_frac = _escolher_variavel_fracionaria(res.x, n_bin, tol_int)
            if idx_frac is None:
                podas_integralidade += 1
                if child_lb < best_val - 1e-9:
                    best_val = child_lb
                    best_a = _vetor_para_matriz(res.x, x, y)
            else:
                contador_id += 1
                heapq.heappush(heap, (child_lb, contador_id, child_fix, idx_frac))

    tempo_s = time.perf_counter() - t0
    if best_a is None:
        return {
            "viavel": False,
            "otimo_provado": (status == "OTIMO"),
            "status": "INVIAVEL" if status == "OTIMO" else status,
            "best_val": None,
            "best_a": None,
            "dual_lb": heap[0][0] if heap else None,
            "gap_rel": None,
            "nos_resolvidos": nos_resolvidos,
            "podas_inviabilidade": podas_inviabilidade,
            "podas_limitante": podas_limitante,
            "podas_integralidade": podas_integralidade,
            "tempo_s": tempo_s,
        }

    otimo_provado = (status == "OTIMO") or (not heap)
    dual_lb = best_val if otimo_provado else heap[0][0]
    if otimo_provado or (abs(best_val) < 1e-9 and abs(dual_lb) < 1e-9):
        gap_rel = 0.0
    else:
        gap_rel = max(0.0, (best_val - dual_lb) / max(abs(best_val), 1e-9))

    return {
        "viavel": True,
        "otimo_provado": otimo_provado,
        "status": "OTIMO" if otimo_provado else status,
        "best_val": best_val,
        "best_a": best_a,
        "dual_lb": dual_lb,
        "gap_rel": gap_rel,
        "nos_resolvidos": nos_resolvidos,
        "podas_inviabilidade": podas_inviabilidade,
        "podas_limitante": podas_limitante,
        "podas_integralidade": podas_integralidade,
        "tempo_s": tempo_s,
    }


def resolver_modelo_2_bnb(
    inst: InstanciaModelo2,
    modo: str = "obrigatorio",
    executar_etapa_4: bool = True,
    eps: float = 1e-6,
    time_limit_etapa: float = 30.0,
    max_nos_etapa: int = 50000,
) -> Dict[str, Any]:
    """Executa (P1)-(P5) e resolve o Modelo 2 com Branch-and-Bound classico."""
    if modo not in ("obrigatorio", "opcional"):
        raise ValueError("modo deve ser 'obrigatorio' ou 'opcional'.")

    t_pre0 = time.perf_counter()
    pre = pre_processar_modelo_2(inst)
    tempo_pre = time.perf_counter() - t_pre0

    x = len(inst.servidores)
    y = len(inst.servicos)

    # Fixacao inicial da restricao (R3): pares inelegiveis ja nascem fixados em 0
    raiz_fix = tuple(
        0 if pre.E[i][j] == 0 else -1
        for i in range(x)
        for j in range(y)
    )
    vars_ativas = sum(1 for f in raiz_fix if f == -1)

    historico: List[Dict[str, Any]] = []
    a_atual: Optional[List[List[int]]] = None
    incl_star = 1.0
    s_star = 0.0
    umax_star = 1.0

    # ------------------------------------------------------------------
    # ETAPA 1: INCLUSAO (apenas no modo opcional)
    # ------------------------------------------------------------------
    if modo == "opcional":
        c1, A_ub1, b_ub1, A_eq1, b_eq1, b_cont1 = montar_lp_etapa_modelo_2(
            inst, pre, etapa=1, modo=modo, eps=eps
        )
        r1 = branch_and_bound_lp(
            c1, A_ub1, b_ub1, A_eq1, b_eq1, b_cont1, raiz_fix, x, y,
            incumbente_inicial=None,
            time_limit=time_limit_etapa,
            max_nos=max_nos_etapa,
        )
        if not r1["viavel"]:
            return {
                "viavel": False,
                "etapa_falha": 1,
                "status": r1["status"],
                "pre": pre,
                "vars_ativas": vars_ativas,
                "tempo_pre_s": tempo_pre,
                "historico": historico,
            }
        a_atual = r1["best_a"]
        m1 = calcular_metricas_modelo_2(inst, pre, a_atual)
        incl_star = m1["Incl"]
        historico.append({"etapa": "Etapa 1 (Inclusao)", **r1, **m1})

    # ------------------------------------------------------------------
    # ETAPA 2: QUALIDADE DE SERVICO (minimizar folga media S)
    # ------------------------------------------------------------------
    inc2 = None
    if a_atual is not None:
        inc2 = (calcular_metricas_modelo_2(inst, pre, a_atual)["S"], a_atual)

    c2, A_ub2, b_ub2, A_eq2, b_eq2, b_cont2 = montar_lp_etapa_modelo_2(
        inst, pre, etapa=2, modo=modo, eps=eps, incl_star=incl_star
    )
    r2 = branch_and_bound_lp(
        c2, A_ub2, b_ub2, A_eq2, b_eq2, b_cont2, raiz_fix, x, y,
        incumbente_inicial=inc2,
        time_limit=time_limit_etapa,
        max_nos=max_nos_etapa,
    )
    if not r2["viavel"]:
        return {
            "viavel": False,
            "etapa_falha": 2,
            "status": r2["status"],
            "pre": pre,
            "vars_ativas": vars_ativas,
            "tempo_pre_s": tempo_pre,
            "historico": historico,
        }
    a_atual = r2["best_a"]
    m2 = calcular_metricas_modelo_2(inst, pre, a_atual)
    s_star = m2["S"]
    historico.append({"etapa": "Etapa 2 (QoS / S)", **r2, **m2})

    # ------------------------------------------------------------------
    # ETAPA 3: GARGALO (minimizar U_max)
    # ------------------------------------------------------------------
    inc3 = (m2["U_max"], a_atual)
    c3, A_ub3, b_ub3, A_eq3, b_eq3, b_cont3 = montar_lp_etapa_modelo_2(
        inst, pre, etapa=3, modo=modo, eps=eps, incl_star=incl_star, s_star=s_star
    )
    r3 = branch_and_bound_lp(
        c3, A_ub3, b_ub3, A_eq3, b_eq3, b_cont3, raiz_fix, x, y,
        incumbente_inicial=inc3,
        time_limit=time_limit_etapa,
        max_nos=max_nos_etapa,
    )
    if not r3["viavel"]:
        return {
            "viavel": False,
            "etapa_falha": 3,
            "status": r3["status"],
            "pre": pre,
            "vars_ativas": vars_ativas,
            "tempo_pre_s": tempo_pre,
            "historico": historico,
        }
    a_atual = r3["best_a"]
    m3 = calcular_metricas_modelo_2(inst, pre, a_atual)
    umax_star = m3["U_max"]
    historico.append({"etapa": "Etapa 3 (Gargalo)", **r3, **m3})

    # ------------------------------------------------------------------
    # ETAPA 4: EQUILIBRIO ENTRE SERVIDORES (opcional, desvio absoluto L1)
    # ------------------------------------------------------------------
    if executar_etapa_4:
        inc4 = (m3["D"], a_atual)
        c4, A_ub4, b_ub4, A_eq4, b_eq4, b_cont4 = montar_lp_etapa_modelo_2(
            inst,
            pre,
            etapa=4,
            modo=modo,
            eps=eps,
            incl_star=incl_star,
            s_star=s_star,
            umax_star=umax_star,
        )
        r4 = branch_and_bound_lp(
            c4, A_ub4, b_ub4, A_eq4, b_eq4, b_cont4, raiz_fix, x, y,
            incumbente_inicial=inc4,
            time_limit=time_limit_etapa,
            max_nos=max_nos_etapa,
        )
        if not r4["viavel"]:
            return {
                "viavel": False,
                "etapa_falha": 4,
                "status": r4["status"],
                "pre": pre,
                "vars_ativas": vars_ativas,
                "tempo_pre_s": tempo_pre,
                "historico": historico,
            }
        a_atual = r4["best_a"]
        m4 = calcular_metricas_modelo_2(inst, pre, a_atual)
        historico.append({"etapa": "Etapa 4 (Equilibrio)", **r4, **m4})

    assert a_atual is not None
    ok, erros = validar_solucao_modelo_2(inst, pre, a_atual, modo=modo)
    otimo_global = all(h["otimo_provado"] for h in historico)
    tempo_bnb = sum(h["tempo_s"] for h in historico)

    return {
        "viavel": True,
        "valido": ok,
        "otimo_global": otimo_global,
        "erros_validacao": erros,
        "pre": pre,
        "vars_ativas": vars_ativas,
        "vars_totais": x * y,
        "alocacao": a_atual,
        "metricas": historico[-1],
        "historico": historico,
        "tempo_pre_s": tempo_pre,
        "tempo_total_s": tempo_pre + tempo_bnb,
        "nos_totais": sum(h["nos_resolvidos"] for h in historico),
    }


def imprimir_relatorio_m2_bnb(
    titulo: str,
    inst: InstanciaModelo2,
    res: Dict[str, Any],
    mostrar_alocacao: bool = True,
) -> None:
    print("\n" + "=" * 94)
    print(titulo)
    print("=" * 94)
    if not res["viavel"]:
        print(f"Resultado: {res['status']} na Etapa {res['etapa_falha']}.")
        pre = res["pre"]
        sem_eleg = [
            inst.servicos[j]
            for j in range(len(inst.servicos))
            if sum(pre.E[i][j] for i in range(len(inst.servidores))) == 0
        ]
        if sem_eleg:
            print(f"Servicos sem nenhum servidor elegivel: {', '.join(sem_eleg)}")
        return

    rotulo_opt = (
        "OTIMO GLOBAL PROVADO"
        if res["otimo_global"]
        else "SOLUCAO VIAVEL (LIMITE ATINGIDO EM ALGUMA ETAPA)"
    )
    m = res["metricas"]
    print(
        f"Status: {rotulo_opt} | Auditoria (R1-R3): {'OK' if res['valido'] else 'FALHOU'} | "
        f"Binarias elegiveis: {res['vars_ativas']}/{res['vars_totais']} | "
        f"Nos LP: {res['nos_totais']} | Tempo total: {res['tempo_total_s']:.4f} s"
    )
    print(
        f"Distribuicao por arquitetura -> X86: {m['contagem_arq']['X86']} servicos | "
        f"ARM: {m['contagem_arq']['ARM']} servicos"
    )
    if res["erros_validacao"]:
        for erro in res["erros_validacao"]:
            print(f"  [ERRO] {erro}")

    print("\nEvolucao por etapa lexicografica (Branch-and-Bound):")
    print(
        f"  {'Etapa':<21} {'Status':<12} {'Incl':>7} {'S (QoS)':>9} "
        f"{'U_max':>7} {'Desvio D':>9} {'Nos':>6} {'Gap%':>6} {'Tempo(s)':>9}"
    )
    for h in res["historico"]:
        gap_pct = (h["gap_rel"] or 0.0) * 100.0
        print(
            f"  {h['etapa']:<21} {h['status']:<12} {h['Incl']:>7.4f} "
            f"{h['S']:>9.6f} {h['U_max']:>7.4f} {h['D']:>9.6f} "
            f"{h['nos_resolvidos']:>6d} {gap_pct:>6.2f} {h['tempo_s']:>9.4f}"
        )

    print("\nEstatisticas de poda por etapa:")
    for h in res["historico"]:
        print(
            f"  {h['etapa']:<21} -> Inviabilidade: {h['podas_inviabilidade']:<5d} | "
            f"Limitante: {h['podas_limitante']:<5d} | "
            f"Integralidade: {h['podas_integralidade']:<5d}"
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
# SIMULACOES DE VALIDACAO E ESCALABILIDADE DO MODELO 2 (B&B)
# ==========================================================================

def cenario_1_especializacao_arm_x86() -> InstanciaModelo2:
    """Mesma instancia do script HiGHS: especializacao ARM vs x86."""
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
    """Mesma instancia do script HiGHS: A[j,t], q[j], c_lat no pico e e[j,t]."""
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
    """Mesma instancia do script HiGHS: cluster so ARM com servico inelegivel."""
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


def cenario_4_cluster_misto_pequeno(seed: int = 7) -> InstanciaModelo2:
    """Cluster misto intermediario (3 servidores x 9 servicos = 27 binarias)."""
    rng = random.Random(seed)
    servidores = ["S1_IceLake_x86", "S2_Zen3_x86", "S3_Altra_ARM"]
    arch = ["X86", "X86", "ARM"]
    m = [32.0, 64.0, 80.0]
    v = [1.00, 0.87, 0.51]
    C_outros = [
        [25.0, 128.0, 1000.0],
        [25.0, 192.0, 1200.0],
        [25.0, 192.0, 1200.0],
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

    for j in range(9):
        if j < 3:
            servicos.append(f"WebPar_{j+1:02d}")
            wm_j = round(rng.uniform(4.0, 7.0), 2)
            wp_j = round(wm_j * rng.uniform(1.20, 1.35), 2)
            phi_j = round(rng.uniform(0.85, 0.96), 2)
            t1_j = round(rng.uniform(30.0, 45.0), 1)
            l_j = round(rng.uniform(80.0, 110.0), 1)
            q_j = 64.0
            e_j = {"ARM": round(rng.uniform(0.85, 1.00), 2), "X86": 1.0}
            a_j = {"ARM": 1, "X86": 1}
        elif j < 6:
            servicos.append(f"ApiMista_{j-2:02d}")
            wm_j = round(rng.uniform(3.0, 5.5), 2)
            wp_j = round(wm_j * rng.uniform(1.20, 1.35), 2)
            phi_j = round(rng.uniform(0.50, 0.72), 2)
            t1_j = round(rng.uniform(30.0, 40.0), 1)
            l_j = round(rng.uniform(55.0, 72.0), 1)
            q_j = 32.0
            e_j = {"ARM": round(rng.uniform(0.75, 0.95), 2), "X86": 1.0}
            a_j = {"ARM": 1, "X86": 1}
        else:
            servicos.append(f"SeqLat_{j-5:02d}")
            wm_j = round(rng.uniform(2.0, 4.0), 2)
            wp_j = round(wm_j * rng.uniform(1.15, 1.30), 2)
            phi_j = round(rng.uniform(0.20, 0.35), 2)
            t1_j = round(rng.uniform(36.0, 44.0), 1)
            l_j = round(rng.uniform(48.0, 58.0), 1)
            q_j = 16.0
            e_j = {"ARM": 0.85, "X86": 1.0}
            a_j = {"ARM": 0 if j == 8 else 1, "X86": 1}

        rede_m = round(rng.uniform(1.5, 3.5), 2)
        ram_m = round(rng.uniform(8.0, 24.0), 2)
        arm_m = round(rng.uniform(50.0, 150.0), 2)

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
        Dp_outros.append([
            round(rede_m * 1.30, 2),
            round(ram_m * 1.25, 2),
            round(arm_m * 1.15, 2),
        ])

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


def cenario_5_cluster_misto_medio(
    seed: int = 42,
    v_arm: float = 0.51,
) -> InstanciaModelo2:
    """Mesma instancia 6x24 do script HiGHS para comparacao direta de escalabilidade."""
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
    res1 = resolver_modelo_2_bnb(inst1, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio_m2_bnb(
        "SIMULACAO 1 (Modelo 2 - B&B): Especializacao ARM vs x86", inst1, res1
    )
    assert res1["viavel"] and res1["valido"] and res1["otimo_global"]
    assert res1["pre"].E[1][2] == 0
    assert res1["metricas"]["alocados"]["J3_SeqRigido"] == "S1_x86"
    assert res1["metricas"]["alocados"]["J1_Web1"] == "S2_ARM"
    assert res1["metricas"]["alocados"]["J2_Web2"] == "S2_ARM"
    assert abs(res1["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res1["metricas"]["U_max"] - 0.40) < 1e-6
    assert abs(res1["metricas"]["D"] - 0.003125) < 1e-6

    # Simulacao 2: Compatibilidade A, limite q[j], c_lat dominando pico e e[j,t]
    inst2 = cenario_2_regras_preprocessamento()
    res2 = resolver_modelo_2_bnb(inst2, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio_m2_bnb(
        "SIMULACAO 2 (Modelo 2 - B&B): Validacao de A[j,t], q[j], c_lat e e[j,t]",
        inst2,
        res2,
    )
    assert res2["viavel"] and res2["valido"] and res2["otimo_global"]
    assert res2["pre"].E[1][0] == 0
    assert res2["pre"].E[1][1] == 0
    assert res2["pre"].E[1][2] == 1
    assert res2["metricas"]["alocados"]["J3_LatenciaDomina"] == "S1_x86"
    assert res2["metricas"]["alocados"]["J4_EscalaBaixaARM"] == "S2_ARM"
    assert abs(res2["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res2["metricas"]["U_max"] - 0.60) < 1e-6
    assert abs(res2["metricas"]["D"] - 0.0453125) < 1e-6

    # Simulacao 3A e 3B: Inelegibilidade total no obrigatorio -> diagnostico opcional
    inst3 = cenario_3_diagnostico_inelegibilidade()
    res3a = resolver_modelo_2_bnb(inst3, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio_m2_bnb(
        "SIMULACAO 3A (Modelo 2 - B&B): Cluster so ARM no modo obrigatorio",
        inst3,
        res3a,
    )
    assert not res3a["viavel"]

    res3b = resolver_modelo_2_bnb(inst3, modo="opcional", executar_etapa_4=True)
    imprimir_relatorio_m2_bnb(
        "SIMULACAO 3B (Modelo 2 - B&B): Diagnostico no modo opcional",
        inst3,
        res3b,
    )
    assert res3b["viavel"] and res3b["valido"] and res3b["otimo_global"]
    assert abs(res3b["metricas"]["Incl"] - (8.0 / 18.0)) < 1e-6
    assert abs(res3b["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res3b["metricas"]["U_max"] - 0.40) < 1e-6
    assert abs(res3b["metricas"]["D"] - 0.0) < 1e-6
    assert res3b["metricas"]["nao_alocados"] == ["J3_SemElegivel"]

    # Simulacao 4: Cluster misto intermediario (3 servidores x 9 servicos = 27 binarias)
    inst4 = cenario_4_cluster_misto_pequeno(seed=7)
    res4 = resolver_modelo_2_bnb(
        inst4,
        modo="obrigatorio",
        executar_etapa_4=True,
        time_limit_etapa=15.0,
        max_nos_etapa=20000,
    )
    imprimir_relatorio_m2_bnb(
        "SIMULACAO 4 (Modelo 2 - B&B): Cluster misto intermediario (3x9)",
        inst4,
        res4,
    )

    # Simulacao 5: Mesma instancia 6x24 do script HiGHS para contraste de escalabilidade
    inst5 = cenario_5_cluster_misto_medio(seed=42, v_arm=0.51)
    res5 = resolver_modelo_2_bnb(
        inst5,
        modo="obrigatorio",
        executar_etapa_4=True,
        time_limit_etapa=5.0,
        max_nos_etapa=4000,
    )
    imprimir_relatorio_m2_bnb(
        "SIMULACAO 5 (Modelo 2 - B&B): Teste de escalabilidade no cluster misto (6x24)",
        inst5,
        res5,
        mostrar_alocacao=False,
    )

    print("\nTodas as verificacoes analiticas do Modelo 2 (B&B) passaram.")