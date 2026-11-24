from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


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


def montar_lp_etapa(
    inst: InstanciaModelo1,
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
    Constroi a formulacao matricial da relaxacao linear (min c^T z)
    para a etapa lexicografica especificada (1, 2, 3 ou 4).
    """
    x = len(inst.servidores)
    y = len(inst.servicos)
    r = len(inst.recursos)
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
        # Maximizar Incl equivale a minimizar -Incl
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

    # (R2) Capacidade multidimensional no pico
    for i in range(x):
        for k in range(r):
            row = np.zeros(n_vars, dtype=float)
            for j in range(y):
                row[i * y + j] = inst.Dp[j][k]
            A_ub_rows.append(row)
            b_ub_vals.append(inst.C[i][k])

    # Restricao da Etapa 1 fixada nas etapas seguintes (modo opcional)
    if etapa >= 2 and modo == "opcional":
        assert incl_star is not None
        row = np.zeros(n_vars, dtype=float)
        for i in range(x):
            for j in range(y):
                row[i * y + j] = -inst.p[j] / peso_total
        A_ub_rows.append(row)
        b_ub_vals.append(-(incl_star - eps))

    # (R3) Excesso acima da zona segura: u[i,k] - s[i,k] <= tau[k]
    if etapa >= 2:
        for i in range(x):
            for k in range(r):
                row = np.zeros(n_vars, dtype=float)
                for j in range(y):
                    row[i * y + j] = inst.Dm[j][k] / inst.C[i][k]
                row[n_a + i * r + k] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(inst.tau[k])

    # Restricao da Etapa 2 fixada e (R4) Gargalo Min-Max
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
                    row[i * y + j] = inst.Dm[j][k] / inst.C[i][k]
                row[idx_umax] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(0.0)

    # Restricao da Etapa 3 fixada e linearizacao do desvio absoluto L1
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
                        coef = fator * (inst.Dm[j][k] / inst.C[i2][k])
                        row_pos[i2 * y + j] = coef
                        row_neg[i2 * y + j] = -coef
                idx_d = offset_d + i * r + k
                row_pos[idx_d] = -1.0
                row_neg[idx_d] = -1.0
                A_ub_rows.append(row_pos)
                b_ub_vals.append(0.0)
                A_ub_rows.append(row_neg)
                b_ub_vals.append(0.0)

    # Limites das variaveis continuas (a[i,j] tera seus limites gerenciados no B&B)
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
    """Regra classica: seleciona a variavel binaria mais proxima de 0.5."""
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

    raiz_fix = (-1,) * n_bin
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

    # Fila de prioridade: (limitante_dual, id_no, fixacoes, idx_ramificacao)
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


def resolver_modelo_1_bnb(
    inst: InstanciaModelo1,
    modo: str = "obrigatorio",
    executar_etapa_4: bool = True,
    eps: float = 1e-6,
    time_limit_etapa: float = 30.0,
    max_nos_etapa: int = 50000,
) -> Dict[str, Any]:
    """Resolve o Modelo 1 em etapas lexicograficas com Branch-and-Bound classico."""
    if modo not in ("obrigatorio", "opcional"):
        raise ValueError("modo deve ser 'obrigatorio' ou 'opcional'.")

    x = len(inst.servidores)
    y = len(inst.servicos)

    historico: List[Dict[str, Any]] = []
    a_atual: Optional[List[List[int]]] = None
    incl_star = 1.0
    s_star = 0.0
    umax_star = 1.0

    # ------------------------------------------------------------------
    # ETAPA 1: INCLUSAO (apenas no modo opcional)
    # ------------------------------------------------------------------
    if modo == "opcional":
        c1, A_ub1, b_ub1, A_eq1, b_eq1, b_cont1 = montar_lp_etapa(
            inst, etapa=1, modo=modo, eps=eps
        )
        r1 = branch_and_bound_lp(
            c1, A_ub1, b_ub1, A_eq1, b_eq1, b_cont1, x, y,
            incumbente_inicial=None,
            time_limit=time_limit_etapa,
            max_nos=max_nos_etapa,
        )
        if not r1["viavel"]:
            return {
                "viavel": False,
                "etapa_falha": 1,
                "status": r1["status"],
                "historico": historico,
            }
        a_atual = r1["best_a"]
        m1 = calcular_metricas(inst, a_atual)
        incl_star = m1["Incl"]
        historico.append({"etapa": "Etapa 1 (Inclusao)", **r1, **m1})

    # ------------------------------------------------------------------
    # ETAPA 2: QUALIDADE DE SERVICO (minimizar folga media S)
    # ------------------------------------------------------------------
    inc2 = None
    if a_atual is not None:
        inc2 = (calcular_metricas(inst, a_atual)["S"], a_atual)

    c2, A_ub2, b_ub2, A_eq2, b_eq2, b_cont2 = montar_lp_etapa(
        inst, etapa=2, modo=modo, eps=eps, incl_star=incl_star
    )
    r2 = branch_and_bound_lp(
        c2, A_ub2, b_ub2, A_eq2, b_eq2, b_cont2, x, y,
        incumbente_inicial=inc2,
        time_limit=time_limit_etapa,
        max_nos=max_nos_etapa,
    )
    if not r2["viavel"]:
        return {
            "viavel": False,
            "etapa_falha": 2,
            "status": r2["status"],
            "historico": historico,
        }
    a_atual = r2["best_a"]
    m2 = calcular_metricas(inst, a_atual)
    s_star = m2["S"]
    historico.append({"etapa": "Etapa 2 (QoS / S)", **r2, **m2})

    # ------------------------------------------------------------------
    # ETAPA 3: GARGALO (minimizar U_max)
    # ------------------------------------------------------------------
    inc3 = (m2["U_max"], a_atual)
    c3, A_ub3, b_ub3, A_eq3, b_eq3, b_cont3 = montar_lp_etapa(
        inst, etapa=3, modo=modo, eps=eps, incl_star=incl_star, s_star=s_star
    )
    r3 = branch_and_bound_lp(
        c3, A_ub3, b_ub3, A_eq3, b_eq3, b_cont3, x, y,
        incumbente_inicial=inc3,
        time_limit=time_limit_etapa,
        max_nos=max_nos_etapa,
    )
    if not r3["viavel"]:
        return {
            "viavel": False,
            "etapa_falha": 3,
            "status": r3["status"],
            "historico": historico,
        }
    a_atual = r3["best_a"]
    m3 = calcular_metricas(inst, a_atual)
    umax_star = m3["U_max"]
    historico.append({"etapa": "Etapa 3 (Gargalo)", **r3, **m3})

    # ------------------------------------------------------------------
    # ETAPA 4: EQUILIBRIO ENTRE SERVIDORES (opcional, desvio absoluto L1)
    # ------------------------------------------------------------------
    if executar_etapa_4:
        inc4 = (m3["D"], a_atual)
        c4, A_ub4, b_ub4, A_eq4, b_eq4, b_cont4 = montar_lp_etapa(
            inst,
            etapa=4,
            modo=modo,
            eps=eps,
            incl_star=incl_star,
            s_star=s_star,
            umax_star=umax_star,
        )
        r4 = branch_and_bound_lp(
            c4, A_ub4, b_ub4, A_eq4, b_eq4, b_cont4, x, y,
            incumbente_inicial=inc4,
            time_limit=time_limit_etapa,
            max_nos=max_nos_etapa,
        )
        if not r4["viavel"]:
            return {
                "viavel": False,
                "etapa_falha": 4,
                "status": r4["status"],
                "historico": historico,
            }
        a_atual = r4["best_a"]
        m4 = calcular_metricas(inst, a_atual)
        historico.append({"etapa": "Etapa 4 (Equilibrio)", **r4, **m4})

    assert a_atual is not None
    ok, erros = validar_solucao(inst, a_atual, modo=modo)
    otimo_global = all(h["otimo_provado"] for h in historico)

    return {
        "viavel": True,
        "valido": ok,
        "otimo_global": otimo_global,
        "erros_validacao": erros,
        "alocacao": a_atual,
        "metricas": historico[-1],
        "historico": historico,
        "tempo_total_s": sum(h["tempo_s"] for h in historico),
        "nos_totais": sum(h["nos_resolvidos"] for h in historico),
    }


def imprimir_relatorio_bnb(
    titulo: str,
    inst: InstanciaModelo1,
    res: Dict[str, Any],
) -> None:
    print("\n" + "=" * 92)
    print(titulo)
    print("=" * 92)
    if not res["viavel"]:
        if res["status"] == "INVIAVEL":
            print(f"Resultado: INVIAVEL (provado) na Etapa {res['etapa_falha']}.")
        else:
            print(f"Resultado: {res['status']} na Etapa {res['etapa_falha']}. "
                  f"Nada foi provado: nenhuma solucao viavel foi encontrada dentro do limite.")
        return

    rotulo_opt = (
        "OTIMO GLOBAL PROVADO"
        if res["otimo_global"]
        else "SOLUCAO VIAVEL (LIMITE ATINGIDO EM ALGUMA ETAPA)"
    )
    print(
        f"Status: {rotulo_opt} | Auditoria: {'OK' if res['valido'] else 'FALHOU'} | "
        f"Nos LP totais: {res['nos_totais']} | Tempo total: {res['tempo_total_s']:.4f} s"
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
# SIMULACOES DE VALIDACAO E ESCALABILIDADE
# ==========================================================================

def cenario_1_desempate_etapa_4() -> InstanciaModelo1:
    """Mesma instancia do script HiGHS: empate em U_max=0.60 quebrado pela Etapa 4."""
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
    """Mesma instancia do script HiGHS: sobrecarga no obrigatorio e pesos no opcional."""
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
    """Mesma instancia do script HiGHS: picos impedem consolidacao insegura."""
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


def cenario_4_cluster_pequeno_aleatorio(seed: int = 7) -> InstanciaModelo1:
    """Cluster heterogeneo intermediario (3 servidores x 9 servicos = 27 binarias)."""
    rng = random.Random(seed)
    servidores = ["S1_Grande", "S2_Medio", "S3_Pequeno"]
    C = [
        [24.0, 20.0, 96.0, 600.0],
        [16.0, 15.0, 64.0, 400.0],
        [10.0, 10.0, 32.0, 250.0],
    ]
    y = 9
    servicos = [f"Srv_{j+1:02d}" for j in range(y)]
    Dm: List[List[float]] = []
    Dp: List[List[float]] = []
    p: List[float] = []

    for _ in range(y):
        cpu_m = round(rng.uniform(1.5, 4.0), 2)
        rede_m = round(rng.uniform(1.0, 3.0), 2)
        ram_m = round(rng.uniform(4.0, 14.0), 2)
        arm_m = round(rng.uniform(30.0, 90.0), 2)

        cpu_p = round(cpu_m * rng.uniform(1.20, 1.40), 2)
        rede_p = round(rede_m * rng.uniform(1.20, 1.40), 2)
        ram_p = round(ram_m * rng.uniform(1.15, 1.30), 2)
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


def cenario_5_cluster_medio_highs(seed: int = 42) -> InstanciaModelo1:
    """Mesma instancia 6x24 (144 binarias) do script HiGHS para teste de escalabilidade."""
    rng = random.Random(seed)
    servidores = ["S1_G", "S2_G", "S3_M", "S4_M", "S5_P", "S6_P"]
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
    # Simulacao 1: Desempate da Etapa 4 (15 variaveis binarias)
    inst1 = cenario_1_desempate_etapa_4()
    res1 = resolver_modelo_1_bnb(inst1, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio_bnb(
        "SIMULACAO 1 (B&B): Heterogeneo viavel e efeito da Etapa 4", inst1, res1
    )
    assert res1["viavel"] and res1["valido"] and res1["otimo_global"]
    assert abs(res1["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res1["metricas"]["U_max"] - 0.60) < 1e-6
    assert abs(res1["metricas"]["D"] - (1.0 / 30.0)) < 1e-5

    # Simulacao 2A e 2B: Sobrecarga e diagnostico opcional (8 variaveis binarias)
    inst2 = cenario_2_sobrecarga_diagnostico()
    res2a = resolver_modelo_1_bnb(inst2, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio_bnb(
        "SIMULACAO 2A (B&B): Sobrecarga no modo obrigatorio", inst2, res2a
    )
    assert not res2a["viavel"] and res2a["status"] == "INVIAVEL"

    res2b = resolver_modelo_1_bnb(inst2, modo="opcional", executar_etapa_4=True)
    imprimir_relatorio_bnb(
        "SIMULACAO 2B (B&B): Diagnostico no modo opcional com pesos", inst2, res2b
    )
    assert res2b["viavel"] and res2b["valido"] and res2b["otimo_global"]
    assert abs(res2b["metricas"]["Incl"] - (12.0 / 13.0)) < 1e-6
    assert abs(res2b["metricas"]["S"] - 0.015625) < 1e-6
    assert abs(res2b["metricas"]["U_max"] - 0.875) < 1e-6
    assert abs(res2b["metricas"]["D"] - 0.1125) < 1e-6
    assert res2b["metricas"]["nao_alocados"] == ["J4_Baixo"]

    # Simulacao 3: Separacao entre pico e media (6 variaveis binarias)
    inst3 = cenario_3_separacao_pico_media()
    res3 = resolver_modelo_1_bnb(inst3, modo="obrigatorio", executar_etapa_4=True)
    imprimir_relatorio_bnb(
        "SIMULACAO 3 (B&B): Separacao entre pico e media", inst3, res3
    )
    assert res3["viavel"] and res3["valido"] and res3["otimo_global"]
    assert abs(res3["metricas"]["S"] - 0.0) < 1e-6
    assert abs(res3["metricas"]["U_max"] - 0.50) < 1e-6
    assert abs(res3["metricas"]["D"] - 0.0921875) < 1e-6

    # Simulacao 4: Instancia heterogenea intermediaria (3x9 = 27 binarias)
    inst4 = cenario_4_cluster_pequeno_aleatorio(seed=7)
    res4 = resolver_modelo_1_bnb(
        inst4,
        modo="obrigatorio",
        executar_etapa_4=True,
        time_limit_etapa=15.0,
        max_nos_etapa=20000,
    )
    imprimir_relatorio_bnb(
        "SIMULACAO 4 (B&B): Cluster heterogeneo intermediario (3x9)", inst4, res4
    )

    # Simulacao 5: Mesma instancia 6x24 (144 binarias) do script HiGHS
    # Limites curtos por etapa para observar o contraste de escalabilidade com o HiGHS
    inst5 = cenario_5_cluster_medio_highs(seed=42)
    res5 = resolver_modelo_1_bnb(
        inst5,
        modo="obrigatorio",
        executar_etapa_4=True,
        time_limit_etapa=5.0,
        max_nos_etapa=4000,
    )
    imprimir_relatorio_bnb(
        "SIMULACAO 5 (B&B): Teste de escalabilidade no cluster medio (6x24)",
        inst5,
        res5,
    )

    print("\nTodas as verificacoes analiticas do Branch-and-Bound passaram.")