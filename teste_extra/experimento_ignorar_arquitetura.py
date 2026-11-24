"""Experimento extra: o que acontece se o Modelo 1 (sem arquitetura) for usado no cluster do Modelo 2?

A alocacao obtida com o Modelo 1 (demanda de CPU igual em todos os servidores) e avaliada
com as regras do Modelo 2 (elegibilidade e demanda efetiva por arquitetura).
"""
import sys
REPO = "/home/user/Documents/Pesquisa_Operacional"
sys.path.insert(0, REPO + "/Modelo_2/res_1")
sys.path.insert(0, REPO + "/Modelo_1/res_1")
import importlib.util

def carregar(nome, caminho):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m

H2 = carregar("H2", REPO + "/Modelo_2/res_1/HiGHS.py")
H1 = carregar("H1", REPO + "/Modelo_1/res_1/HiGHS.py")

capturado = {}
orig = H2.construir_modelo_2
def captura(*args, **kw):
    capturado["args"] = args
    return orig(*args, **kw)
H2.construir_modelo_2 = captura

def avaliar_no_modelo_2(m2, a):
    x, y = len(m2.servidores), len(m2.servicos)
    inel = [(m2.servicos[j], m2.servidores[i]) for i in range(x) for j in range(y) if a[i][j] and not m2.E[i][j]]
    viol = []
    for i in range(x):
        for k in range(4):
            pico = sum(m2.Dp_ef[i][j][k] * a[i][j] for j in range(y))
            if pico > m2.C[i][k] + 1e-6:
                viol.append((m2.servidores[i], ["CPU", "Rede", "RAM", "Armaz."][k], pico / m2.C[i][k]))
    return inel, viol

for v_arm in (0.51, 0.80):
    m2 = H2.cenario_4_misto_6x24(seed=42, v_arm=v_arm)
    (servidores, arch, m, v, C_out, servicos, p, Wp, Wm, phi, T1, L, q, e, A, Dp_out, Dm_out) = capturado["args"][:17]
    inst1 = H1.InstanciaModelo1(
        servidores=servidores, servicos=servicos,
        C=[[m[i], *C_out[i]] for i in range(len(servidores))],
        Dp=[[Wp[j], *Dp_out[j]] for j in range(len(servicos))],
        Dm=[[Wm[j], *Dm_out[j]] for j in range(len(servicos))],
        p=p,
    )
    r1 = H1.resolver_modelo_1(inst1, modo="obrigatorio", time_limit=30.0)
    print(f"\n=== v_ARM = {v_arm}")
    if not r1["viavel"]:
        print("Modelo 1 sem solucao:", r1["status"]); continue
    mm = r1["metricas"]
    print(f"Modelo 1 (ignora arquitetura): status={'OTIMO' if r1['otimo_global'] else 'LIMITE'} "
          f"U_max_modelo1={mm['U_max']:.4f} D={mm['D']:.4f} "
          f"servicos em x86/ARM = {sum(r1['alocacao'][i][j] for i in range(6) for j in range(24) if arch[i]=='X86')}/"
          f"{sum(r1['alocacao'][i][j] for i in range(6) for j in range(24) if arch[i]=='ARM')}")
    inel, viol = avaliar_no_modelo_2(m2, r1["alocacao"])
    print(f"  Avaliado no Modelo 2: servicos em servidor inelegivel = {len(inel)} -> {inel}")
    print(f"  Avaliado no Modelo 2: violacoes de capacidade de pico = {len(viol)} -> "
          + ", ".join(f"{s}/{k} {u*100:.0f}%" for s, k, u in viol))
    # utilizacao media real apos aplicar demanda efetiva (apenas onde elegivel)
    x, y = 6, 24
    a = r1["alocacao"]
    u = [[sum(m2.Dm_ef[i][j][k] * a[i][j] for j in range(y)) / m2.C[i][k] for k in range(4)] for i in range(x)]
    print(f"  Utilizacao media real maxima (Modelo 2) = {max(max(r) for r in u):.4f}")
    r2 = H2.resolver_modelo_2_highs(m2, "obrigatorio", time_limit=30.0)
    print(f"Modelo 2 (considera arquitetura): status={r2['status']} U_max={r2['U_max']:.4f} "
          f"x86/ARM={r2['arq']['X86']}/{r2['arq']['ARM']}")
