"""Experimento extra: o que acontece se o Modelo 1 (sem arquitetura) for usado no cluster do Modelo 2?

A alocacao obtida com o Modelo 1 (demanda de CPU igual em todos os servidores) e avaliada
com as regras do Modelo 2 (elegibilidade e demanda efetiva por arquitetura).
"""
import sys
# Caminho do repositorio na maquina local. Para rodar em outro computador
# e presciso ajustar essa linha.
REPO = "/home/user/Documents/Pesquisa_Operacional"
sys.path.insert(0, REPO + "/Modelo_2/res_1")
sys.path.insert(0, REPO + "/Modelo_1/res_1")
import importlib.util

# Os dois arquivos se chamam HiGHS.py, entao cada um e carregado direto pelo
# caminho com um nome diferente (H1 e H2) para nao dar conflito.
def carregar(nome, caminho):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m

H2 = carregar("H2", REPO + "/Modelo_2/res_1/HiGHS.py")
H1 = carregar("H1", REPO + "/Modelo_1/res_1/HiGHS.py")

# O cenario do Modelo 2 retorna apenas o modelo ja pre-processado. Para pegar os
# dados de entrada originais, a funcao construir_modelo_2 e substituida por uma
# que salva os argumentos e depois chama a original. Assim os dois modelos usam
# exatamente os mesmos dado.
capturado = {}
orig = H2.construir_modelo_2
def captura(*args, **kw):
    capturado["args"] = args
    return orig(*args, **kw)
H2.construir_modelo_2 = captura

# Testa uma alocacao a nas regras do Modelo 2. Retorna:
# inel = servicos colocados em servidor inelegivel
# viol = capacidades de pico ultrapassadas (servidor, recurso, ocupacao)
# Obs: pares inelegiveis tem demanda zerada em Dp_ef, entao eles nao entram na
# soma do pico (em nenhum recurso).
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

# Repete o experimento com duas velocidades relativas do nucleo ARM.
for v_arm in (0.51, 0.80):
    m2 = H2.cenario_4_misto_6x24(seed=42, v_arm=v_arm)
    # Desempacota os dados capturados, na mesma ordem dos parametros da funcao.
    (servidores, arch, m, v, C_out, servicos, p, Wp, Wm, phi, T1, L, q, e, A, Dp_out, Dm_out) = capturado["args"][:17]
    # Monta a instancia do Modelo 1 com os mesmos dados. A CPU usa Wp e Wm
    # direto, sem considerar velocidade, eficiencia, latencia ou compatibilidade
    # de arquitetura.
    inst1 = H1.InstanciaModelo1(
        servidores=servidores, servicos=servicos,
        C=[[m[i], *C_out[i]] for i in range(len(servidores))],
        Dp=[[Wp[j], *Dp_out[j]] for j in range(len(servicos))],
        Dm=[[Wm[j], *Dm_out[j]] for j in range(len(servicos))],
        p=p,
    )
    # Resolve o Modelo 1 no modo obrigatorio, com limite de 30 s por etapa.
    # Se nao tiver solucao, passa para o proximo valor de v_arm.
    r1 = H1.resolver_modelo_1(inst1, modo="obrigatorio", time_limit=30.0)
    print(f"\n=== v_ARM = {v_arm}")
    if not r1["viavel"]:
        print("Modelo 1 sem solucao:", r1["status"]); continue
    # Metricas segundo o proprio Modelo 1 e quantos servicos foram para x86 e ARM.
    mm = r1["metricas"]
    print(f"Modelo 1 (ignora arquitetura): status={'OTIMO' if r1['otimo_global'] else 'LIMITE'} "
          f"U_max_modelo1={mm['U_max']:.4f} D={mm['D']:.4f} "
          f"servicos em x86/ARM = {sum(r1['alocacao'][i][j] for i in range(6) for j in range(24) if arch[i]=='X86')}/"
          f"{sum(r1['alocacao'][i][j] for i in range(6) for j in range(24) if arch[i]=='ARM')}")
    # Agora a mesma alocacao e avaliada com as regras do Modelo 2.
    inel, viol = avaliar_no_modelo_2(m2, r1["alocacao"])
    print(f"  Avaliado no Modelo 2: servicos em servidor inelegivel = {len(inel)} -> {inel}")
    print(f"  Avaliado no Modelo 2: violacoes de capacidade de pico = {len(viol)} -> "
          + ", ".join(f"{s}/{k} {u*100:.0f}%" for s, k, u in viol))
    # Como no pico, pares inelegiveis contribuem com zero. Por isso, quando existe
    # servico inelegivel, a utilizacao maxima abaixo fica subestimanda.
    # utilizacao media real apos aplicar demanda efetiva (apenas onde elegivel)
    x, y = 6, 24
    a = r1["alocacao"]
    u = [[sum(m2.Dm_ef[i][j][k] * a[i][j] for j in range(y)) / m2.C[i][k] for k in range(4)] for i in range(x)]
    print(f"  Utilizacao media real maxima (Modelo 2) = {max(max(r) for r in u):.4f}")
    # Para comparar, resolve o mesmo cenario com o Modelo 2, que ja considera
    # a arquitetura na hora de otimizar. Essa parte espera que exista solucao
    # viavel, senao as chaves U_max e arq nao existem.
    r2 = H2.resolver_modelo_2_highs(m2, "obrigatorio", time_limit=30.0)
    print(f"Modelo 2 (considera arquitetura): status={r2['status']} U_max={r2['U_max']:.4f} "
          f"x86/ARM={r2['arq']['X86']}/{r2['arq']['ARM']}")
