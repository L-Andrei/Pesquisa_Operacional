#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <glpk.h>

// Estrutura que representa os dados de entrada da instancia do Modelo 1
struct InstanciaModelo1 {
    std::string nome_cenario;
    int x; // Quantidade de servidores fisicos (Conjunto I)
    int y; // Quantidade de servicos candidatos (Conjunto J)
    int R; // Quantidade de recursos computacionais (Conjunto K, R = 4)

    std::vector<std::vector<double>> C; // C[i][k]: Capacidade do recurso k no servidor i
    std::vector<std::vector<double>> D; // D[j][k]: Demanda do recurso k pelo servico j
    std::vector<double> p;              // p[j]: Prioridade de inclusao do servico j
    std::vector<double> tau;            // tau[k]: Limiar seguro de SLA/QoS do recurso k

    double w1, w2, w3; // Pesos da funcao objetivo escalarizada
};

// Estrutura que armazena a solucao e as metricas de verificacao
struct ResultadoMetodo {
    std::string nome_metodo;
    bool resolvido;
    std::string status_texto;
    double tempo_ms;
    double z_obtido;
    double z_auditado;

    std::vector<std::vector<int>> a;    // a[i][j] em {0, 1}
    std::vector<std::vector<double>> u; // u[i][k] em [0, 1]
    std::vector<std::vector<double>> s; // s[i][k] >= 0
    double U_max;

    int servicos_incluidos;
    double taxa_inclusao_ponderada;
    bool auditoria_aprovada;
};

// Gerador estocastico de casos de teste (identico ao usado no HiGHS para comparacao justa)
InstanciaModelo1 gerarCasoDeTesteRandomico(
    const std::string& nome,
    int x,
    int y,
    double fator_escala_demanda,
    unsigned int semente)
{
    InstanciaModelo1 inst;
    inst.nome_cenario = nome;
    inst.x = x;
    inst.y = y;
    inst.R = 4; // {0: CPU, 1: Rede, 2: RAM, 3: Armazenamento}

    std::mt19937 rng(semente);

    std::vector<double> cap_base_min = {16.0,  10.0,  64.0,  500.0};
    std::vector<double> cap_base_max = {64.0,  40.0, 256.0, 2000.0};

    inst.C.assign(x, std::vector<double>(inst.R, 0.0));
    std::vector<double> soma_cap_recurso(inst.R, 0.0);

    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < inst.R; ++k) {
            std::uniform_real_distribution<double> dist_cap(cap_base_min[k], cap_base_max[k]);
            inst.C[i][k] = std::round(dist_cap(rng));
            soma_cap_recurso[k] += inst.C[i][k];
        }
    }

    inst.D.assign(y, std::vector<double>(inst.R, 0.0));
    for (int j = 0; j < y; ++j) {
        for (int k = 0; k < inst.R; ++k) {
            double media_demanda = (soma_cap_recurso[k] * fator_escala_demanda) / static_cast<double>(y);
            std::uniform_real_distribution<double> dist_dem(0.35 * media_demanda, 1.65 * media_demanda);
            inst.D[j][k] = std::max(1.0, std::round(dist_dem(rng) * 10.0) / 10.0);
        }
    }

    inst.p.assign(y, 1.0);
    std::uniform_int_distribution<int> dist_prio(1, 10);
    double soma_p = 0.0;
    double min_p = 1e9;
    for (int j = 0; j < y; ++j) {
        inst.p[j] = static_cast<double>(dist_prio(rng));
        soma_p += inst.p[j];
        if (inst.p[j] < min_p) min_p = inst.p[j];
    }

    inst.tau = {0.75, 0.75, 0.80, 0.80};

    // Regra de calibracao lexicografica da Secao 7: w1 = 1.0 e w2 + w3 < min(p_j) / sum(p_j)
    double limite_seguro_pesos = min_p / soma_p;
    inst.w1 = 1.0;
    inst.w2 = 0.60 * limite_seguro_pesos;
    inst.w3 = 0.30 * limite_seguro_pesos;

    return inst;
}

// Funcao auxiliar que constroi o objeto glp_prob do Modelo 1 (indexacao 1-based do GLPK)
glp_prob* construirProblemaGLPK(const InstanciaModelo1& inst) {
    const int x = inst.x;
    const int y = inst.y;
    const int R = inst.R;

    // Mapeamento 1-based das colunas no GLPK
    auto col_a = [y](int i, int j) { return 1 + i * y + j; };
    auto col_u = [x, y, R](int i, int k) { return 1 + x * y + i * R + k; };
    const int col_Umax = 1 + x * y + x * R;
    auto col_s = [x, y, R](int i, int k) { return 1 + x * y + x * R + 1 + i * R + k; };

    const int num_cols = x * y + 2 * (x * R) + 1;
    // Total de linhas: R1 (y) + R3 (x*R) + R4 (x*R) + R5 (x*R)
    // Nota: Como u[i][k] in [0, 1] e (R3) fixa u[i][k] = sum(D*a)/C, o limite u[i][k] <= 1
    // ja impoe estritamente (R2) sum(D*a) <= C sem duplicar linhas na matriz barreira!
    const int num_rows = y + 3 * (x * R);

    glp_prob* lp = glp_create_prob();
    glp_set_prob_name(lp, inst.nome_cenario.c_str());
    glp_set_obj_dir(lp, GLP_MAX);

    glp_add_cols(lp, num_cols);
    glp_add_rows(lp, num_rows);

    double soma_p = std::accumulate(inst.p.begin(), inst.p.end(), 0.0);

    // Configura variaveis a[i][j] em {0, 1}
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            int c = col_a(i, j);
            glp_set_col_bnds(lp, c, GLP_DB, 0.0, 1.0);
            glp_set_col_kind(lp, c, GLP_BV);
            glp_set_obj_coef(lp, c, inst.w1 * (inst.p[j] / soma_p));
        }
    }

    // Configura variaveis u[i][k] em [0, 1] (impõe simultaneamente R2 e R6)
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            int c = col_u(i, k);
            glp_set_col_bnds(lp, c, GLP_DB, 0.0, 1.0);
            glp_set_col_kind(lp, c, GLP_CV);
            glp_set_obj_coef(lp, c, 0.0);
        }
    }

    // Configura variavel U_max em [0, 1]
    glp_set_col_bnds(lp, col_Umax, GLP_DB, 0.0, 1.0);
    glp_set_col_kind(lp, col_Umax, GLP_CV);
    glp_set_obj_coef(lp, col_Umax, -inst.w2);

    // Configura variaveis s[i][k] em [0, 1]
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            int c = col_s(i, k);
            glp_set_col_bnds(lp, c, GLP_DB, 0.0, 1.0);
            glp_set_col_kind(lp, c, GLP_CV);
            glp_set_obj_coef(lp, c, -inst.w3 / static_cast<double>(x * R));
        }
    }

    // Vetores 1-based para carga da matriz esparsa no GLPK
    std::vector<int> ia(1, 0);
    std::vector<int> ja(1, 0);
    std::vector<double> ar(1, 0.0);
    int row_idx = 1;

    // (R1) Unicidade de Alocacao: sum_{i in I} a[i][j] <= 1
    for (int j = 0; j < y; ++j) {
        glp_set_row_bnds(lp, row_idx, GLP_UP, 0.0, 1.0);
        for (int i = 0; i < x; ++i) {
            ia.push_back(row_idx);
            ja.push_back(col_a(i, j));
            ar.push_back(1.0);
        }
        row_idx++;
    }

    // (R3 + R2 acopladas) Taxa de Utilizacao e Capacidade: sum_{j in J} (D[j][k]/C[i][k])*a[i][j] - u[i][k] == 0
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            glp_set_row_bnds(lp, row_idx, GLP_FX, 0.0, 0.0);
            for (int j = 0; j < y; ++j) {
                ia.push_back(row_idx);
                ja.push_back(col_a(i, j));
                ar.push_back(inst.D[j][k] / inst.C[i][k]);
            }
            ia.push_back(row_idx);
            ja.push_back(col_u(i, k));
            ar.push_back(-1.0);
            row_idx++;
        }
    }

    // (R4) Gargalo Maximo: U_max - u[i][k] >= 0
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            glp_set_row_bnds(lp, row_idx, GLP_LO, 0.0, 0.0);
            ia.push_back(row_idx);
            ja.push_back(col_Umax);
            ar.push_back(1.0);

            ia.push_back(row_idx);
            ja.push_back(col_u(i, k));
            ar.push_back(-1.0);
            row_idx++;
        }
    }

    // (R5) Excesso Acima do Limiar tau[k]: s[i][k] - u[i][k] >= -tau[k]
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            glp_set_row_bnds(lp, row_idx, GLP_LO, -inst.tau[k], 0.0);
            ia.push_back(row_idx);
            ja.push_back(col_s(i, k));
            ar.push_back(1.0);

            ia.push_back(row_idx);
            ja.push_back(col_u(i, k));
            ar.push_back(-1.0);
            row_idx++;
        }
    }

    glp_load_matrix(lp, static_cast<int>(ia.size()) - 1, ia.data(), ja.data(), ar.data());
    return lp;
}

// Funcao que avalia analiticamente u, s, U_max e Z para qualquer matriz binaria a[i][j]
void atualizarVariaveisContinuas(const InstanciaModelo1& inst, ResultadoMetodo& res) {
    const int x = inst.x;
    const int y = inst.y;
    const int R = inst.R;

    res.u.assign(x, std::vector<double>(R, 0.0));
    res.s.assign(x, std::vector<double>(R, 0.0));
    res.U_max = 0.0;

    double soma_s = 0.0;
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            double carga = 0.0;
            for (int j = 0; j < y; ++j) {
                if (res.a[i][j] == 1) carga += inst.D[j][k];
            }
            res.u[i][k] = carga / inst.C[i][k];
            if (res.u[i][k] > res.U_max) res.U_max = res.u[i][k];
            res.s[i][k] = std::max(0.0, res.u[i][k] - inst.tau[k]);
            soma_s += res.s[i][k];
        }
    }

    double soma_p_alocados = 0.0;
    double soma_p_total = std::accumulate(inst.p.begin(), inst.p.end(), 0.0);
    res.servicos_incluidos = 0;

    for (int j = 0; j < y; ++j) {
        for (int i = 0; i < x; ++i) {
            if (res.a[i][j] == 1) {
                soma_p_alocados += inst.p[j];
                res.servicos_incluidos++;
                break;
            }
        }
    }

    res.taxa_inclusao_ponderada = soma_p_alocados / soma_p_total;
    res.z_obtido = inst.w1 * res.taxa_inclusao_ponderada
                 - inst.w2 * res.U_max
                 - inst.w3 * (soma_s / static_cast<double>(x * R));
}

// ============================================================================
// METODO 1 (ALTERNATIVO): MATHEURISTICA RELAX-AND-FIX VIA PONTOS INTERIORES
//                         (BARREIRA PRIMAL-DUAL) + BUSCA LOCAL 2-OPT (SWAP)
// ============================================================================
ResultadoMetodo resolverViaPontosInterioresMatheuristica(const InstanciaModelo1& inst) {
    ResultadoMetodo res;
    res.nome_metodo = "GLPK Pontos Interiores (Barreira) + Matheuristica Relax-and-Fix & Busca Local";
    res.resolvido = false;
    res.auditoria_aprovada = false;

    const int x = inst.x;
    const int y = inst.y;
    const int R = inst.R;
    auto col_a = [y](int i, int j) { return 1 + i * y + j; };

    glp_prob* lp = construirProblemaGLPK(inst);

    glp_iptcp parm_barreira;
    glp_init_iptcp(&parm_barreira);
    parm_barreira.msg_lev = GLP_MSG_OFF;
    parm_barreira.ord_alg = GLP_ORD_AMD; // Ordenacao de grau minimo aproximado na matriz barreira

    auto t_inicio = std::chrono::high_resolution_clock::now();

    // Fase 1: Resolve a relaxacao continua pelo Metodo de Pontos Interiores (sem Simplex)
    int ret_ipt = glp_interior(lp, &parm_barreira);
    int status_ipt = glp_ipt_status(lp);

    // Extrai a matriz de afinidade continua a_til[i][j] do ponto interior
    std::vector<std::vector<double>> a_til(x, std::vector<double>(y, 0.0));
    if (ret_ipt == 0 && (status_ipt == GLP_OPT || status_ipt == GLP_FEAS)) {
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                a_til[i][j] = glp_ipt_col_prim(lp, col_a(i, j));
            }
        }
    }
    glp_delete_prob(lp);

    // Fase 2: Construcao Inteira Guiada pela Solucao de Pontos Interiores (Relax-and-Fix)
    res.a.assign(x, std::vector<int>(y, 0));
    std::vector<std::vector<double>> uso_abs(x, std::vector<double>(R, 0.0));
    std::vector<int> srv_do_servico(y, -1);

    // Ordena os servicos por prioridade p[j] combinada com a confianca da barreira
    std::vector<int> ordem_servicos(y);
    std::iota(ordem_servicos.begin(), ordem_servicos.end(), 0);
    std::sort(ordem_servicos.begin(), ordem_servicos.end(), [&](int j1, int j2) {
        double max_a1 = 0.0, max_a2 = 0.0;
        for (int i = 0; i < x; ++i) {
            max_a1 = std::max(max_a1, a_til[i][j1]);
            max_a2 = std::max(max_a2, a_til[i][j2]);
        }
        return (inst.p[j1] * (1.0 + 0.2 * max_a1)) > (inst.p[j2] * (1.0 + 0.2 * max_a2));
    });

    for (int j : ordem_servicos) {
        int melhor_i = -1;
        double melhor_score = -1e18;

        for (int i = 0; i < x; ++i) {
            bool cabe = true;
            double novo_u_max_local = 0.0;
            double novo_excesso_sla = 0.0;

            for (int k = 0; k < R; ++k) {
                if (uso_abs[i][k] + inst.D[j][k] > inst.C[i][k] + 1e-7) {
                    cabe = false;
                    break;
                }
                double u_proj = (uso_abs[i][k] + inst.D[j][k]) / inst.C[i][k];
                novo_u_max_local = std::max(novo_u_max_local, u_proj);
                novo_excesso_sla += std::max(0.0, u_proj - inst.tau[k]);
            }

            if (cabe) {
                // Combina a direcao do Ponto Interior com a minimizacao de U_max e SLA
                double score = 0.35 * a_til[i][j] - 1.0 * novo_u_max_local - 0.5 * novo_excesso_sla;
                if (score > melhor_score) {
                    melhor_score = score;
                    melhor_i = i;
                }
            }
        }

        if (melhor_i != -1) {
            res.a[melhor_i][j] = 1;
            srv_do_servico[j] = melhor_i;
            for (int k = 0; k < R; ++k) uso_abs[melhor_i][k] += inst.D[j][k];
        }
    }

    atualizarVariaveisContinuas(inst, res);

    // Fase 3: Busca Local (Vizinhancas de Realocacao, Insercao de Nao-Alocados e Permutacao 2-Opt)
    bool melhorou = true;
    int iter = 0;
    while (melhorou && iter < 150) {
        melhorou = false;
        iter++;

        // Movimento A: Tenta realocar servico j do servidor i_orig para outro servidor i_dest
        for (int j = 0; j < y; ++j) {
            int i_orig = srv_do_servico[j];
            if (i_orig == -1) continue;

            for (int i_dest = 0; i_dest < x; ++i_dest) {
                if (i_dest == i_orig) continue;
                bool cabe = true;
                for (int k = 0; k < R; ++k) {
                    if (uso_abs[i_dest][k] + inst.D[j][k] > inst.C[i_dest][k] + 1e-7) {
                        cabe = false;
                        break;
                    }
                }
                if (!cabe) continue;

                double z_antigo = res.z_obtido;
                res.a[i_orig][j] = 0;
                res.a[i_dest][j] = 1;
                atualizarVariaveisContinuas(inst, res);

                if (res.z_obtido > z_antigo + 1e-7) {
                    srv_do_servico[j] = i_dest;
                    for (int k = 0; k < R; ++k) {
                        uso_abs[i_orig][k] -= inst.D[j][k];
                        uso_abs[i_dest][k] += inst.D[j][k];
                    }
                    melhorou = true;
                    break;
                } else {
                    res.a[i_dest][j] = 0;
                    res.a[i_orig][j] = 1;
                    atualizarVariaveisContinuas(inst, res);
                }
            }
        }

        // Movimento B: Permutacao (Swap 2-Opt) entre dois servicos j1 e j2 em servidores distintos
        for (int j1 = 0; j1 < y; ++j1) {
            int i1 = srv_do_servico[j1];
            if (i1 == -1) continue;
            for (int j2 = j1 + 1; j2 < y; ++j2) {
                int i2 = srv_do_servico[j2];
                if (i2 == -1 || i1 == i2) continue;

                bool cabe_troca = true;
                for (int k = 0; k < R; ++k) {
                    if (uso_abs[i1][k] - inst.D[j1][k] + inst.D[j2][k] > inst.C[i1][k] + 1e-7 ||
                        uso_abs[i2][k] - inst.D[j2][k] + inst.D[j1][k] > inst.C[i2][k] + 1e-7) {
                        cabe_troca = false;
                        break;
                    }
                }
                if (!cabe_troca) continue;

                double z_antigo = res.z_obtido;
                res.a[i1][j1] = 0; res.a[i1][j2] = 1;
                res.a[i2][j2] = 0; res.a[i2][j1] = 1;
                atualizarVariaveisContinuas(inst, res);

                if (res.z_obtido > z_antigo + 1e-7) {
                    srv_do_servico[j1] = i2;
                    srv_do_servico[j2] = i1;
                    for (int k = 0; k < R; ++k) {
                        uso_abs[i1][k] += (inst.D[j2][k] - inst.D[j1][k]);
                        uso_abs[i2][k] += (inst.D[j1][k] - inst.D[j2][k]);
                    }
                    melhorou = true;
                } else {
                    res.a[i1][j2] = 0; res.a[i1][j1] = 1;
                    res.a[i2][j1] = 0; res.a[i2][j2] = 1;
                    atualizarVariaveisContinuas(inst, res);
                }
            }
        }

        // Movimento C: Tenta inserir um servico ainda nao alocado apos abertura de espaco pelos Swaps
        for (int j = 0; j < y; ++j) {
            if (srv_do_servico[j] != -1) continue;
            for (int i = 0; i < x; ++i) {
                bool cabe = true;
                for (int k = 0; k < R; ++k) {
                    if (uso_abs[i][k] + inst.D[j][k] > inst.C[i][k] + 1e-7) {
                        cabe = false;
                        break;
                    }
                }
                if (cabe) {
                    res.a[i][j] = 1;
                    srv_do_servico[j] = i;
                    for (int k = 0; k < R; ++k) uso_abs[i][k] += inst.D[j][k];
                    atualizarVariaveisContinuas(inst, res);
                    melhorou = true;
                    break;
                }
            }
        }
    }

    auto t_fim = std::chrono::high_resolution_clock::now();
    res.tempo_ms = std::chrono::duration<double, std::milli>(t_fim - t_inicio).count();
    res.resolvido = true;
    res.status_texto = "Convergencia Barreira + Otimalidade Local 2-Opt";

    return res;
}

// ============================================================================
// METODO 2 (EXATO NO GLPK): SIMPLEX PRIMAL + FEASIBILITY PUMP + CORTES GOMORY
//                           E RAMIFICACAO DE DRIEBEEK-TOMLIN
// ============================================================================
ResultadoMetodo resolverViaGLPKFeasibilityPumpGomory(const InstanciaModelo1& inst) {
    ResultadoMetodo res;
    res.nome_metodo = "GLPK Intopt (Simplex Primal + Feasibility Pump + Cortes de Gomory/MIR)";
    res.resolvido = false;
    res.auditoria_aprovada = false;

    const int x = inst.x;
    const int y = inst.y;
    const int R = inst.R;

    auto col_a = [y](int i, int j) { return 1 + i * y + j; };
    auto col_u = [x, y, R](int i, int k) { return 1 + x * y + i * R + k; };
    const int col_Umax = 1 + x * y + x * R;
    auto col_s = [x, y, R](int i, int k) { return 1 + x * y + x * R + 1 + i * R + k; };

    glp_prob* lp = construirProblemaGLPK(inst);

    // Configura o Simplex Primal (em contraste ao Dual Simplex usado no HiGHS)
    glp_smcp parm_simplex;
    glp_init_smcp(&parm_simplex);
    parm_simplex.msg_lev = GLP_MSG_OFF;
    parm_simplex.meth = GLP_PRIMAL;

    // Configura o motor inteiro com Feasibility Pump, Cortes de Gomory/MIR e Driebeek-Tomlin
    glp_iocp parm_int;
    glp_init_iocp(&parm_int);
    parm_int.msg_lev = GLP_MSG_OFF;
    parm_int.presolve = GLP_ON;
    parm_int.fp_heur = GLP_ON;       // Ativa a matheuristica interna Feasibility Pump
    parm_int.gmi_cuts = GLP_ON;      // Planos de corte de Gomory (Gomory Mixed Integer)
    parm_int.mir_cuts = GLP_ON;      // Planos de corte Mixed Integer Rounding
    parm_int.cov_cuts = GLP_ON;      // Cortes de cobertura
    parm_int.br_tech = GLP_BR_DTH;   // Ramificacao pela heuristica de Driebeek-Tomlin
    parm_int.bt_tech = GLP_BT_BPH;   // Backtracking pela melhor projecao
    parm_int.tm_lim = 15000;         // Limite de 15 segundos

    auto t_inicio = std::chrono::high_resolution_clock::now();
    glp_simplex(lp, &parm_simplex);
    int ret = glp_intopt(lp, &parm_int);
    auto t_fim = std::chrono::high_resolution_clock::now();
    res.tempo_ms = std::chrono::duration<double, std::milli>(t_fim - t_inicio).count();

    int status = glp_mip_status(lp);
    if (status == GLP_OPT) res.status_texto = "Optimal (GLPK MIP)";
    else if (status == GLP_FEAS) res.status_texto = "Feasible (Time Limit / Sub-Optimal)";
    else {
        res.status_texto = "Falha / Inviavel";
        glp_delete_prob(lp);
        return res;
    }

    res.resolvido = true;
    res.z_obtido = glp_mip_obj_val(lp);

    res.a.assign(x, std::vector<int>(y, 0));
    res.u.assign(x, std::vector<double>(R, 0.0));
    res.s.assign(x, std::vector<double>(R, 0.0));

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            res.a[i][j] = static_cast<int>(std::round(glp_mip_col_val(lp, col_a(i, j))));
        }
        for (int k = 0; k < R; ++k) {
            res.u[i][k] = glp_mip_col_val(lp, col_u(i, k));
            res.s[i][k] = glp_mip_col_val(lp, col_s(i, k));
        }
    }
    res.U_max = glp_mip_col_val(lp, col_Umax);

    glp_delete_prob(lp);
    return res;
}

// Modulo Independente de Auditoria: testa do zero todas as restricoes (R1) a (R6)
void auditarSolucao(const InstanciaModelo1& inst, ResultadoMetodo& res) {
    if (!res.resolvido) return;

    const double tol = 1e-5;
    bool ok = true;

    res.servicos_incluidos = 0;
    double soma_p_alocados = 0.0;
    double soma_p_total = std::accumulate(inst.p.begin(), inst.p.end(), 0.0);

    for (int j = 0; j < inst.y; ++j) {
        int alocacoes_j = 0;
        for (int i = 0; i < inst.x; ++i) {
            if (res.a[i][j] != 0 && res.a[i][j] != 1) ok = false; // (R6)
            alocacoes_j += res.a[i][j];
        }
        if (alocacoes_j > 1) ok = false; // Violacao de (R1)
        if (alocacoes_j == 1) {
            res.servicos_incluidos++;
            soma_p_alocados += inst.p[j];
        }
    }
    res.taxa_inclusao_ponderada = soma_p_alocados / soma_p_total;

    double u_max_recalculado = 0.0;
    double soma_s_recalculado = 0.0;

    for (int i = 0; i < inst.x; ++i) {
        for (int k = 0; k < inst.R; ++k) {
            double consumo_real = 0.0;
            for (int j = 0; j < inst.y; ++j) {
                consumo_real += inst.D[j][k] * res.a[i][j];
            }
            if (consumo_real > inst.C[i][k] + tol) ok = false; // Violacao de (R2)

            double u_real = consumo_real / inst.C[i][k];
            if (std::abs(u_real - res.u[i][k]) > tol) ok = false; // Violacao de (R3)

            if (u_real > u_max_recalculado) u_max_recalculado = u_real;

            double s_real = std::max(0.0, u_real - inst.tau[k]);
            if (std::abs(s_real - res.s[i][k]) > tol) ok = false; // Violacao de (R5)
            soma_s_recalculado += s_real;
        }
    }

    if (std::abs(u_max_recalculado - res.U_max) > tol) ok = false; // Violacao de (R4)

    res.z_auditado = inst.w1 * res.taxa_inclusao_ponderada
                   - inst.w2 * u_max_recalculado
                   - inst.w3 * (soma_s_recalculado / static_cast<double>(inst.x * inst.R));

    if (std::abs(res.z_auditado - res.z_obtido) > tol) ok = false;

    res.auditoria_aprovada = ok;
}

// Exibicao detalhada comparando os dois metodos no solver GLPK
void exibirRelatorioMetodo(const InstanciaModelo1& inst, const ResultadoMetodo& res) {
    const std::vector<std::string> nomes_rec = {"CPU", "Rede", "RAM", "Disco"};

    std::cout << "  >>> METODO: " << res.nome_metodo << "\n";
    std::cout << "  Status           : " << res.status_texto << " (Tempo: " << std::fixed
              << std::setprecision(2) << res.tempo_ms << " ms)\n";
    std::cout << "  Servicos Alocados: " << res.servicos_incluidos << " de " << inst.y
              << " (Inclusao Ponderada: " << std::setprecision(2)
              << (res.taxa_inclusao_ponderada * 100.0) << "%)\n";
    std::cout << "  Gargalo Global   : U_max = " << std::setprecision(2) << (res.U_max * 100.0) << "%\n";
    std::cout << "  Objetivo Z Obtido: " << std::setprecision(6) << res.z_obtido
              << " | Z Auditado: " << res.z_auditado << "\n";
    std::cout << "  Auditoria R1-R6  : "
              << (res.auditoria_aprovada ? "[APROVADA - 100% CONSISTENTE]" : "[FALHOU]") << "\n";

    for (int i = 0; i < inst.x; ++i) {
        std::cout << "    Servidor " << (i + 1) << " -> { ";
        for (int j = 0; j < inst.y; ++j) {
            if (res.a[i][j] == 1) std::cout << "S" << (j + 1) << " ";
        }
        std::cout << "} | ";
        for (int k = 0; k < inst.R; ++k) {
            std::cout << nomes_rec[k] << ": " << std::setprecision(1) << (res.u[i][k] * 100.0) << "%";
            if (res.s[i][k] > 1e-4) {
                std::cout << "(+" << (res.s[i][k] * 100.0) << "% SLA)";
            }
            if (k < inst.R - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "--------------------------------------------------------------------------------\n";
}

int main() {
    // Usa a mesma semente base demonstrada no teste anterior para permitir comparacao direta
    unsigned int semente_base = 289410521;

    std::cout << "Iniciando simulacao do Modelo 1 no Solver GLPK (Semente Base: " << semente_base << ")\n\n";

    InstanciaModelo1 caso1 = gerarCasoDeTesteRandomico(
        "Caso Aleatorio 1 - Carga Confortavel (Balanceamento Puro Min-Max)",
        4, 14, 0.45, semente_base + 1);

    InstanciaModelo1 caso2 = gerarCasoDeTesteRandomico(
        "Caso Aleatorio 2 - Carga Elevada (Transicao de Limiar SLA tau_k)",
        5, 22, 0.85, semente_base + 2);

    InstanciaModelo1 caso3 = gerarCasoDeTesteRandomico(
        "Caso Aleatorio 3 - Saturacao Extrema (Controle de Admissao por Prioridade)",
        4, 30, 1.25, semente_base + 3);

    std::vector<InstanciaModelo1> bateria_testes = {caso1, caso2, caso3};

    for (const auto& inst : bateria_testes) {
        std::cout << "================================================================================\n";
        std::cout << "CENARIO: " << inst.nome_cenario << "\n";
        std::cout << "Servidores (x): " << inst.x << " | Servicos (y): " << inst.y
                  << " | Pesos: w1=" << std::fixed << std::setprecision(4) << inst.w1
                  << ", w2=" << inst.w2 << ", w3=" << inst.w3 << "\n";
        std::cout << "--------------------------------------------------------------------------------\n";

        // Executa Metodo Alternativo 1: Pontos Interiores + Matheuristica Relax-and-Fix & Busca Local
        ResultadoMetodo res_mat = resolverViaPontosInterioresMatheuristica(inst);
        auditarSolucao(inst, res_mat);
        exibirRelatorioMetodo(inst, res_mat);

        // Executa Metodo Alternativo 2: GLPK Exato (Simplex Primal + Feasibility Pump + Gomory)
        ResultadoMetodo res_exato = resolverViaGLPKFeasibilityPumpGomory(inst);
        auditarSolucao(inst, res_exato);
        exibirRelatorioMetodo(inst, res_exato);

        double gap_relativo = std::abs(res_exato.z_obtido - res_mat.z_obtido) / res_exato.z_obtido * 100.0;
        std::cout << "  Comparativo Direto -> Gap da Matheuristica Barreira vs. Exato GLPK: "
                  << std::setprecision(4) << gap_relativo << "%\n";
        std::cout << "================================================================================\n\n";
    }

    return 0;
}