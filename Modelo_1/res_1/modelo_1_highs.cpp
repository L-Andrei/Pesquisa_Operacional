#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <Highs.h>

// Estrutura que armazena uma instancia gerada para o Modelo 1
struct InstanciaModelo1 {
    std::string nome_cenario;
    int x; // Numero de servidores fisicos (Conjunto I)
    int y; // Numero de servicos candidatos (Conjunto J)
    int R; // Numero de recursos por servidor (Conjunto K, R = 4)

    std::vector<std::vector<double>> C; // C[i][k]: Capacidade do recurso k no servidor i
    std::vector<std::vector<double>> D; // D[j][k]: Demanda do recurso k pelo servico j
    std::vector<double> p;              // p[j]: Prioridade de inclusao do servico j
    std::vector<double> tau;            // tau[k]: Limiar seguro de SLA/QoS do recurso k

    double w1, w2, w3; // Pesos de escalarizacao multiobjetivo
};

// Estrutura que armazena a solucao extraida do solver e metricas de auditoria
struct ResultadoModelo1 {
    bool resolvido;
    std::string status_texto;
    double tempo_ms;
    double z_solver;
    double z_auditado;

    std::vector<std::vector<int>> a;    // a[i][j] em {0, 1}
    std::vector<std::vector<double>> u; // u[i][k] em [0, 1]
    std::vector<std::vector<double>> s; // s[i][k] >= 0
    double U_max;

    int servicos_incluidos;
    double taxa_inclusao_ponderada;
    bool auditoria_aprovada;
};

// Gerador randomico de casos de teste com controle de intensidade de carga
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
    inst.R = 4; // {0: CPU (vCores), 1: Rede (Gbps), 2: RAM (GB), 3: Armazenamento (GB)}

    std::mt19937 rng(semente);

    // Perfis tipicos de capacidade de servidores (heterogeneos sorteados)
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

    // Geracao randomica das demandas D[j][k] proporcional ao fator de carga do cenario
    inst.D.assign(y, std::vector<double>(inst.R, 0.0));
    for (int j = 0; j < y; ++j) {
        for (int k = 0; k < inst.R; ++k) {
            double media_demanda = (soma_cap_recurso[k] * fator_escala_demanda) / static_cast<double>(y);
            std::uniform_real_distribution<double> dist_dem(0.35 * media_demanda, 1.65 * media_demanda);
            inst.D[j][k] = std::max(1.0, std::round(dist_dem(rng) * 10.0) / 10.0);
        }
    }

    // Prioridades p[j] sorteadas inteiras entre 1 e 10
    inst.p.assign(y, 1.0);
    std::uniform_int_distribution<int> dist_prio(1, 10);
    double soma_p = 0.0;
    double min_p = 1e9;
    for (int j = 0; j < y; ++j) {
        inst.p[j] = static_cast<double>(dist_prio(rng));
        soma_p += inst.p[j];
        if (inst.p[j] < min_p) min_p = inst.p[j];
    }

    // Limiares seguros tau[k] (75% para CPU e Rede, 80% para RAM e Armazenamento)
    inst.tau = {0.75, 0.75, 0.80, 0.80};

    // Calibracao lexicografica da Secao 7: w1 = 1.0 e w2 + w3 < min(p_j) / sum(p_j)
    double limite_seguro_pesos = min_p / soma_p;
    inst.w1 = 1.0;
    inst.w2 = 0.60 * limite_seguro_pesos;
    inst.w3 = 0.30 * limite_seguro_pesos;

    return inst;
}

// Funcao que monta e resolve o Modelo 1 via solver HiGHS
ResultadoModelo1 resolverModelo1HiGHS(const InstanciaModelo1& inst) {
    ResultadoModelo1 res;
    res.resolvido = false;
    res.auditoria_aprovada = false;

    const int x = inst.x;
    const int y = inst.y;
    const int R = inst.R;

    // Funcoes auxiliares de mapeamento de indices de colunas no solver
    auto idx_a = [y](int i, int j) { return i * y + j; };
    auto idx_u = [x, y, R](int i, int k) { return x * y + i * R + k; };
    const int idx_Umax = x * y + x * R;
    auto idx_s = [x, y, R](int i, int k) { return x * y + x * R + 1 + i * R + k; };

    const int total_variaveis = x * y + 2 * (x * R) + 1;

    Highs highs;
    highs.setOptionValue("output_flag", false); // Silencia o log interno do solver
    highs.setOptionValue("time_limit", 30.0);   // Limite de 30s por instancia

    double soma_p = std::accumulate(inst.p.begin(), inst.p.end(), 0.0);
    const double inf = kHighsInf;

    // 1. Criacao das variaveis (Colunas) e definicao de limites (Restricao R6)
    // Bloco a[i][j] em {0, 1}
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            int col = idx_a(i, j);
            highs.addVar(0.0, 1.0);
            highs.changeColIntegrality(col, HighsVarType::kInteger);
            double coef_obj = inst.w1 * (inst.p[j] / soma_p);
            highs.changeColCost(col, coef_obj);
        }
    }

    // Bloco u[i][k] em [0, 1]
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            int col = idx_u(i, k);
            highs.addVar(0.0, 1.0);
            highs.changeColCost(col, 0.0);
        }
    }

    // Variavel escalar U_max em [0, 1]
    highs.addVar(0.0, 1.0);
    highs.changeColCost(idx_Umax, -inst.w2);

    // Bloco s[i][k] >= 0
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            int col = idx_s(i, k);
            highs.addVar(0.0, inf);
            double coef_obj = -inst.w3 / static_cast<double>(x * R);
            highs.changeColCost(col, coef_obj);
        }
    }

    // Configura sentido para Maximizacao
    highs.changeObjectiveSense(ObjSense::kMaximize);

    // 2. Adicao das Restricoes Estruturais (R1 a R5)
    // (R1) Unicidade de Alocacao: sum_{i in I} a[i][j] <= 1 para cada j in J
    for (int j = 0; j < y; ++j) {
        std::vector<HighsInt> indices;
        std::vector<double> valores;
        for (int i = 0; i < x; ++i) {
            indices.push_back(idx_a(i, j));
            valores.push_back(1.0);
        }
        highs.addRow(-inf, 1.0, static_cast<HighsInt>(indices.size()), indices.data(), valores.data());
    }

    // (R2) Capacidade Multidimensional: sum_{j in J} D[j][k] * a[i][j] <= C[i][k]
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            std::vector<HighsInt> indices;
            std::vector<double> valores;
            for (int j = 0; j < y; ++j) {
                indices.push_back(idx_a(i, j));
                valores.push_back(inst.D[j][k]);
            }
            highs.addRow(-inf, inst.C[i][k], static_cast<HighsInt>(indices.size()), indices.data(), valores.data());
        }
    }

    // (R3) Taxa de Utilizacao Normalizada: sum_{j in J} (D[j][k]/C[i][k])*a[i][j] - u[i][k] == 0
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            std::vector<HighsInt> indices;
            std::vector<double> valores;
            for (int j = 0; j < y; ++j) {
                indices.push_back(idx_a(i, j));
                valores.push_back(inst.D[j][k] / inst.C[i][k]);
            }
            indices.push_back(idx_u(i, k));
            valores.push_back(-1.0);
            highs.addRow(0.0, 0.0, static_cast<HighsInt>(indices.size()), indices.data(), valores.data());
        }
    }

    // (R4) Linearizacao do Gargalo Maximo: U_max - u[i][k] >= 0
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            std::vector<HighsInt> indices = {idx_Umax, idx_u(i, k)};
            std::vector<double> valores = {1.0, -1.0};
            highs.addRow(0.0, inf, 2, indices.data(), valores.data());
        }
    }

    // (R5) Excesso Acima do Limiar de Performance: s[i][k] - u[i][k] >= -tau[k]
    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < R; ++k) {
            std::vector<HighsInt> indices = {idx_s(i, k), idx_u(i, k)};
            std::vector<double> valores = {1.0, -1.0};
            highs.addRow(-inst.tau[k], inf, 2, indices.data(), valores.data());
        }
    }

    // 3. Execucao do Solver e Medicao de Tempo
    auto t_inicio = std::chrono::high_resolution_clock::now();
    HighsStatus status_exec = highs.run();
    auto t_fim = std::chrono::high_resolution_clock::now();
    res.tempo_ms = std::chrono::duration<double, std::milli>(t_fim - t_inicio).count();

    HighsModelStatus status_modelo = highs.getModelStatus();
    res.status_texto = highs.modelStatusToString(status_modelo);

    if (status_exec != HighsStatus::kOk ||
        (status_modelo != HighsModelStatus::kOptimal && status_modelo != HighsModelStatus::kTimeLimit)) {
        return res;
    }

    res.resolvido = true;
    res.z_solver = highs.getInfo().objective_function_value;
    const std::vector<double>& col_val = highs.getSolution().col_value;

    // 4. Extracao dos Valores das Variaveis de Decisao
    res.a.assign(x, std::vector<int>(y, 0));
    res.u.assign(x, std::vector<double>(R, 0.0));
    res.s.assign(x, std::vector<double>(R, 0.0));

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            res.a[i][j] = static_cast<int>(std::round(col_val[idx_a(i, j)]));
        }
        for (int k = 0; k < R; ++k) {
            res.u[i][k] = col_val[idx_u(i, k)];
            res.s[i][k] = col_val[idx_s(i, k)];
        }
    }
    res.U_max = col_val[idx_Umax];

    return res;
}

// Modulo Independente de Auditoria: verifica todas as restricoes do zero
void auditarSolucao(const InstanciaModelo1& inst, ResultadoModelo1& res) {
    if (!res.resolvido) return;

    const double tol = 1e-5;
    bool ok = true;

    // Verifica (R1) e contabiliza inclusao
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

    // Verifica (R2), (R3), (R4) e (R5) recalculando os consumos reais
    double u_max_recalculado = 0.0;
    double soma_s_recalculado = 0.0;

    for (int i = 0; i < inst.x; ++i) {
        for (int k = 0; k < inst.R; ++k) {
            double consumo_real = 0.0;
            for (int j = 0; j < inst.y; ++j) {
                consumo_real += inst.D[j][k] * res.a[i][j];
            }
            // Verifica (R2): Consumo <= Capacidade
            if (consumo_real > inst.C[i][k] + tol) ok = false;

            // Verifica (R3): Taxa u[i][k]
            double u_real = consumo_real / inst.C[i][k];
            if (std::abs(u_real - res.u[i][k]) > tol) ok = false;

            // Atualiza U_max real (R4)
            if (u_real > u_max_recalculado) u_max_recalculado = u_real;

            // Verifica (R5): Excesso s[i][k] acima de tau[k]
            double s_real = std::max(0.0, u_real - inst.tau[k]);
            if (std::abs(s_real - res.s[i][k]) > tol) ok = false;
            soma_s_recalculado += s_real;
        }
    }

    if (std::abs(u_max_recalculado - res.U_max) > tol) ok = false;

    // Recalcula a Funcao Objetivo Z a partir das variaveis auditadas
    res.z_auditado = inst.w1 * res.taxa_inclusao_ponderada
                   - inst.w2 * u_max_recalculado
                   - inst.w3 * (soma_s_recalculado / static_cast<double>(inst.x * inst.R));

    if (std::abs(res.z_auditado - res.z_solver) > tol) ok = false;

    res.auditoria_aprovada = ok;
}

// Impressao formatada do relatorio de cada caso de teste
void exibirRelatorio(const InstanciaModelo1& inst, const ResultadoModelo1& res) {
    const std::vector<std::string> nomes_rec = {"CPU", "Rede", "RAM", "Disco"};

    std::cout << "================================================================================\n";
    std::cout << "CENARIO: " << inst.nome_cenario << "\n";
    std::cout << "Servidores (x): " << inst.x << " | Servicos (y): " << inst.y
              << " | Recursos (R): " << inst.R << "\n";
    std::cout << "Pesos Calibrados: w1 = " << std::fixed << std::setprecision(4) << inst.w1
              << ", w2 = " << inst.w2 << ", w3 = " << inst.w3 << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Status do Solver : " << res.status_texto << " (Tempo: " << std::setprecision(2)
              << res.tempo_ms << " ms)\n";
    std::cout << "Servicos Alocados: " << res.servicos_incluidos << " de " << inst.y
              << " (Inclusao Ponderada: " << std::setprecision(2)
              << (res.taxa_inclusao_ponderada * 100.0) << "%)\n";
    std::cout << "Gargalo Global   : U_max = " << std::setprecision(2) << (res.U_max * 100.0) << "%\n";
    std::cout << "Objetivo Z Solver: " << std::setprecision(6) << res.z_solver
              << " | Z Auditado: " << res.z_auditado << "\n";
    std::cout << "Auditoria R1-R6  : " << (res.auditoria_aprovada ? "[APROVADA - 100% CONSISTENTE]" : "[FALHOU]") << "\n\n";

    for (int i = 0; i < inst.x; ++i) {
        std::cout << "  Servidor " << (i + 1) << " -> Servicos alocados: { ";
        for (int j = 0; j < inst.y; ++j) {
            if (res.a[i][j] == 1) std::cout << "S" << (j + 1) << " ";
        }
        std::cout << "}\n    Utilizacao [u_ik (s_ik)]: ";
        for (int k = 0; k < inst.R; ++k) {
            std::cout << nomes_rec[k] << ": " << std::setprecision(1) << (res.u[i][k] * 100.0) << "%";
            if (res.s[i][k] > 1e-4) {
                std::cout << " (+" << (res.s[i][k] * 100.0) << "% SLA)";
            }
            if (k < inst.R - 1) std::cout << " | ";
        }
        std::cout << "\n";
    }
    std::cout << "================================================================================\n\n";
}

int main() {
    std::random_device rd;
    unsigned int semente_base = rd();

    std::cout << "Iniciando simulacao estocastica do Modelo 1 (Semente Base: " << semente_base << ")\n\n";

    // Caso 1: Carga Leve (~45% da capacidade total) - Deve incluir 100% sem violar SLA (s_ik = 0)
    InstanciaModelo1 caso1 = gerarCasoDeTesteRandomico(
        "Caso Aleatorio 1 - Carga Confortavel (Balanceamento Puro Min-Max)",
        4, 14, 0.45, semente_base + 1);

    // Caso 2: Carga Alta (~85% da capacidade total) - Deve incluir 100% gerenciando excesso s_ik
    InstanciaModelo1 caso2 = gerarCasoDeTesteRandomico(
        "Caso Aleatorio 2 - Carga Elevada (Transicao de Limiar SLA tau_k)",
        5, 22, 0.85, semente_base + 2);

    // Caso 3: Sobrecarga (~125% da capacidade total) - Atua como Controlador de Admissao (R1 <= 1)
    InstanciaModelo1 caso3 = gerarCasoDeTesteRandomico(
        "Caso Aleatorio 3 - Saturacao Extrema (Controle de Admissao por Prioridade)",
        4, 30, 1.25, semente_base + 3);

    std::vector<InstanciaModelo1> bateria_testes = {caso1, caso2, caso3};

    for (const auto& inst : bateria_testes) {
        ResultadoModelo1 res = resolverModelo1HiGHS(inst);
        auditarSolucao(inst, res);
        exibirRelatorio(inst, res);
    }

    return 0;
}

//