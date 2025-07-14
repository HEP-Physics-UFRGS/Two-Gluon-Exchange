def parse_data_line(line):
    """Extrai as colunas de interesse de uma linha de dados."""
    parts = [p.strip() for p in line.strip().split('|')]
    if len(parts) < 4:
        return None  # Ignora linhas malformadas
    try:
        down = float(parts[0])
        strategy = int(float(parts[1]))  # tratando casos como 1.00
        ncall = int(float(parts[2]))
        return (down, strategy, ncall)
    except ValueError:
        return None

def read_file(filepath):
    """Lê o arquivo e retorna um set de tuplas (down, strategy, ncall)."""
    data_set = set()
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().lower().startswith("down"):
                continue  # pula o cabeçalho
            parsed = parse_data_line(line)
            if parsed:
                data_set.add(parsed)
    return data_set

def write_output(common_lines, output_path):
    """Escreve os dados comuns no arquivo de saída."""
    with open(output_path, 'w') as f:
        f.write("down | strategy | ncall\n")
        for down, strategy, ncall in sorted(common_lines):
            f.write(f"{down:.2f} | {strategy} | {ncall}\n")

def main():
    eps = 0
    mg = 1
    a1 = 1
    a2 = 1
    arquivo1 = f'results/iteration_over_all_log/sigma_tot/output_resultados_otimizacao_log_atlas_eps_{eps}_mg_{mg}_a2_{a2}.txt'
    arquivo2 = f'results/iteration_over_all_pl/sigma_tot/output_resultados_otimizacao_pl_atlas_eps_{eps}_mg_{mg}_a1_{a1}.txt'
    saida = f'results/testeAAA.txt'

    dados1 = read_file(arquivo1)
    dados2 = read_file(arquivo2)

    comuns = dados1.intersection(dados2)

    write_output(comuns, saida)
    print(f"Arquivo de saída criado com {len(comuns)} linhas: {saida}")

if __name__ == "__main__":
    main()
