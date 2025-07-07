def parse_data_line(line):
    """Extrai as colunas de interesse de uma linha de dados."""
    parts = [p.strip() for p in line.strip().split('|')]
    if len(parts) < 4:
        return None  # Ignora linhas malformadas
    try:
        down = float(parts[0])
        strategy = int(parts[2])
        migrad = int(parts[3])
        return (down, strategy, migrad)
    except ValueError:
        return None

def read_file(filepath):
    """Lê o arquivo e retorna um set de tuplas (down, strategy, migrad)."""
    common_data = set()
    with open(filepath, 'r') as f:
        for line in f:
            parsed = parse_data_line(line)
            if parsed:
                common_data.add(parsed)
    return common_data

def write_output(common_lines, output_path):
    """Escreve os dados comuns no arquivo de saída."""
    with open(output_path, 'w') as f:
        f.write("down | strategy | migrad\n")
        for down, strategy, migrad in sorted(common_lines):
            f.write(f"{down} | {strategy} | {migrad}\n")

def main():
    arquivo1 = '/home/victor/personal/Two-Gluon-Exchange/results/iteration_over_all/resultados_otimizacao_log_atlas_eps_0_mg_1_a2_1.txt'
    arquivo2 = '/home/victor/personal/Two-Gluon-Exchange/results/iteration_over_all/resultados_otimizacao_log_totem_eps_0_mg_1_a2_1.txt'
    saida = 'comum_down_strategy_migrad_0_1_1.txt'

    dados1 = read_file(arquivo1)
    dados2 = read_file(arquivo2)

    comuns = dados1.intersection(dados2)

    write_output(comuns, saida)
    print(f"Arquivo de saída criado com {len(comuns)} linhas: {saida}")

if __name__ == "__main__":
    main()
