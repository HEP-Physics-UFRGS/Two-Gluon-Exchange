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
    eps = 0
    mg = 1
    a1 = 1
    arquivo1 = f'results/iteration_over_all_pl/resultados_otimizacao_pl_atlas_eps_{eps}_mg_{mg}_a1_{a1}.txt'
    arquivo2 = f'results/iteration_over_all_pl/resultados_otimizacao_pl_totem_eps_{eps}_mg_{mg}_a1_{a1}.txt'
    saida = f'results/comum_down_strategy_migrad_eps_{eps}_mg_{mg}_a1_{a1}.txt'

    dados1 = read_file(arquivo1)
    dados2 = read_file(arquivo2)

    comuns = dados1.intersection(dados2)

    write_output(comuns, saida)
    print(f"Arquivo de saída criado com {len(comuns)} linhas: {saida}")

if __name__ == "__main__":
    main()
