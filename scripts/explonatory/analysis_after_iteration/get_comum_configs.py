import os
from itertools import product

def parse_data_line(line):
    parts = [p.strip() for p in line.strip().split('|')]
    if len(parts) < 3:
        print(f"[WARN] Linha ignorada por formato inválido: {line.strip()}")
        return None
    try:
        down = round(float(parts[0]), 3)
        strategy = int(float(parts[1]))
        ncall = int(float(parts[2]))
        return (down, strategy, ncall)
    except ValueError as e:
        print(f"[WARN] Erro ao converter valores na linha: {line.strip()} -> {e}")
        return None

def read_file(filepath):
    data_set = set()
    if not os.path.exists(filepath):
        print(f"[ERRO] Arquivo não encontrado: {filepath}")
        return data_set
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().lower().startswith("down"):
                    continue
                parsed = parse_data_line(line)
                if parsed:
                    data_set.add(parsed)
        print(f"[INFO] {len(data_set)} entradas válidas lidas de {filepath}")
    except Exception as e:
        print(f"[ERRO] Falha ao ler o arquivo {filepath}: {e}")
    return data_set

def write_output(common_lines, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("down | strategy | ncall\n")
            for down, strategy, ncall in sorted(common_lines):
                f.write(f"{down:.3f} | {strategy} | {ncall}\n")
        print(f"[SUCESSO] Arquivo de saída criado com {len(common_lines)} linhas: {output_path}")
    except Exception as e:
        print(f"[ERRO] Falha ao escrever no arquivo {output_path}: {e}")

def main():
    base_dir_atlas = "/home/victorli/personal/Two-Gluon-Exchange/results/all_possible_iterations/pl_atlas/sigma_tot/"
    base_dir_totem = "/home/victorli/personal/Two-Gluon-Exchange/results/all_possible_iterations/pl_totem/sigma_tot/"
    output_base = "results/all_possible_iterations/comum_configs/"

    valores_binarios = [0, 1]

    for eps, mg, a1, a2 in product(valores_binarios, repeat=4):
        nome = f"eps_{eps}_mg_{mg}_a1_{a1}_a2_{a2}"
        arq_atlas = os.path.join(base_dir_atlas, f"output_resultados_otimizacao_pl_atlas_{nome}.txt")
        arq_totem = os.path.join(base_dir_totem, f"output_resultados_otimizacao_pl_totem_{nome}.txt")
        arq_saida = os.path.join(output_base, f"{nome}.txt")

        print(f"\n[INFO] Processando: {nome}")

        dados1 = read_file(arq_atlas)
        dados2 = read_file(arq_totem)

        if not dados1 or not dados2:
            print(f"[INFO] Pulando {nome} por falta de dados válidos.")
            continue

        comuns = dados1.intersection(dados2)
        print(f"[INFO] {len(comuns)} configurações comuns encontradas.")

        if comuns:
            write_output(comuns, arq_saida)
        else:
            print("[INFO] Nenhuma configuração comum encontrada. Nenhum arquivo foi gerado.")

if __name__ == "__main__":
    main()
