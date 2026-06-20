import os

def parse_filters(file_path):
    """Lê os filtros (down, strategy, ncall) do segundo txt"""
    filters = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('down'):
                continue
            down, strategy, ncall = [x.strip() for x in line.split('|')]
            filters.add((float(down), int(strategy), int(ncall)))
    return filters


def filtrar_resultados(arquivo_completo, arquivo_filtros, arquivo_saida):
    filtros = parse_filters(arquivo_filtros)

    with open(arquivo_completo, 'r') as f:
        linhas = f.readlines()

    for i, linha in enumerate(linhas):
        if linha.strip().startswith('==='):
            cabecalho = linhas[:i+1]
            dados = linhas[i+1:]
            break
    else:
        print(f"Aviso: não encontrou separador '===' em {arquivo_completo}")
        return

    linhas_filtradas = []
    for linha in dados:
        partes = [p.strip() for p in linha.strip().split('|')]
        if len(partes) < 4:
            continue
        try:
            down = float(partes[0])
            strategy = int(partes[2])
            ncall = int(partes[3])
            if (down, strategy, ncall) in filtros:
                linhas_filtradas.append(linha)
        except ValueError:
            continue

    if linhas_filtradas:
        with open(arquivo_saida, 'w') as f_out:
            f_out.writelines(cabecalho)
            f_out.writelines(linhas_filtradas)
        print(f"✔ Resultado salvo: {arquivo_saida}")
    else:
        print(f"⚠ Nenhuma linha correspondente em {arquivo_completo} com {arquivo_filtros}")


# Diretórios
dir_completo = '/home/victorli/personal/Two-Gluon-Exchange/results/all_possible_iterations/pl_totem'
dir_filtros = '/home/victorli/personal/Two-Gluon-Exchange/results/all_possible_iterations/comum_configs/pl_atlas_pl_totem'
dir_saida = '/home/victorli/personal/Two-Gluon-Exchange/results/all_possible_iterations/filter'  # ou outro diretório, se preferir

# Mapeia arquivos de filtro pelo nome-base
filtros_dict = {
    os.path.basename(f): os.path.join(dir_filtros, f)
    for f in os.listdir(dir_filtros)
    if f.endswith('.txt')
}

# Itera sobre os arquivos completos
for nome_arquivo_completo in os.listdir(dir_completo):
    if not nome_arquivo_completo.endswith('.txt'):
        continue
    if not nome_arquivo_completo.startswith('resultados_otimizacao_pl_totem_'):
        continue

    nome_base = nome_arquivo_completo.replace('resultados_otimizacao_pl_totem_', '')
    caminho_completo = os.path.join(dir_completo, nome_arquivo_completo)

    if nome_base not in filtros_dict:
        print(f"⚠ Filtro não encontrado para {nome_arquivo_completo}")
        continue

    caminho_filtro = filtros_dict[nome_base]
    nome_saida = f"resultados_filtrados_pl_totem_{nome_base}"
    caminho_saida = os.path.join(dir_saida, nome_saida)

    filtrar_resultados(caminho_completo, caminho_filtro, caminho_saida)
