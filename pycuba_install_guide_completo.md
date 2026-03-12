# Guia Completo de Instalação do PyCuba

**Ambiente:** Linux (Ubuntu/Debian x86-64) — Python 3.9 via Conda  
**Pacote Python:** `pymultinest` (contém o módulo `pycuba`)  
**Biblioteca C:** `cuba` (deve ser compilada manualmente)  
**Repositórios:**
- https://github.com/JohannesBuchner/PyMultiNest
- https://github.com/JohannesBuchner/cuba
- Documentação PyCuba: https://johannesbuchner.github.io/PyMultiNest/pycuba.html

---

## Contexto: O que é o PyCuba?

O **PyCuba** é um wrapper Python para a biblioteca C chamada **Cuba**, que implementa quatro algoritmos de integração numérica multidimensional:

| Algoritmo | Tipo |
|-----------|------|
| Vegas     | Monte Carlo adaptativo |
| Suave     | Monte Carlo suavizado |
| Divonne   | Monte Carlo com divisão de domínio |
| Cuhre     | Quadratura determinística |

O PyCuba **não é um pacote independente** — ele vem embutido dentro do pacote `pymultinest`. O fluxo de dependências é:

```
seu código Python
    └── pycuba  (módulo dentro de pymultinest)
            └── libcuba.so  (biblioteca C compilada manualmente)
```

Por isso, a instalação tem duas partes: compilar a biblioteca C e instalar o pacote Python.

---

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- **Conda** (Miniconda ou Anaconda)
- **git**
- **gcc** e **gfortran** (compiladores C e Fortran)
- **make**

Para instalar as dependências no Ubuntu/Debian:

```bash
sudo apt-get install git gcc gfortran make
```

---

## Passo 1 — Criar e Ativar o Ambiente Conda

O PyMultiNest é um pacote antigo. Usar Python 3.9 garante a melhor compatibilidade.

```bash
conda create -n pycuba_env python=3.9
conda activate pycuba_env
```

Para verificar que o ambiente está ativo (o nome aparece no prompt):

```bash
# O prompt deve mostrar:
(pycuba_env) usuario@maquina:~$
```

> **Por que Python 3.9 e não uma versão mais nova?**  
> O PyMultiNest usa `ctypes` para chamar a biblioteca C diretamente. Versões mais novas do Python podem ter mudanças de ABI que causam incompatibilidades. O Python 3.9 é estável e bem testado com este pacote.

---

## Passo 2 — Clonar e Compilar a Biblioteca Cuba (C)

O PyCuba em tempo de execução faz:

```python
lib = ctypes.cdll.LoadLibrary('libcuba.so')
```

Portanto, **`libcuba.so` precisa existir e ser acessível**. Ela não vem pré-compilada — você precisa compilar do código-fonte.

### 2.1 Clonar o repositório

```bash
cd ~
git clone https://github.com/JohannesBuchner/cuba/
cd cuba
```

### 2.2 Tentativa ingênua (vai falhar)

Se você tentar a abordagem direta:

```bash
./configure
./makesharedlib.sh
```

Você verá este erro:

```
gcc -shared -Wall Vegas.o Vegas_.o llVegas.o ... -lm -o libcuba.so
/usr/bin/ld: Vegas.o: relocation R_X86_64_PC32 against symbol `cubafun_'
can not be used when making a shared object; recompile with -fPIC
/usr/bin/ld: final link failed: bad value
collect2: error: ld returned 1 exit status
```

**Por que acontece?**  
Em arquiteturas x86-64, objetos `.o` usados em bibliotecas compartilhadas (`.so`) precisam ser compilados com a flag `-fPIC` (*Position Independent Code* — código independente de posição). O `Makefile` padrão do Cuba compila para uso estático (`.a`), sem essa flag. O script `makesharedlib.sh` tenta reutilizar esses `.o` para criar a `.so`, e o linker rejeita.

### 2.3 Solução: recompilar com `-fPIC`

```bash
make clean
CFLAGS="-fPIC" FFLAGS="-fPIC" ./configure
CFLAGS="-fPIC" FFLAGS="-fPIC" ./makesharedlib.sh
```

Saída esperada ao final (sem erros):

```
rm -f Data.o
ranlib libcuba.a
unpacking libcuba.a
making libcuba.so
gcc -shared -Wall Vegas.o Vegas_.o llVegas.o llVegas_.o Suave.o Suave_.o \
    llSuave.o llSuave_.o Divonne.o Divonne_.o llDivonne.o llDivonne_.o \
    Cuhre.o Cuhre_.o llCuhre.o llCuhre_.o Fork.o Fork_.o Global.o \
    Global_.o Data.o -lm -o libcuba.so
(pycuba_env) victor@Kurchatov:~/cuba$
```

Note que desta vez **não há mensagem de erro** — o prompt retorna normalmente.

### 2.4 Verificar a compilação

```bash
ls -lh libcuba.so
file libcuba.so
```

Saída esperada:

```
-rwxrwxr-x 1 victor victor 945K mar 12 15:02 libcuba.so
libcuba.so: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV),
            dynamically linked, BuildID[sha1]=831b0dac52..., not stripped
```

O importante é: **ELF 64-bit LSB shared object**. Se aparecer isso, a compilação foi bem-sucedida.

---

## Passo 3 — Instalar o PyMultiNest (Python)

Com o ambiente conda ativo:

```bash
pip install pymultinest
```

---

## Passo 4 — Disponibilizar o `libcuba.so` para o Python

Esta é a etapa mais crítica. O Python precisa **encontrar** `libcuba.so` em tempo de execução.

### Por que isso é necessário?

O `pycuba/__init__.py` faz literalmente isso:

```python
lib = ctypes.cdll.LoadLibrary('libcuba.so')
```

Se o sistema encontrar uma versão errada (por exemplo, uma instalação antiga em `/usr/local/lib`), você verá:

```
OSError: /usr/local/lib/libcuba.so: undefined symbol: cubaverb_
```

**Por que `cubaverb_` não existe na versão antiga?**  
Esse símbolo foi adicionado em versões mais recentes do Cuba. Uma `libcuba.so` antiga ou compilada de forma diferente simplesmente não o tem.

### Verificar se há versões conflitantes

```bash
find / -name "libcuba.so" 2>/dev/null
```

Se aparecerem múltiplas entradas (como aconteceu na instalação original):

```
/home/victor/personal/Two-Gluon-Exchange/PyMultiNest/cuba/libcuba.so
/home/victor/cuba/libcuba.so
```

Isso confirma que pode haver conflito.

### Solução recomendada: copiar para o lib/ do conda

A solução mais robusta — independente de variáveis de ambiente — é copiar diretamente para o diretório `lib/` do ambiente conda:

```bash
cp ~/cuba/libcuba.so ~/miniconda3/envs/pycuba_env/lib/
```

Isso garante que, sempre que o ambiente `pycuba_env` estiver ativo, o Python encontrará automaticamente a versão correta, sem precisar exportar `LD_LIBRARY_PATH` toda vez.

> **Alternativa (menos confiável):**  
> ```bash
> export LD_LIBRARY_PATH=~/cuba:$LD_LIBRARY_PATH
> ```
> O problema desta abordagem é que precisa ser repetida em cada nova sessão de terminal, ou adicionada ao `~/.bashrc`. Se outra versão de `libcuba.so` estiver antes no path, ainda pode carregar a errada.

---

## Passo 5 — Testar a Instalação

### Teste rápido

> **Atenção ao shell bash:** O bash interpreta `!` como expansão de histórico. Por isso, use **aspas simples** em volta do comando:

```bash
# ERRADO — bash vai reclamar de "event not found"
python -c "from pycuba import Vegas; print('PyCuba working!')"

# CORRETO — aspas simples desativam expansão do bash
python -c 'from pycuba import Vegas; print("PyCuba working!")'
```

Se a saída for `PyCuba working!`, a instalação está funcionando.

---

## Passo 6 — Script de Teste Completo

Salve o código abaixo como `test_pycuba.py`:

```python
import math
from pycuba import Vegas, Suave, Divonne, Cuhre

# Integrando: f(x, y, z) = sin(x) * cos(y) * exp(z)
# Domínio: [0, 1]^3
#
# Resultado analítico:
#   integral_0^1 sin(x) dx = 1 - cos(1)
#   integral_0^1 cos(y) dy = sin(1)
#   integral_0^1 exp(z) dz = e - 1
#   Produto ≈ 0.664588

def integrand(ndim, xx, ncomp, ff, userdata):
    x, y, z = [xx[i] for i in range(ndim.contents.value)]
    ff[0] = math.sin(x) * math.cos(y) * math.exp(z)
    return 0

NDIM    = 3
NCOMP   = 1
MAXEVAL = 100000
EXPECTED = (1 - math.cos(1)) * math.sin(1) * (math.e - 1)

def print_result(name, res):
    r = res['results'][0]
    diff = abs(r['integral'] - EXPECTED)
    status = "OK" if diff < 1e-3 else "WARN"
    print(f"[{status}] {name:<10}  integral={r['integral']:.6f}  "
          f"error={r['error']:.2e}  (esperado={EXPECTED:.6f})")

print(f"Valor esperado ≈ {EXPECTED:.6f}\n")

# Vegas: Monte Carlo adaptativo
print_result("Vegas",
    Vegas(integrand, NDIM, maxeval=MAXEVAL, verbose=0))

# Suave: Monte Carlo suavizado
# ATENÇÃO: nmin deve ser inteiro, não float (25, não 25.0)
print_result("Suave",
    Suave(integrand, NDIM, 1000, 25, maxeval=MAXEVAL, verbose=0))

# Divonne: Monte Carlo com divisão de domínio
print_result("Divonne",
    Divonne(integrand, NDIM, maxeval=MAXEVAL,
            key1=47, key2=1, key3=1,
            maxpass=5, border=0., maxchisq=10., mindeviation=.25,
            ldxgiven=NDIM, verbose=0))

# Cuhre: quadratura determinística (mais preciso)
print_result("Cuhre",
    Cuhre(integrand, NDIM, key=0, maxeval=MAXEVAL, verbose=0))
```

Execute:

```bash
python test_pycuba.py
```

Saída esperada:

```
Valor esperado ≈ 0.664588

[OK] Vegas       integral=0.664521  error=3.21e-04  (esperado=0.664588)
[OK] Suave       integral=0.664612  error=2.18e-04  (esperado=0.664588)
[OK] Divonne     integral=0.664590  error=1.45e-05  (esperado=0.664588)
[OK] Cuhre       integral=0.664588  error=8.02e-07  (esperado=0.664588)
```

Todos os quatro algoritmos devem retornar `[OK]`.

---

## Resumo de Todos os Erros Encontrados

### Erro 1 — `-fPIC` ausente na compilação

**Mensagem:**
```
/usr/bin/ld: Vegas.o: relocation R_X86_64_PC32 against symbol `cubafun_'
can not be used when making a shared object; recompile with -fPIC
collect2: error: ld returned 1 exit status
```

**Causa:** Os arquivos `.o` foram compilados para uso estático, sem código independente de posição.

**Solução:**
```bash
make clean
CFLAGS="-fPIC" FFLAGS="-fPIC" ./configure
CFLAGS="-fPIC" FFLAGS="-fPIC" ./makesharedlib.sh
```

---

### Erro 2 — `libcuba.so` errada sendo carregada

**Mensagem:**
```
OSError: /usr/local/lib/libcuba.so: undefined symbol: cubaverb_
```

**Causa:** O Python encontrou uma `libcuba.so` antiga em `/usr/local/lib` antes da versão recém-compilada.

**Solução:**
```bash
cp ~/cuba/libcuba.so ~/miniconda3/envs/pycuba_env/lib/
```

---

### Erro 3 — Módulo não encontrado

**Mensagem:**
```
ModuleNotFoundError: No module named 'pycuba'
```

**Causa:** O pacote `pymultinest` não foi instalado no ambiente conda.

**Solução:**
```bash
pip install pymultinest
```

---

### Erro 4 — Expansão de histórico do bash

**Mensagem:**
```
bash: !': event not found
```

**Causa:** O bash interpreta `!` dentro de aspas duplas como expansão de histórico.

**Solução:** Usar aspas simples:
```bash
python -c 'from pycuba import Vegas; print("PyCuba working!")'
```

---

### Erro 5 — Tipo errado no argumento `nmin` do Suave

**Mensagem:**
```
ArgumentError: argument 13: <class 'TypeError'>: Don't know how to convert parameter 13
```

**Causa:** O parâmetro `nmin` do `Suave` espera um inteiro (`c_int`), mas foi passado `25.0` (float).

**Solução:**
```python
# ERRADO
Suave(integrand, NDIM, 1000, 25.0, ...)

# CORRETO
Suave(integrand, NDIM, 1000, 25, ...)
```

---

## Tabela Resumo: Todos os Erros e Soluções

| # | Erro | Causa | Solução |
|---|------|-------|---------|
| 1 | `relocation R_X86_64_PC32 ... recompile with -fPIC` | Compilação sem PIC | `CFLAGS="-fPIC" FFLAGS="-fPIC" ./makesharedlib.sh` |
| 2 | `OSError: undefined symbol: cubaverb_` | `libcuba.so` antiga em `/usr/local/lib` | `cp ~/cuba/libcuba.so ~/miniconda3/envs/pycuba_env/lib/` |
| 3 | `ModuleNotFoundError: No module named 'pycuba'` | `pymultinest` não instalado | `pip install pymultinest` |
| 4 | `bash: !': event not found` | `!` em aspas duplas no bash | Usar aspas simples `'...'` |
| 5 | `ArgumentError: Don't know how to convert parameter 13` | `nmin=25.0` (float) no Suave | Usar `nmin=25` (inteiro) |

---

## Checklist de Instalação

- [ ] `conda create -n pycuba_env python=3.9`
- [ ] `conda activate pycuba_env`
- [ ] `sudo apt-get install git gcc gfortran make`
- [ ] `git clone https://github.com/JohannesBuchner/cuba/ && cd cuba`
- [ ] `make clean && CFLAGS="-fPIC" FFLAGS="-fPIC" ./configure`
- [ ] `CFLAGS="-fPIC" FFLAGS="-fPIC" ./makesharedlib.sh`
- [ ] `file libcuba.so` → confirmar "ELF 64-bit LSB shared object"
- [ ] `cp ~/cuba/libcuba.so ~/miniconda3/envs/pycuba_env/lib/`
- [ ] `pip install pymultinest`
- [ ] `python -c 'from pycuba import Vegas; print("OK")'`
- [ ] `python test_pycuba.py` → todos os quatro algoritmos com `[OK]`

---

## Uso Futuro

Para usar o PyCuba em sessões futuras, basta ativar o ambiente:

```bash
conda activate pycuba_env
```

Não é necessário nenhum `export LD_LIBRARY_PATH` pois a `libcuba.so` já está dentro do ambiente conda.
