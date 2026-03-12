# Guia de Instalação do PyCuba

**Ambiente:** Linux (Ubuntu/Debian) — Python 3.9 via Conda  
**Pacote:** `pymultinest` (inclui PyCuba)  
**Repositório:** https://github.com/JohannesBuchner/PyMultiNest

---

## 1. Criar o Ambiente Conda

```bash
conda create -n pycuba_env python=3.9
conda activate pycuba_env
```

> **Por quê Python 3.9?** O PyMultiNest é um pacote antigo e Python 3.9 oferece a melhor compatibilidade.

---

## 2. Instalar Dependências do Sistema

```bash
sudo apt-get install git gcc gfortran make
```

---

## 3. Compilar a Biblioteca Cuba (C)

O PyCuba é apenas um wrapper Python — ele precisa da biblioteca `libcuba.so` compilada a partir do código C.

### 3.1 Clonar o repositório

```bash
git clone https://github.com/JohannesBuchner/cuba/
cd cuba
```

### 3.2 Compilar com `-fPIC`

> **Erro comum:** Se você rodar `./configure && ./makesharedlib.sh` sem as flags abaixo, vai ocorrer este erro:
> ```
> relocation R_X86_64_PC32 against symbol `cubafun_' can not be used
> when making a shared object; recompile with -fPIC
> ```

A solução é passar `-fPIC` explicitamente:

```bash
make clean
CFLAGS="-fPIC" FFLAGS="-fPIC" ./configure
CFLAGS="-fPIC" FFLAGS="-fPIC" ./makesharedlib.sh
```

### 3.3 Verificar a compilação

```bash
ls -lh libcuba.so
file libcuba.so
```

Saída esperada:
```
libcuba.so: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked
```

---

## 4. Instalar o `libcuba.so` no Ambiente Conda

> **Erro comum:** Se `libcuba.so` não estiver no lugar certo, ao importar PyCuba você verá:
> ```
> OSError: /usr/local/lib/libcuba.so: undefined symbol: cubaverb_
> ```
> Isso acontece porque existe uma versão antiga/quebrada do `libcuba.so` no sistema. A solução mais confiável é copiar a versão compilada diretamente para dentro do ambiente conda.

```bash
cp ~/cuba/libcuba.so ~/miniconda3/envs/pycuba_env/lib/
```

> **Por que não usar `LD_LIBRARY_PATH`?** Essa variável precisa ser exportada toda vez que uma nova sessão é aberta. Copiar para o `lib/` do ambiente conda garante que a biblioteca correta seja sempre encontrada automaticamente.

---

## 5. Instalar o PyMultiNest (Python)

```bash
pip install pymultinest
```

---

## 6. Testar a Instalação

> **Atenção ao usar o shell:** O bash interpreta `!` como expansão de histórico. Use **aspas simples** no comando abaixo:

```bash
python -c 'from pycuba import Vegas; print("PyCuba funcionando!")'
```

Se aparecer `PyCuba funcionando!`, a instalação está completa.

---

## 7. Script de Teste Completo

Salve como `test_pycuba.py` e rode com `python test_pycuba.py`:

```python
import math
from pycuba import Vegas, Suave, Divonne, Cuhre

# Integrando: sin(x) * cos(y) * exp(z) sobre [0,1]^3
# Resultado analítico ≈ 0.6646

def integrand(ndim, xx, ncomp, ff, userdata):
    x, y, z = [xx[i] for i in range(ndim.contents.value)]
    ff[0] = math.sin(x) * math.cos(y) * math.exp(z)
    return 0

NDIM    = 3
MAXEVAL = 100000
EXPECTED = (1 - math.cos(1)) * math.sin(1) * (math.e - 1)

def print_result(name, res):
    r = res['results'][0]
    diff = abs(r['integral'] - EXPECTED)
    status = "OK" if diff < 1e-3 else "WARN"
    print(f"[{status}] {name:<10}  integral={r['integral']:.6f}  "
          f"error={r['error']:.2e}  (esperado={EXPECTED:.6f})")

print(f"Valor esperado ≈ {EXPECTED:.6f}\n")

print_result("Vegas",   Vegas(integrand, NDIM, maxeval=MAXEVAL, verbose=0))
print_result("Suave",   Suave(integrand, NDIM, 1000, 25, maxeval=MAXEVAL, verbose=0))
print_result("Divonne", Divonne(integrand, NDIM, maxeval=MAXEVAL, key1=47, key2=1, key3=1,
                                maxpass=5, border=0., maxchisq=10., mindeviation=.25,
                                ldxgiven=NDIM, verbose=0))
print_result("Cuhre",   Cuhre(integrand, NDIM, key=0, maxeval=MAXEVAL, verbose=0))
```

> **Erro comum no Suave:** O segundo argumento posicional `nmin` deve ser **inteiro**, não float.  
> ❌ `Suave(integrand, NDIM, 1000, 25.0, ...)` → `ArgumentError`  
> ✔ `Suave(integrand, NDIM, 1000, 25, ...)`

Saída esperada:
```
Valor esperado ≈ 0.664588

[OK] Vegas       integral=0.664521  error=3.21e-04  (esperado=0.664588)
[OK] Suave       integral=0.664612  error=2.18e-04  (esperado=0.664588)
[OK] Divonne     integral=0.664590  error=1.45e-05  (esperado=0.664588)
[OK] Cuhre       integral=0.664588  error=8.02e-07  (esperado=0.664588)
```

---

## Resumo dos Erros e Soluções

| Erro | Causa | Solução |
|------|-------|---------|
| `relocation R_X86_64_PC32 ... recompile with -fPIC` | Cuba compilada sem flag PIC | `CFLAGS="-fPIC" FFLAGS="-fPIC" ./makesharedlib.sh` |
| `OSError: undefined symbol: cubaverb_` | `libcuba.so` errada sendo carregada | `cp ~/cuba/libcuba.so ~/miniconda3/envs/pycuba_env/lib/` |
| `ModuleNotFoundError: No module named 'pycuba'` | pymultinest não instalado | `pip install pymultinest` |
| `event not found` no terminal | Bash interpreta `!` | Usar aspas simples: `python -c '...'` |
| `ArgumentError: Don't know how to convert parameter 13` | `nmin` passado como float | Usar inteiro: `25` em vez de `25.0` |

---

## Ativação para Uso Futuro

```bash
conda activate pycuba_env
python test_pycuba.py
```
