# TCC — Modelagem Numérica 2D das Equações de Navier–Stokes

Projeto computacional do Trabalho de Conclusão de Curso dedicado à resolução
numérica das equações de Navier–Stokes bidimensionais para escoamento
incompressível, comparando:

- **FDM/MDF** — Método das Diferenças Finitas;
- **FVM/MVF** — Método dos Volumes Finitos.

## Problema de referência

O caso-base é a cavidade com tampa móvel (*lid-driven cavity*) em domínio
bidimensional unitário. O projeto será usado para estudar convergência,
independência de malha, sensibilidade ao passo temporal e influência do número
de Reynolds, além da comparação entre FDM e FVM.

## Estrutura

```text
src/navierstokes/     implementação reutilizável
  fdm/                solver por diferenças finitas
  fvm/                solver por volumes finitos
experiments/          campanhas de experimentos
tests/                testes automatizados
results/
  raw/                dados brutos (não versionados)
  figures/            figuras selecionadas
  tables/             tabelas selecionadas
notebooks/            análise exploratória
docs/                 documentação metodológica
```

## Ambiente

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
```

Ative o ambiente virtual e instale o projeto em modo de desenvolvimento:

```bash
pip install -e ".[dev]"
```

Execute os testes com:

```bash
pytest
```

## Estado atual

A estrutura do projeto está preparada para a migração e validação do solver
FDM já utilizado nos experimentos do TCC. O solver FVM será implementado e
validado antes da produção dos resultados comparativos.

## Reprodutibilidade

Cada experimento deve registrar seus parâmetros, métricas de convergência,
tempo de execução e dados necessários para reconstruir tabelas e figuras.
Resultados numéricos não devem ser adicionados ao trabalho sem a configuração
correspondente.

## Autor

Luiz Felipe Oliveira Ribeiro — Engenharia da Computação, FAESA Centro
Universitário.
