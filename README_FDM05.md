# FDM-05 — Arquivos para incorporar à branch

- `src/navierstokes/fdm/predictor.py`: advecção conservativa com upwind, laplaciano Dirichlet com ghost cells, limite de estabilidade e preditor explícito.
- `tests/test_fdm_05.py`: testes de propriedades, fronteiras, Euler, estabilidade e convergência do laplaciano no interior.

Os arquivos são **novos**; não substitua `operators.py`, `boundary.py` ou `solver.py`.

## Execução

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_fdm_05.py -q
python -m pytest -q
ruff check src/navierstokes/fdm/predictor.py tests/test_fdm_05.py
```

## Limites da entrega

- A projeção de pressão e a sincronização dos campos centrais com os fluxos projetados ficam para FDM-06.
- O contrato assume cavidade fechada, com fluxos normais externos nulos.
- `SimulationConfig.kinematic_viscosity` exige `lid_velocity > 0` para o preditor atual.
- O teste de refinamento cobre o laplaciano **no interior**. A ordem global nas paredes e a ordem da advecção exigem testes manufaturados adicionais, se desejado.
- A estimativa de estabilidade é conservadora, baseada na soma de advecção e difusão; não é prova de estabilidade do solver Navier–Stokes completo.
