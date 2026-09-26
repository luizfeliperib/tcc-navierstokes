# Metodologia computacional

Este documento acompanha a implementação e registra decisões que precisam ser
reproduzíveis no TCC.

## Problema físico

O caso-base é a cavidade bidimensional com tampa móvel (*lid-driven cavity*),
em domínio unitário e fluido incompressível.

## Métodos

- FDM: discretização por diferenças finitas e acoplamento pressão-velocidade
  por método de projeção.
- FVM: discretização por volumes de controle, fluxos nas faces e acoplamento
  pressão-velocidade pelo algoritmo SIMPLE.

## Regra de reprodutibilidade

Nenhum resultado deve entrar no TCC sem registrar, no mínimo: método, Reynolds,
malha, passo temporal, tolerância, limite de iterações, critério de convergência
e tempo de execução.

Resultados FVM só devem ser comparados ao FDM após validação do solver.
