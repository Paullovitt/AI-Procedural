# Regras permanentes de arquitetura — AI-Procedural

Estas regras são obrigatórias para toda evolução futura do projeto.

## 1. Proibido usar redes neurais

O modelo não pode conter ou depender de:

- neurônios artificiais;
- camadas neurais;
- MLP, CNN, RNN, LSTM, GRU, Transformer, Attention neural ou equivalentes;
- embeddings treináveis;
- parâmetros treináveis em tensores;
- backpropagation;
- gradient descent;
- autograd para aprendizagem;
- optimizers neurais;
- fine-tuning neural.

PyTorch/CUDA pode ser usado somente como backend de computação paralela sobre tensores discretos/estatísticos. O uso de `torch` não transforma o sistema em rede neural desde que não existam parâmetros treináveis, gradientes, módulos neurais ou backpropagation.

## 2. Proibido armazenar conhecimento em pesos

Conhecimento do modelo deve existir como estruturas explícitas e auditáveis, por exemplo:

- contagens;
- tabelas de frequência;
- grafos;
- fatos;
- episódios;
- regras induzidas;
- truth tables;
- índices;
- sketches;
- bytecode simbólico;
- hipóteses e respectivas evidências;
- estatísticas de confiança verificáveis.

Não usar vetores/matrizes de pesos treináveis como representação de conhecimento.

Coeficientes escalares de engenharia, quando inevitáveis para ranking ou combinação de scores, não podem carregar conhecimento do domínio. Devem ser poucos, interpretáveis, registrados e preferencialmente selecionados automaticamente por heldout/MDL. Sempre que possível, devem ser substituídos por comparação direta de evidência ou seleção de modelo.

## 3. Regras do problema não podem ser codificadas manualmente

É proibido resolver tarefas por lógica específica embutida no código, por exemplo:

```python
if A_depende_de_B and B_depende_de_C:
    concluir(A_depende_de_C)
```

ou qualquer equivalente que já contenha a lei que o modelo deveria descobrir.

`if/else` é permitido apenas para controle genérico de execução, integridade, I/O, GPU, matching, promoção/rejeição de hipóteses e outros mecanismos universais.

As regras de domínio devem ser descobertas pelo modelo a partir de observações/experiências e armazenadas como dados executáveis, não como branches escritos pelo programador.

## 4. Autonomia de aprendizagem

O código deve fornecer mecanismos gerais para:

- observar;
- associar;
- gerar hipóteses;
- testar hipóteses;
- medir evidência;
- selecionar complexidade;
- promover/rejeitar regras;
- revisar regras após erro;
- esquecer/enfraquecer hipóteses ruins;
- planejar;
- verificar;
- explicar por trilha de evidência.

O código não deve fornecer a resposta que esses mecanismos precisam descobrir.

## 5. GPU/VRAM

Quando houver trabalho paralelizável de modelo, usar GPU/VRAM por padrão.

Prioridades:

1. operações vetorizadas/batched;
2. evitar milhares de kernels pequenos;
3. caches compactos e versionados;
4. manter margem de VRAM;
5. CPU somente para controle simbólico irregular, I/O ou operações que não ganham com GPU.

## 6. Código limpo, otimizado e performático

Toda nova implementação deve buscar:

- código simples e legível;
- eliminar duplicação;
- funções pequenas com responsabilidades claras;
- estruturas de dados explícitas;
- nomes que descrevam função real;
- nenhum caminho morto após promoção/rejeição;
- evitar otimização prematura sem benchmark;
- medir tempo, RAM, VRAM e throughput;
- preferir batch/vectorização a loops Python quando houver ganho real;
- manter compatibilidade com validação e auditoria.

Mudança de performance só é promovida se preservar correção.

## 7. Shadow → teste → promoção

Mudanças de aprendizagem/raciocínio seguem:

`baseline -> shadow candidate -> heldout -> adversarial -> stress -> promote/rollback`

Uma mudança não deve ser promovida apenas porque melhora um exemplo.

Gates mínimos:

- nenhuma regressão semântica;
- nenhuma regra de domínio inserida manualmente;
- nenhuma rede neural/backprop/peso treinável;
- generalização em dados não usados para calibrar;
- resultado reproduzível;
- uso de GPU validado quando aplicável.

## 8. Auditoria

`architecture_guard.py` deve ser executado antes de promover uma mudança. `project_audit.py` deve ser executado antes de commit/push de uma promoção. O guard não substitui revisão arquitetural, mas serve como barreira automática.

Estas regras têm prioridade sobre conveniência de implementação e sobre ganhos rápidos de benchmark.

## 9. Runtime promovido e documentação

- Toda implementação nova promovida deve continuar dentro da **V14**.
- Não criar V15/V16 etc. apenas para adicionar mecanismo novo; a V14 é a linha ativa de evolução até decisão explícita em contrário.
- Mudança relevante na V14 deve atualizar no mesmo conjunto de alterações: `README.md`, `README.txt`, `GPU_README.txt` e `PROJECT_STATE_2026-08-21.md`.
- Quando a mudança alterar testes, métricas, runtime, arquivos centrais ou comportamento, atualizar também benchmark/resultados e `SHA256SUMS.txt`.
- A documentação deve registrar limitações reais; é proibido declarar qualidade/performance não medida ou esconder `evidence_limited`/regressões.
