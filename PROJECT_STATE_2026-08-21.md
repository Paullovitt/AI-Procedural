# AI-Procedural — Estado técnico consolidado

Data-base: 2026-08-21
Atualizado: 2026-08-22
Projeto local atual: `C:\Users\USER\Downloads\CODIGOS\TESTE\AI-Procedural`

## 1. Princípios que não podem ser quebrados

- Não usar redes neurais, neurônios, backpropagation, gradientes ou pesos neurais.
- GPU/VRAM deve ser usada nas partes estatísticas e de busca onde isso for adequado.
- Regras de domínio, conclusões e estratégias específicas de benchmark não podem ser programadas manualmente.
- `if/else` pode existir apenas como controle genérico de programa, segurança, verificação, matching, promoção/rejeição de hipóteses e infraestrutura.
- O conteúdo das regras deve ser descoberto pelo modelo e persistido como dados executáveis / rulebank / bytecode simbólico.
- Simuladores podem conter leis ocultas apenas para gerar experiências e auditar resultados; o learner não pode acessar essas leis.
- Ajustes permitidos: mecanismos gerais de memória, indução, busca, MDL, hipótese, revisão, confiança, planejamento, verificação e geração.
- Ajustes proibidos: inserir manualmente a regra que o teste quer que o modelo descubra.

## 2. Arquitetura atual

Fluxo conceitual:

```text
observações / fatos / experiências
        ↓
representação simbólica
        ↓
indução de padrões e regras
        ↓
RuleBank persistente (regras como dados)
        ↓
VM / executor genérico
        ↓
encadeamento + planejamento + prova
        ↓
plano semântico imutável
        ↓
planejamento discursivo
        ↓
renderer Bagaço p2–p5 em GPU
        ↓
verificadores semântico + ordem
        ↓
texto
```

Não há rede neural. O backend usa PyTorch apenas para tensores CUDA, hashing/busca/contagem e operações vetorizadas. `gradients=False` e nenhum treino por gradiente é usado.

## 3. Renderer e linguagem

### V8 / V9
- Gramática induzida do Bagaço por anti-unificação de n-grams.
- Wrapper descoberto e promovido: `relativamente a`, associado distribucionalmente a uma construção segura já verificada.
- Propostas induzidas passaram a competir com realizações verificadas.

### GPU runtime
Arquivo principal: `procedural_runtime_gpu.py`

- p2, p3, p4 e p5 em CUDA.
- Cache GPU versionado `xxh64_signed_v3`.
- Paridade CPU↔GPU p5 conferida; erro máximo observado: ~4.44e-16.
- Sem gradientes.

### Verificadores corrigidos
1. `ProtectedSlotVerifier`: presença dos slots semânticos.
2. `SemanticTraceVerifier`: preserva também a ordem/estrutura dos slots e detecta troca de papéis/valores.
3. `GpuLocalOrderVerifier`: verificador de ordem local calibrado por score p2–p5 na GPU, sem regras manuais de português.

Teste do verificador de ordem:
- recall em corrupção adjacente: 100%
- falso positivo no heldout limpo: 0%

## 4. Stress grande V10/V12

### Stress de 1.000.000 de fatos
Resultado do benchmark GPU rigoroso:

- 130 documentos
- 1.000.000 fatos
- 343.064 sentenças
- 39.424 parágrafos
- 32.862.824 caracteres
- 244.39 s
- 4.091,8 fatos/s
- erros semânticos: 0
- erros de slot: 0
- erros de semantic trace: 0
- conflitos funcionais de entrada: 0
- foco de parágrafo incorreto: 0
- suporte trigramal médio: 0.73794
- p90: 36 palavras
- p95: 44
- p99: 61
- realizações induzidas: 48.319 (~14.08%)

GPU durante esse stress:
- GPU: NVIDIA GeForce GTX 1660 SUPER 6 GB
- p2–p5 em GPU
- pico total de VRAM observado: ~1.706 GB
- utilização média: ~7.34%
- utilização máxima: 41%
- temperatura máxima: 45 °C

Adversarial:
- 5.457 trocas de papel → 100% detectadas
- 4.723 trocas de valor → 100% detectadas
- 10.000 substituições de slot → 100% detectadas
- 2.431 trocas de ordem → 100% detectadas
- falso positivo de ordem: 0%

### Problemas revelados pelo stress
- Diversidade abstrata global ainda baixa em documentos muito grandes.
- Planejamento discursivo antigo tinha contraste de posição ruim (~40% no primeiro stress).
- O antigo verificador de fluência com `LocalFluencyVerifier` não tinha poder discriminativo suficiente; foi substituído.

## 5. Auto-calibrações aprendidas

### Proposal weight
Busca shadow expandida em GPU.
- melhor pico interno encontrado: `proposal_weight = 0.24`
- escolha baseada em objetivo geral de suporte + diversidade − repetição, sob gate semântico rígido.

### Position weight
Busca e seleção estatística separada.
- `position_weight = 7.0`
- método: menor peso dentro do limite inferior Wilson 95% do melhor contraste, mantendo gates de suporte/repetição/semântica.

Heldout de posição:
- baseline 0.15: contraste correto ~48.27%
- candidato 7.0: ~95.52%
- erros semânticos/slots/trace: 0

### Diversidade V12
Planejador de foco por grafo + penalidade global de formas.

Baseline V11:
- continuidade entre focos/parágrafos ligados: ~5.74%
- unicidade abstrata média no doc: ~28.64%

V12:
- continuidade ligada: ~80.5–81.1%
- unicidade abstrata: ~35% em calibração
- peso mínimo próximo ao melhor absoluto: `diversity_weight = 2.6`
- preservação semântica: 100%

Resultado V12 final, 500.000 fatos:
- 173.600 sentenças
- 22.616 parágrafos
- 214.03 s
- 0 erros semânticos/slots/trace
- continuidade de grafo: 0.79989
- suporte p2–p5: 0.74789
- contraste de posição correto: 0.94176
- ataques adversariais: todos 100% detectados

## 6. V14 — parágrafos mais realistas

Comparação V12 vs V14 em 60.000 fatos:

V12:
- média de 4.78 sentenças/parágrafo
- mediana 5
- p90 7
- p95 7
- 1 foco/parágrafo

V14:
- média 30.16 sentenças/parágrafo
- mediana 21
- p90 73
- p95 100
- corpus alvo: média 31.02, mediana 21, p90 77, p95 100
- média de 7.08 focos por parágrafo
- transições ligadas dentro do parágrafo: 15.56%
- posição correta: ~96.68%
- 0 erros semânticos/slots/trace
- promotion gate: true

Calibração de repetição V14 concluiu que aumentar agressivamente a penalidade piorava outros critérios; a configuração conservadora foi mantida.

## 7. Rule VM autônoma — regras como dados

### Objetivo
Eliminar regras inteligentes codificadas como `if A então B` no código. O learner observa experiências e descobre funções/regras. As regras são persistidas no `RuleBank` e executadas por uma VM genérica.

### Hidden world
O simulador possui regras ocultas apenas para gerar observações e auditoria. O learner recebe apenas estados/transições/resultados e não acessa os schemas ocultos.

### V5 certificada
Resultado `Autonomous-RuleVM-MDL-Certified-GPU-v5`:
- treino: 150.000 experiências
- ruído de treino: 1.5%
- validação: 60.000
- teste: 100.000
- aprendizagem: ~0.92 s
- accuracy de transição: 100%
- erro de validação: 0 para todas as 12 relações
- 40 mundos de closure: 100% exatos
- Jaccard: 1.0
- validade das provas: 100%
- profundidade máxima de prova: 12
- GPU: CUDA, sem neural/gradientes
- pico PyTorch: ~80.8 MB

Observação importante: a ordem do modelo selecionado pode ser maior que a ordem mínima escondida. Isso é aceitável se a função for exatamente equivalente; auditorias funcionais devem prevalecer sobre comparação de schema textual.

### Generalidade
6 mundos novos, ruído de 0% a 5%:
- transition accuracy: 100% em todos
- closure exato: 100% em todos
- validade de prova: 100%
- todos certificados em paralelo
- todos os mundos exigiram busca até ordem 5 em algum ponto
- tempo total de aprendizado: ~6.22 s

### Drift / revisão autônoma
3 regras ocultas foram alteradas sem aviso.
- accuracy antes da revisão: 97.123%
- depois da revisão: 100%
- 3/3 mudanças detectadas
- 0 revisões falsas
- closure exato após revisão: 100%
- proof validity: 100%

A decisão de revisar usa ganho de compressão/MDL, não uma lista manual de relações alteradas.

## 8. Integração raciocínio → relatório

Teste `Integrated-Autonomy-V5-V13`:
- aprendizado: ~1.475 s
- transition accuracy: 100%
- fechamento primário: exato
- prova: válida
- alcance primário: 159 componentes
- profundidade da prova: 14
- 24 fatos no plano semântico
- V12/V13: 0 erros semânticos, slots ou trace

Limitação observada:
- ranking top-5 exato não foi perfeito nesse caso (`top5_exact_order=false`).
- isso é um alvo real para melhoria de inteligência/planejamento; não deve ser corrigido com regra manual.

## 9. O que NÃO conta como inteligência aprendida

Não considerar como aprendizado:
- `if risco == alto: ...`
- `if A depende B e B depende C: concluir A depende C`
- tabelas manuais de transitividade, causalidade ou exceções
- palavras/conectivos inseridos apenas para passar um benchmark

Permitido:
- `if hipótese passou no gate: promover`
- `if checksum falhou: rejeitar arquivo`
- `if GPU disponível: usar CUDA`
- loops, máscaras, dispatch de VM, matching genérico e verificações de integridade

A regra concreta deve estar em dados aprendidos, não em branches de domínio no código.

## 10. Bugs / abordagens rejeitadas

- Verificador de fluência antigo: recall efetivo 0 em corrupção de ordem em stress grande → rejeitado/substituído.
- Benchmark antigo `make_world()` permitia múltiplos valores para a mesma propriedade funcional → corrigido no benchmark; conflitos passaram a ser um teste separado.
- Position model antigo: contraste real vs aleatório ruim → recalibrado e validado em heldout.
- Busca RuleVM V2 por milhares de kernels pequenos: correta, mas muito lenta → interrompida sem aproveitar resultado parcial e substituída por busca vetorizada.
- Taxonomia retórica discreta por BIC: não convergiu de forma confiável → hipótese rejeitada; não foi forçada.
- Sentence-initial specificity como detector de conectivo retórico: aprendeu boilerplate web → rejeitado, sem blacklist manual.

## 11. Arquivos centrais atuais

Renderer / linguagem:
- `procedural_runtime_gpu.py`
- `procedural_runtime_v12.py`
- `procedural_runtime_v13.py`
- `procedural_runtime_v14.py`

Raciocínio / regras:
- `autonomous_reasoning_gpu.py`
- `autonomous_rule_vm_v2.py` (experimento lento/rejeitado como implementação principal)
- `autonomous_rule_vm_v3.py` / v4 (etapas intermediárias)
- RuleBank certificado atual: `rigorous_results_v12/AUTONOMOUS_RULEBANK_MDL_V5.json`

Resultados:
- `rigorous_results_v2/` — stress de 1M fatos
- `rigorous_results_v12/final.json` — V12 final
- `rigorous_results_v12/autonomous_rule_vm_v5.json`
- `rigorous_results_v12/rule_vm_v5_generality.json`
- `rigorous_results_v12/rule_vm_v5_drift.json`
- `rigorous_results_v12/integrated_autonomy_v5_v13.json`
- `rigorous_results_v12/v12_v14_comparison.json`
- `rigorous_results_v12/v14_repetition_calibration.json`

## 12. Estado atual / próximos problemas reais

Fortes:
- fidelidade semântica
- detecção de corrupção
- GPU p2–p5
- descoberta de regras opacas
- revisão após drift
- closure multi-hop e prova
- estrutura de comprimento de sentença/parágrafo próxima do corpus

Ainda fracos / abertos:
- naturalidade lexical livre ainda limitada por famílias seguras de microcláusulas
- diversidade abstrata em documentos enormes ainda não é alta
- ranking de intervenções/top-k pode não ser exatamente ordenado em todos os mundos
- geração de explicações ainda é mais factual/estruturada que humana
- descoberta de gramática deve continuar substituindo gradualmente templates/microcláusulas programadas
- raciocínio deve ser testado em universos mais variados, com parcial observabilidade, exceções, relações não booleanas e planejamento adversarial

## 13. Regra de promoção futura

Uma nova versão só deve ser promovida se:
1. não reduzir preservação semântica;
2. passar regressão adversarial;
3. demonstrar melhoria em heldout separado;
4. não depender de regra do benchmark inserida manualmente;
5. manter regra aprendida persistida como dado/RuleBank;
6. registrar falhas e versões rejeitadas, não apagar histórico;
7. manter GPU ativa nas partes estatísticas e de busca relevantes.

## 14. Prompt + Learned RuleVM V6

A camada de prompt foi conectada a um RuleBank aprendido, mantendo o RuleVM simples.

Fluxo promovido:

```text
prompt -> conceitos observados -> learner de associações Bagaço em CUDA
       -> LearnedAssociationRuleBank -> IndexedAssociationRuleVM
       -> grafo explícito -> V14 CUDA -> verificadores -> saída lexicalizada
```

Arquivos:
- `autonomous_rule_vm_v6.py`: learner PMI positivo em GPU, RuleBank explícito e VM indexado.
- `prompt_runtime_v14.py`: extrai conceitos, invoca o learner/VM e mantém provenance/confiança/suporte/evidência.
- `prompt_session_v14.py`: mantém V14, GPU e caches residentes entre prompts.
- `benchmark_prompt_rulevm_v6.py`: bateria de generalidade/performance.
- `test_rulevm_v6_prompt.py`: regressões do V6.

Decisões de qualidade:
- conceitos compostos de 2 a 5 palavras são tratados como unidade e usam evidência da expressão inteira;
- se não existir evidência específica suficiente, não há fallback silencioso para a última palavra ambígua;
- p3-p5 de expressões compostas é consultado sob demanda e cacheado;
- regras fracas não são promovidas apenas para preencher comprimento;
- `evidence_limited=true` sinaliza quando o alvo de tamanho não pode ser alcançado com o RuleBank promovido.

Bateria de 6 temas em sessão única:
- semântica: 6/6 verificadas;
- slot errors: 0;
- trace errors: 0;
- IDs internos expostos: 0;
- cobertura dos conceitos do prompt: 100%;
- sentenças exatas únicas: 100%;
- RuleVM máximo observado: ~0,04 ms;
- prompts posteriores com expressões novas: aproximadamente 13–18 ms de raciocínio na bateria;
- prompt repetido/caches aquecidos: camada de raciocínio observada em ~2 ms.

Limitação deliberada: corpora esparsos podem produzir texto abaixo do tamanho solicitado. Exemplo de segurança digital mostrou pouca evidência específica; o sistema prefere declarar `evidence_limited` a reintroduzir associações ambíguas de `digital`.

### Regressão V5 após introdução do V6

Benchmark completo V5 novamente executado:
- treino 150.000, validação 60.000, teste 100.000;
- aprendizagem: ~1,075 s;
- transition accuracy: 1.0;
- closure exact: 1.0;
- proof validity: 1.0;
- 12/12 relações certificadas para fixed point paralelo.

Generalidade V5 novamente executada:
- 6/6 mundos com transition accuracy, closure e proof validity = 1.0;
- todos certificados;
- tempo total de aprendizado: ~5,498 s.

Drift novamente executado:
- pré-revisão: 0,97123;
- pós-revisão: 1.0;
- relações revisadas: [1,5,9], exatamente as alteradas;
- falsas revisões: 0;
- closure/provas após revisão: 1.0.

Resultado V6 principal: `rigorous_results_v12/prompt_rulevm_v6_generality.json`.


## 15. Auditoria final de código e integridade

Foi adicionada `project_audit.py` como gate reproduzível antes de promoção/commit. Ela verifica:
- compilação de todos os Python versionáveis;
- `architecture_guard.py` em toda a árvore de código;
- parse de todos os JSON e JSON/XZ;
- `SHA256SUMS.txt`;
- presença e consistência dos arquivos centrais;
- README.md == README.txt;
- configuração V14 + Learned-Association-RuleVM-v6;
- `RUN_GPU.bat` apontando para a sessão persistente;
- ausência de temporários conhecidos.

Correções adicionais da auditoria:
- `--prompt`, `--prompt-file`, `--facts` e `--smoke` agora são mutuamente exclusivos;
- conceitos compostos de 3 a 5 palavras não podem cair para a última palavra ambígua;
- observações do próprio prompt têm reserva explícita no RuleBank/seleção, mesmo sob orçamento pequeno;
- `requirements-gpu.txt` foi atualizado para V14/RuleVM V6;
- anotações de tipos foram ampliadas nos runtimes históricos sem alterar suas regras ou resultados.

Regressão final após essas mudanças:
- testes automáticos: 9/9;
- Prompt-RuleVM-V6: 6/6 sem erros semânticos/slot/trace, cobertura de conceitos 100%;
- V5 principal: accuracy/closure/proof = 1.0, aprendizagem ~1,075 s;
- V5 generalidade: 6/6 mundos = 1.0, total ~5,498 s;
- drift: revisões [1,5,9], falsas revisões 0, pós-revisão 1.0.

## 16. Evidence Argument Planner V14

A linha ativa continua sendo **V14**. Foi adicionada uma camada argumentativa baseada somente em regras/evidências já aprendidas, sem criar V15 e sem introduzir conhecimento de domínio manual.

Fluxo promovido:

```text
prompt -> conceitos -> RuleBank V6 aprendido em CUDA -> RuleVM indexado
       -> Evidence Argument Planner V14
       -> opening -> development -> synthesis
       -> Renderer V14 CUDA -> verificadores -> texto
```

Arquivo novo:
- `argument_planner_v14.py`: seleciona, ordena e filtra regras já disparadas; não inventa fatos, causalidade ou conclusão de domínio.

Integrações alteradas:
- `prompt_runtime_v14.py`: executa o planner após a RuleVM e antes de converter regras em fatos de renderização.
- `procedural_runtime_v14.py`: realiza superfícies específicas por fase e força fronteiras de parágrafo coerentes com as fases.
- `autonomous_rule_vm_v6.py`: contextos de conceitos compostos preservam a expressão observada completa; um token solto não é mais apresentado como conceito independente quando a evidência real é uma construção p3-p5.
- `run_gpu.py` / `prompt_session_v14.py`: reportam estatísticas do Argument Planner e mantêm refill de regras fortes quando a desambiguação remove candidatos.
- `gpu_config.json`: `argument_planner_enabled=true`.

Mecanismos genéricos do planner:
- confiança, suporte, score e profundidade da regra;
- cobertura dos conceitos do prompt;
- alinhamento contextual global usando estatísticas do próprio corpus;
- fases monotônicas `opening -> development -> synthesis`;
- rejeição de associação isolada sem corroboração suficiente;
- invalidação de síntese construída a partir de vizinhos posteriormente rejeitados;
- busca de regras fortes adicionais quando a filtragem contextual abre espaço, sem baixar thresholds de evidência;
- prevenção de repetição imediata do mesmo template quando existe alternativa semanticamente equivalente.

Exemplos de bugs de qualidade corrigidos durante a implementação:
- `energia solar` deixou de promover tokens soltos como `movidos`; a evidência passa a ser preservada como expressão observada, por exemplo `movidos a energia solar` / `produção de energia solar`.
- no prompt de música, `composição musical` passou a ter prioridade contextual e expansões `composição química/corporal/nutricional` deixaram de ser realizadas quando incompatíveis com o restante do prompt.
- associação isolada `privacidade cofina` foi removida no caso esparso de segurança digital.
- `hospitais públicos` permanece elegível por possuir alinhamento contextual/morfológico com `saúde pública`.
- fragmentos compostos são aparados ao menor trecho observado que contém a expressão e sua evidência, reduzindo finais quebrados de n-grama.

Bateria final de 6 temas, alvo de 1000 caracteres:
- exploração espacial: 957, dentro da tolerância;
- energia solar: 1019, dentro da tolerância;
- agricultura sustentável: 949, dentro da tolerância;
- música clássica: 953, dentro da tolerância;
- segurança digital: 175, `evidence_limited=true`;
- saúde pública: 1032, dentro da tolerância.

Gates finais dessa bateria:
- semantic_verified: 6/6;
- slot errors: 0;
- trace errors: 0;
- IDs internos expostos: 0;
- cobertura dos conceitos do prompt: 100%;
- sentenças exatas duplicadas: 0;
- repetição imediata de template: 0;
- fases argumentativas monotônicas: 6/6;
- 5/6 prompts dentro da tolerância;
- falhas de tamanho não explicadas: 0;
- 9 regras removidas pelo filtro contextual na bateria.

Performance medida:
- carga inicial única GPU/modelo: ~2,927 s;
- RuleVM máximo: ~0,029 ms;
- Argument Planner máximo: ~0,631 ms;
- raciocínio dos prompts posteriores: média ~16,98 ms, máximo ~23,0 ms.

Testes automáticos após a implementação: **14/14**.
Novo arquivo de regressão: `test_argument_planner_v14.py`.
Resultado multi-tema atualizado: `rigorous_results_v12/prompt_rulevm_v6_generality.json`, formato `Prompt-RuleVM-V6-ArgumentPlanner-V14-Generality`.

Limitação que permanece: o sistema é associativo/evidencial, não um LLM causal. Em áreas esparsas, ele deliberadamente produz menos texto e marca `evidence_limited` em vez de fabricar explicações não suportadas.

Regra operacional permanente: novas implementações continuam na V14 e a documentação do projeto deve ser atualizada no mesmo commit de promoção.

### Regressão V5 final após o Argument Planner

Executada novamente depois da implementação e documentação da V14:
- benchmark principal: transition accuracy = 1.0, closure exact = 1.0, proof validity = 1.0, 12/12 relações certificadas; aprendizagem ~1,092 s;
- generalidade: 6/6 mundos com transition/closure/proof = 1.0; tempo total de aprendizado ~5,645 s;
- drift: relações [1,5,9] detectadas e revisadas exatamente, 0 falsas revisões, pós-revisão = 1.0.


## 17. Robust Semantic Intake V14 — texto imperfeito sem pipeline de correção

A linha promovida continua sendo **V14** e permanece não neural. A nova camada processa texto bruto
imperfeito diretamente e produz uma projeção explícita para as camadas seguintes; não cria uma versão
textual corrigida nem reescreve o dataset.

Arquivos promovidos:
- `robust_semantic_intake_v14.py`: tokens, nós, arestas, frases, âncoras numéricas, fuzzy e learner de aliases.
- `train_robust_semantic_v14.py`: ingestão TXT/JSON/JSONL e promoção explícita de aliases recorrentes.
- `test_robust_semantic_v14.py`: regressões unitárias/performance/integridade.
- `robust_semantic_regressions_v14.json`: 16 referências permanentes.
- `robust_semantic_battery_v14.py`: fuzzing textual, corrupções sintéticas, combinação de erros e busca adversarial.
- `robust_semantic_failure_archive_v14.jsonl`: arquivo cumulativo de falhas encontradas.
- `robust_semantic_failure_replay_v14.py`: reexecuta todas as falhas arquivadas como casos permanentes.
- `rigorous_results_v12/robust_semantic_failure_replay_v14.json`: estado resolvido/não resolvido do archive.
- `rigorous_results_v12/robust_semantic_v14.json`: relatório integral da última bateria.

Mecanismos:
- texto original preservado em `raw_text` e spans;
- canonicalização apenas como hipótese interna auditável;
- fast-path exato O(1);
- 225.058 assinaturas de uma deleção para typo comum;
- fallback amplo somente sob demanda;
- Damerau limitado + probes O(length) para transposição/remoção e nomes raros;
- recuperação de mojibake, acentos, HTML/JSON estrutural, Unicode/emoji;
- segmentação de palavra grudada/partida apenas com evidência p2 do corpus;
- âncoras explícitas para número/data/hora;
- repetição local descartável, repetição distante preservada quando pode carregar contexto novo;
- negação e dupla negação preservadas nas bridges;
- palavras válidas nunca são reescritas silenciosamente como outra palavra válida;
- learner de aliases exige suporte, dominância, confiança combinada e pelo menos dois contextos distintos.

Performance do caminho quente observada durante desenvolvimento:
- texto limpo: mediana ~0,21 ms;
- quatro typos comuns: mediana ~0,25 ms;
- exemplo longo ruidoso: mediana ~0,75 ms.

Bateria extrema final de 22/08/2026:
- 16/16 referências permanentes aprovadas;
- 448 avaliações;
- recall médio de informação: 0.904935;
- recall numérico/data/quantidade: 0.994978;
- p50: ~0,474 ms; p95: ~13,333 ms; máximo: ~69,047 ms;
- 121 avaliações com falhas classificadas;
- classes: 113 perdas de informação, 15 inclusões de ruído, 9 erros de relação/significado, 3 numéricos;
- archive cumulativo: 353 casos distintos;
- gate `ROBUST SEMANTIC V14 BATTERY GATE: OK`.

Regressões após integração:
- `python -m unittest discover -v`: 24/24;
- Prompt RuleVM V6/Argument Planner: gate OK, 6/6 semantic verified, 5/6 dentro da tolerância e 1 `evidence_limited`;
- V5 principal: transition/closure/proof = 1.0, 12/12 certificadas, aprendizagem ~0,853 s;
- V5 generalidade: 6/6 mundos = 1.0, todos exigiram ordem 5, total ~6,501 s;
- drift: [1,5,9] detectadas exatamente, falsas revisões 0, pós-revisão = 1.0;
- `architecture_guard.py`: OK; `validate_model.py`: OK.

Hardware desta validação final: NVIDIA GeForce RTX 3050 6 GB, CUDA 12.8, PyTorch 2.11.0+cu128.

Limitações conhecidas permanecem explícitas: corrupções simultâneas muito fortes, várias palavras grudadas e
truncamentos severos ainda conseguem causar perda de sinal. Esses casos permanecem no archive para o
ciclo detectar -> minimizar -> regressão permanente -> corrigir -> rerodar.

### Replay permanente das falhas adversariais

O archive deixou de ser somente registro passivo. `robust_semantic_failure_replay_v14.py` reexecuta
todos os casos já descobertos usando a referência original do caso-base e informa quais foram resolvidos
por mudanças posteriores sem apagar o histórico. Na validação final: 353 casos arquivados, 32 já
resolvidos e 321 ainda não resolvidos; p50 ~2,146 ms, p95 ~25,741 ms e máximo ~50,701 ms no conjunto
adversarial histórico. Casos não resolvidos permanecem visíveis; eles não são tratados como regressões
aprovadas nem usados para justificar redução dos gates atuais.

## 18. Persistent Dimensional Memory V14 — memória episódica externa complementar

A linha ativa permanece **V14**. Foi integrada uma memória persistente externa inspirada no repositório
`AI-Memory`, sem substituir a memória estatística Bagaço já existente.

### Arquitetura promovida

```text
prompt bruto
   ↓
Robust Semantic Intake V14 → shadow canônico de índice
   ↓
Persistent Dimensional Memory V14 (SQLite incremental)
   ↓ episódios recuperados + provenance
RuleVM V6 / memory_retrieval
   ↓
Evidence Argument Planner V14
   ↓
Renderer V14 CUDA + verificadores
```

Arquivo principal: `persistent_memory_v14.py`. O conhecimento episódico é explícito e auditável em
`episodes`, `terms`, `postings` e `edges`; não há embedding treinável, rede neural ou peso de domínio.
O banco de uso real fica em `memory_v14/episodic_v14.sqlite3` e é ignorado pelo Git.

Decisões de segurança/qualidade:
- somente entradas não interrogativas do usuário são autoarmazenadas após uma geração válida; perguntas explícitas (`?`/`¿`) são usadas para consulta, mas não promovidas como fatos;
- texto gerado pelo próprio modelo não vira memória automaticamente;
- o texto bruto é preservado; canonicalização serve apenas como shadow de índice;
- repetição exata aumenta `recurrence` sem duplicar episódio;
- conflitos permanecem registros separados em vez de serem mesclados como fato verdadeiro;
- recuperação é convertida em regra genérica `memory_retrieval` com ID/origem como evidência;
- esquecimento atualiza termos/postings/arestas incrementalmente;
- expansão associativa rejeita hubs cujo document frequency excede 20% dos episódios; busca exata permanece disponível;
- `/memoria` mostra o estado da memória na sessão persistente.

Configuração promovida em `gpu_config.json`:
- `persistent_memory_enabled=true`;
- `persistent_memory_path=memory_v14/episodic_v14.sqlite3`;
- `persistent_memory_top_k=4`;
- `persistent_memory_candidate_limit=512`;
- `persistent_memory_associative=true`;
- `persistent_memory_associative_per_term=4`;
- `persistent_memory_auto_store_user=true`;
- `persistent_memory_inject_chars=480`;
- `persistent_memory_rule_cap=4`;
- `persistent_memory_min_query_term_coverage=0.30`;
- `persistent_memory_max_associative_document_ratio=0.20`.

### Bateria final de 22/08/2026

`python persistent_memory_battery_v14.py`:
- 20.000 registros sintéticos base. 20.005 episódios no pico com casos auxiliares;
- 500 consultas principais;
- exact top-1 = 1.0; numeric top-1 = 1.0;
- 50 consultas irrelevantes: 0 falso positivo;
- noisy semantic shadow retrieval = true;
- duplicate recurrence = true;
- contradição isolada = true;
- persistência após reopen = true; reopen ~2,347 ms;
- forget incremental = true;
- latência: mean ~3,958 ms, p50 ~4,584 ms, p95 ~4,930 ms, p99 ~5,095 ms, max ~5,246 ms;
- pico: 61.928 dimensões. 123.819 arestas dirigidas;
- integração viva: o fato `Meu carro é um Civic 2015.` é armazenado; a pergunta seguinte não é gravada (`interrogative`),
  e tanto a segunda quanto a terceira pergunta recuperam 1 episódio, injetam 1 regra e preservam `Civic 2015`;
  semantic_verified=true, slot_errors=0, trace_errors=0;
- gate: **OK**.

Novo resultado: `rigorous_results_v12/persistent_memory_v14.json`.
Novos testes: `test_persistent_memory_v14.py` (10 casos). A suíte total após a integração passa a 34 testes.

Limitações explícitas:
- a integração SQLite V14 foi medida até ~20 mil episódios nesta promoção; os benchmarks de 1 milhão do
  repositório AI-Memory original não são automaticamente transferíveis para esta implementação;
- busca sem sobreposição lexical depende do grafo dimensional já observado e ainda não oferece recall semântico geral;
- ainda não existe política automática de expiração, consolidação temporal, resolução de conflito ou importância;
- memória é local ao banco configurado e ainda não possui separação multiusuário.

### Validação isolada do Prompt RuleVM após memória persistente

`benchmark_prompt_rulevm_v6.py` agora força `persistent_memory_enabled=false` para que a memória episódica
local não contamine a regressão de generalidade do corpus/RuleVM/Planner. O resultado salvo registra
`persistent_memory_isolated=true` e a auditoria rejeita qualquer resultado que contenha evidência
`memory_retrieval`/`memória recuperada`.

Última execução isolada de 22/08/2026:
- tamanhos: 957 / 1019 / 949 / 953 / 175 / 1032 caracteres;
- 5/6 dentro da tolerância; segurança digital permanece `evidence_limited=true`;
- semantic verified 6/6; slot/trace errors 0; cobertura de conceitos 100%;
- carga única ~2,927 s; raciocínio posterior médio ~16,98 ms, máximo ~23,0 ms;
- RuleVM máximo ~0,029 ms; Argument Planner máximo ~0,631 ms;
- 9 regras removidas pelo filtro contextual; fases monotônicas 6/6; repetição imediata de template 0;
- gate: OK.
