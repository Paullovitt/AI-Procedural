AI-Procedural / Bagaço Renderer V14 + Learned RuleVM V6 + Evidence Argument Planner

REGRA ARQUITETURAL PERMANENTE
=============================
Este projeto é deliberadamente NÃO NEURAL.

É proibido introduzir redes neurais, neurônios, embeddings treináveis, backpropagation,
gradient descent, autograd de aprendizagem, torch.nn/optimizers ou regras de domínio
codificadas manualmente para resolver benchmarks.

PyTorch/CUDA é permitido SOMENTE como backend tensorial para computação discreta,
estatística e busca vetorizada. O conhecimento deve permanecer explícito e auditável:
contagens, fatos, grafos, episódios, regras induzidas, truth tables, RuleBanks, índices,
hipóteses, evidências, confiança, planos e provas.

Regras completas: PROJECT_RULES.md
Estado técnico: PROJECT_STATE_2026-08-21.md
Guard obrigatório antes de promoção: python architecture_guard.py
Auditoria completa antes de commit/promoção: python project_audit.py

POLÍTICA DE VERSÃO E DOCUMENTAÇÃO
================================
- Toda implementação nova promovida continua na V14.
- Não abrir uma nova versão de runtime apenas para adicionar mecanismo novo.
- Toda alteração relevante da V14 deve atualizar README.md/README.txt, GPU_README.txt e PROJECT_STATE_2026-08-21.md no mesmo conjunto de mudanças.
- Testes, benchmark e SHA256SUMS.txt devem acompanhar o estado promovido.

RUNTIME PADRÃO
==============
O launcher padrão mantém V14 + tabelas Bagaço na GPU/VRAM durante uma sessão de prompts:

  RUN_GPU.bat

A partir da pasta TESTE:

  RUN_AI_GPU.bat

A sessão abre "Prompt>" e permanece carregada até /sair.

Linha de comando, para uma única execução:

  python run_gpu.py --prompt "Escreva 2000 caracteres sobre exploração espacial, tecnologia e futuro."

Prompt em arquivo:

  python run_gpu.py --prompt-file prompt_example_v14.txt --target-chars 2000 --output saida.txt

Fatos simbólicos próprios:

  python run_gpu.py --facts meus_fatos.json --output saida.txt

Teste sintético:

  python run_gpu.py --smoke

PROMPT + RULEVM V6 + ARGUMENT PLANNER V14
=========================================
O caminho promovido de prompt é:

  prompt -> conceitos explícitos -> learner de associações em CUDA -> RuleBank V6
         -> RuleVM indexado -> Evidence Argument Planner V14
         -> abertura/desenvolvimento/síntese -> V14 CUDA -> verificadores -> texto

Características:
- autonomous_rule_vm_v6.py aprende associações a partir das tabelas Bagaço; o VM apenas executa regras aprendidas.
- PMI positivo e ranking de candidatos são calculados em lote na GPU.
- O RuleBank contém source/target/predicate/kind/confidence/support/score/evidence.
- argument_planner_v14.py não inventa causalidade ou fatos: ordena e filtra somente regras já aprendidas/provadas.
- O planner usa confiança, suporte, score, profundidade, cobertura dos conceitos e alinhamento com o contexto global do prompt.
- As fases discursivas são explícitas e auditáveis: opening -> development -> synthesis.
- Conceitos compostos observados no corpus usam contexto da expressão completa; não caem silenciosamente para a última palavra ambígua.
- Contextos compostos preservam a expressão observada completa, em vez de promover um token solto como conceito autônomo.
- Expressões compostas p3-p5 são consultadas sob demanda e cacheadas na sessão.
- O filtro contextual pode rejeitar sentidos locais incompatíveis com o restante do prompt; por exemplo, em contexto musical, "composição musical" supera expansões químicas/corporais quando o corpus fornece esse alinhamento.
- Associação isolada sem corroboração suficiente pode ser descartada; o sistema prefere evidence_limited a preencher o texto com ruído.
- O refinador pode buscar regras fortes adicionais quando a desambiguação abriu espaço, sem baixar os thresholds de evidência.
- Regras fracas não são admitidas somente para atingir um número de caracteres.
- O RuleVM continua indexado por source e executa em dezenas de microssegundos.

SAÍDA LEGÍVEL
=============
IDs e000/a000/v000/r000 continuam internos para prova e verificação.
Tela/arquivo mostram palavras e frases por padrão.
--raw-slots mostra a representação interna somente para depuração.

O V14 mantém:
- Counter de fatos representados;
- ProtectedSlotVerifier;
- SemanticTraceVerifier;
- seleção p2-p5 CUDA;
- planejamento de grafo/parágrafo;
- Evidence Argument Planner;
- diversidade/repetição;
- superfície lexicalizada;
- preservação de evidência e provenance.

PERFORMANCE / VALIDAÇÃO ATUAL
=============================
Hardware auditado: NVIDIA GeForce GTX 1660 SUPER 6 GB, PyTorch 2.10.0+cu128, CUDA 12.8.

Bateria Prompt-RuleVM-V6-ArgumentPlanner-V14, 6 temas, alvo 1000 caracteres:
- exploração espacial: 957 caracteres, dentro da tolerância;
- energia solar: 1019 caracteres, dentro da tolerância;
- agricultura sustentável: 949 caracteres, dentro da tolerância;
- música clássica: 953 caracteres, dentro da tolerância;
- segurança digital: 223 caracteres, evidence_limited=true por falta de evidência forte suficiente;
- saúde pública: 1032 caracteres, dentro da tolerância.

Gates da bateria:
- semantic_verified: 6/6;
- slot_errors: 0;
- trace_errors: 0;
- IDs internos expostos: 0;
- cobertura dos conceitos do prompt: 100%;
- sentenças exatamente duplicadas: 0;
- repetição imediata do mesmo template: 0;
- fases argumentativas monotônicas: 6/6;
- prompts dentro da tolerância: 5/6;
- falhas de tamanho não explicadas: 0;
- regras removidas pelo filtro contextual na bateria: 9.

Performance medida na bateria final:
- carga inicial do modelo/GPU: ~3,235 s, uma vez por sessão;
- RuleVM máximo: ~0,036 ms;
- Evidence Argument Planner máximo: ~0,861 ms;
- raciocínio dos prompts posteriores: média ~17,4 ms, máximo ~23,3 ms;
- sessão persistente evita recarregar ~3-4 s de modelo por prompt.

Regressão RuleVM V5 de referência:
- 150.000 treino / 60.000 validação / 100.000 teste;
- transition accuracy: 1.0;
- closure exact: 1.0;
- proof validity: 1.0;
- 12/12 relações certificadas;
- generalidade: 6/6 mundos com transition/closure/proof = 1.0;
- drift: relações alteradas 1,5,9 detectadas exatamente; falsas revisões = 0.

TESTES
======
  python -m unittest -v test_v14_prompt.py test_rulevm_v6_prompt.py test_argument_planner_v14.py
  python benchmark_prompt_rulevm_v6.py
  python autonomous_rule_vm_v5.py
  python generalize_rule_vm_v5.py
  python rule_vm_drift_v5.py
  python architecture_guard.py
  python project_audit.py

ARQUIVOS CENTRAIS
=================
- procedural_runtime_gpu.py: backend CUDA discreto
- procedural_runtime_v12.py: planejamento/discurso de base
- procedural_runtime_v13.py: scorer lexicalizado
- procedural_runtime_v14.py: renderer promovido V14
- prompt_runtime_v14.py: interpretação, RuleVM V6 e integração do Argument Planner
- argument_planner_v14.py: planejamento argumentativo baseado somente em evidência aprendida
- prompt_session_v14.py: sessão persistente GPU/VRAM
- autonomous_rule_vm_v5.py: RuleVM MDL certificada para transições
- autonomous_rule_vm_v6.py: RuleBank de associações para conteúdo de prompt
- benchmark_prompt_rulevm_v6.py: generalidade/qualidade/performance multi-tema
- test_v14_prompt.py, test_rulevm_v6_prompt.py e test_argument_planner_v14.py: regressões automáticas
- architecture_guard.py: gate não neural
- project_audit.py: compilação, arquitetura, JSON/XZ, manifesto, launchers/configs e temporários

AUDITORIA DO PROJETO
====================
Antes de commit/promoção, execute:
  python project_audit.py

O auditor verifica a árvore versionável: sintaxe Python, regras arquiteturais, JSON/XZ,
SHA256SUMS, runtime V14/RuleVM V6, launcher persistente, documentação obrigatória e ausência
de artefatos temporários. Os modos --prompt, --prompt-file, --facts e --smoke são mutuamente
exclusivos para impedir combinações de entrada ambíguas.

Princípio central:
"Melhorar o mecanismo que aprende; nunca ensinar a resposta que ele deve aprender."
