AI-Procedural / Bagaço Renderer V14 + Learned RuleVM V6

REGRA ARQUITETURAL PERMANENTE
=============================
Este projeto é deliberadamente NÃO NEURAL.

É proibido introduzir redes neurais, neurônios, embeddings treináveis, backpropagation,
gradient descent, autograd de aprendizagem, torch.nn/optimizers ou regras de domínio
codificadas manualmente para resolver benchmarks.

PyTorch/CUDA é permitido SOMENTE como backend tensorial para computação discreta,
estatística e busca vetorizada. O conhecimento deve permanecer explícito e auditável:
contagens, fatos, grafos, episódios, regras induzidas, truth tables, RuleBanks, índices,
hipóteses, evidências, confiança e provas.

Regras completas: PROJECT_RULES.md
Estado técnico: PROJECT_STATE_2026-08-21.md
Guard obrigatório antes de promoção: python architecture_guard.py

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

PROMPT + RULEVM V6
==================
O caminho promovido de prompt é:

  prompt -> conceitos explícitos -> learner de associações em CUDA -> RuleBank V6
         -> RuleVM indexado -> grafo semântico -> V14 CUDA -> verificadores -> texto

Características:
- autonomous_rule_vm_v6.py aprende associações a partir das tabelas Bagaço; o VM apenas executa as regras aprendidas.
- PMI positivo e ranking de candidatos são calculados em lote na GPU.
- O RuleBank contém source/target/predicate/kind/confidence/support/score/evidence.
- Conceitos compostos observados no corpus usam contexto da expressão completa; não caem silenciosamente para a última palavra ambígua.
- Busca de p3-p5 para conceitos compostos é sob demanda e cacheada na sessão.
- Regras fracas não são admitidas somente para atingir um número de caracteres.
- Quando a evidência é insuficiente, o relatório usa evidence_limited=true e preserva qualidade/semântica.
- O RuleVM é indexado por source; execução observada na bateria V6 ficou em dezenas de microssegundos.

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
- diversidade/repetição;
- superfície lexicalizada.

PERFORMANCE / VALIDAÇÃO ATUAL
=============================
Hardware auditado: NVIDIA GeForce GTX 1660 SUPER 6 GB, PyTorch 2.10.0+cu128, CUDA 12.8.

Bateria Prompt-RuleVM-V6, 6 temas:
- semantic_verified: 6/6
- slot_errors: 0
- trace_errors: 0
- IDs expostos: 0
- cobertura dos conceitos do prompt: 100%
- razão média de sentenças únicas: 100%
- RuleVM: máximo observado ~0,04 ms
- raciocínio de prompts posteriores na sessão: tipicamente poucos/dezenas de ms
- prompts com corpus esparso podem ficar abaixo do tamanho solicitado; isso é sinalizado como evidence_limited.

Regressão RuleVM V5 executada após o V6:
- 150.000 treino / 60.000 validação / 100.000 teste
- aprendizagem: ~1,075 s
- transition accuracy: 1.0
- closure exact: 1.0
- proof validity: 1.0
- 12/12 relações certificadas
- generalidade: 6/6 mundos com transition/closure/proof = 1.0 (tempo total ~5,498 s)
- drift: relações alteradas 1,5,9 detectadas exatamente; falsas revisões = 0

TESTES
======
  python -m unittest -v test_v14_prompt.py test_rulevm_v6_prompt.py
  python benchmark_prompt_rulevm_v6.py
  python autonomous_rule_vm_v5.py
  python generalize_rule_vm_v5.py
  python rule_vm_drift_v5.py
  python architecture_guard.py
  python project_audit.py

ARQUIVOS CENTRAIS
=================
- procedural_runtime_gpu.py: backend CUDA discreto
- procedural_runtime_v12.py: planejamento/discurso
- procedural_runtime_v13.py: scorer lexicalizado
- procedural_runtime_v14.py: renderer promovido
- prompt_runtime_v14.py: interpretação de prompt e plano V6
- prompt_session_v14.py: sessão persistente GPU/VRAM
- autonomous_rule_vm_v5.py: RuleVM MDL certificada para transições
- autonomous_rule_vm_v6.py: RuleBank de associações para conteúdo de prompt
- benchmark_prompt_rulevm_v6.py: generalidade/performance em temas variados
- test_v14_prompt.py e test_rulevm_v6_prompt.py: regressões automáticas
- architecture_guard.py: gate não neural
- project_audit.py: compilação, arquitetura, JSON/XZ, manifesto, launchers/configs e temporários

Princípio central:
"Melhorar o mecanismo que aprende; nunca ensinar a resposta que ele deve aprender."

AUDITORIA DO PROJETO
===================
Antes de commit/promoção, execute:
  python project_audit.py

O auditor verifica toda a árvore versionável: sintaxe Python, regras arquiteturais, JSON/XZ,
SHA256SUMS, runtime V14/RuleVM V6, launcher persistente, documentação obrigatória e ausência
de artefatos temporários. Os modos --prompt, --prompt-file, --facts e --smoke são mutuamente
exclusivos para impedir combinações de entrada ambíguas.

O RuleBank V6 reserva as observações explícitas do próprio prompt mesmo quando o orçamento
de regras aprendidas está saturado; contexto corpus é usado como expansão, nunca para apagar
os conceitos solicitados pelo usuário.
