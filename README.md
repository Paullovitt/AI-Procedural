# AI-Procedural â€” BagaÃ§o GPU Discourse Model

Este repositÃ³rio contÃ©m somente o modelo linguÃ­stico/discursivo nÃ£o neural treinado sobre BagaÃ§o2 e o validador mÃ­nimo do artefato.

## Arquitetura

O modelo nÃ£o usa rede neural, backpropagation nem gradient descent. O treinamento combina contagens discretas, Count-Min Sketch em GPU, histogramas estruturais e tabelas lexicais/discursivas persistentes.

Estrutura publicada:

- `model/full/`: modelo amplo treinado a partir dos shards disponÃ­veis do corpus.
- `model/quality/`: camada promovida de maior qualidade, treinada com filtro geral pelos metadados do prÃ³prio dataset.
- `validate_model.py`: verifica hashes, formato, ordenaÃ§Ã£o e invariantes do artefato.

## Treinamento amplo

Hardware usado: NVIDIA GeForce RTX 3050 6 GB.

Resumo do modelo amplo:

- 459 / 460 shards processados.
- shard ausente: Ã­ndice 109 (`001_00043.parquet`).
- 32.982.796 documentos.
- 90.377.289.911 caracteres.
- 2.061.433 documentos na leitura profunda.
- 42.688.638 sentenÃ§as.
- 931.938.051 tokens na leitura profunda.
- 257.676 documentos na mineraÃ§Ã£o lexical.
- 120.000 tokens persistidos.
- 120.000 bigramas.
- 100.000 trigramas.
- 70.000 4-grams.
- 50.000 5-grams.
- 100.000 aberturas.
- 100.000 fechamentos.
- 100.000 conectores.

O shard 459 foi mantido fora do fold para auditoria antes de ser incorporado ao artefato final. A distÃ¢ncia estrutural registrada foi 0,2004475735. Cobertura pesada no heldout: bigramas 0,63252; trigramas 0,23755; 4-grams 0,08395; 5-grams 0,03247; aberturas 0,63541; conectores 0,65945.

## Camada de qualidade

A camada `quality` usa um filtro geral de qualidade baseado somente nos metadados fornecidos pelo BagaÃ§o2:

- `educational_score >= 2`
- `ptpt_score >= 0.90`
- `language_score >= 0.99`

Na bateria usada para sua construÃ§Ã£o foram examinados 80 shards distribuÃ­dos pelo corpus, 5.696.334 documentos, dos quais 774.357 passaram pelo filtro e 96.885 foram amostrados para a mineraÃ§Ã£o lexical/discursiva.

O filtro reduz fortemente padrÃµes de spam/boilerplate que aparecem por frequÃªncia bruta no corpus amplo.

## ValidaÃ§Ã£o

Execute:

```bash
python validate_model.py
```

O script valida:

- presenÃ§a de todos os componentes;
- SHA-256 conforme o manifesto;
- JSONL e UTF-8;
- contagens esperadas do modelo amplo;
- ordenaÃ§Ã£o por frequÃªncia;
- estatÃ­sticas principais e shard ausente;
- consulta de sanidade para um trigram frequente.

## ObservaÃ§Ã£o

Os dados brutos do BagaÃ§o2 nÃ£o estÃ£o incluÃ­dos. Este repositÃ³rio contÃ©m somente o estado persistente do modelo treinado e o validador.

