# Fisheye Object Detection

Pesquisa em visão computacional para **detecção de objetos em imagens fisheye**, com ênfase na qualidade das detecções, no custo computacional e na viabilidade de execução em dispositivos de borda.

O projeto compara **YOLO11s, YOLO11m, YOLO11l e RT-DETR-L** no **WoodScape**, incluindo métricas tradicionais de detecção e uma avaliação operacional de verdadeiros positivos, falsos positivos e falsos negativos. A etapa seguinte investiga **transferência de aprendizado para ambientes de mineração**, usando o **Mines Dataset v4**. Uma linha complementar estuda **HOG, PCA e One-Class SVM** sobre recortes de objetos anotados, ainda sem detecção completa.

## Estado da pesquisa

| Frente | Situação |
| --- | --- |
| Preparação do WoodScape e divisão agrupada por captura | Concluída nos experimentos originais |
| Treinamento YOLO11s, YOLO11m, YOLO11l e RT-DETR-L no WoodScape | Concluído |
| Avaliação padrão e operacional no teste do WoodScape | Concluída |
| Inspeção e preparação do Mines Dataset v4 (`mines_clean`) | Concluída |
| Fine-tuning no Mines: M1 (YOLO11m inicializado no COCO) | Iniciado; resultados pendentes |
| Fine-tuning no Mines: M2 e M3 (inicializados no WoodScape) | Planejado |
| Extração de recortes da classe `person` | Teste exploratório concluído (50 recortes) |
| HOG e PCA / One-Class SVM | Pendentes |
| Recuperação do manifesto da divisão original do WoodScape | Pendente |

> **Reprodutibilidade:** os resultados do WoodScape abaixo correspondem aos conjuntos de dados locais usados nos experimentos. É necessário preservar/recuperar os identificadores exatos de `train`, `val` e `test` para reproduzir a divisão agrupada por captura. A mesma proporção ou semente, isoladamente, não garante a mesma composição dos conjuntos. O teste não deve ser usado para selecionar limiares de confiança ou hiperparâmetros.

## Dataset: WoodScape

Fonte: [ValeoAI / WoodScape](https://github.com/valeoai/WoodScape). O conjunto utilizado contém **8.234 imagens** com anotações 2D de cinco classes:

| ID | Classe |
| ---: | --- |
| 0 | `vehicles` |
| 1 | `person` |
| 2 | `bicycle` |
| 3 | `traffic_light` |
| 4 | `traffic_sign` |

A divisão empregada nos baselines foi:

| Conjunto | Imagens | Objetos anotados |
| --- | ---: | ---: |
| Treinamento | 6.587 | 57.719 |
| Validação | 823 | 7.118 |
| Teste | 824 | 7.226 |
| **Total** | **8.234** | **72.063** |

As imagens são identificadas por captura e câmera (`FV`, `RV`, `MVL`, `MVR`). Na preparação dos dados para os baselines, imagens relacionadas à mesma captura foram agrupadas para reduzir vazamentos entre os conjuntos.

**Os dados do WoodScape e os recortes derivados não são distribuídos neste repositório.** Faça o download pelas fontes autorizadas e respeite as condições de uso.

### Formato das anotações

Os arquivos originais de bounding boxes possuem registros no formato:

```text
classe,class_id,xmin,ymin,xmax,ymax
```

Exemplo:

```text
vehicles,0,11,387,250,570
person,1,865,295,915,375
```

O script de preparação converte as coordenadas para o formato YOLO (`class_id x_center y_center width height`, normalizado entre 0 e 1).

## Instalação

Recomenda-se criar um ambiente virtual Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O arquivo `requirements.txt` reúne as dependências dos pipelines de deep learning e aprendizado de máquina clássico. As versões específicas utilizadas nos treinamentos originais estão registradas abaixo; para reprodução exata, também devem ser preservados os artefatos e o ambiente de treinamento.

## Estrutura do repositório

```text
fisheye-object-detection/
├── classical_ml/
│   └── extract_crops.py           # Recortes de objetos anotados
├── scripts/
│   ├── prepare_woodscape.py       # Preparação do WoodScape
│   ├── inspect_mines_rare_classes.py  # Inspeção visual de classes raras
│   └── prepare_mines_dataset.py   # Limpeza, remapeamento e auditoria do Mines
├── training/
│   ├── train_rtdetr.py
│   └── train_yolo11.py
├── evaluation/
│   ├── evaluate_model.py
│   └── evaluate_operational.py
├── docs/                           # Documentação dos experimentos
├── datasets/                       # Dados locais, não versionados
│   ├── woodscape_raw/
│   ├── woodscape/
│   ├── mines/
│   └── mines_clean/
├── runs/                           # Treinamentos e avaliações locais
├── requirements.txt
└── README.md
```

**Não versionar:** imagens, anotações de datasets, checkpoints (`.pt`), caches, vídeos e saídas volumosas de `runs/`. Os scripts de preparação e os comandos descritos aqui permitem reproduzir o processamento localmente.

## Baselines de deep learning

### Preparação dos dados

Organize os arquivos originais da seguinte forma:

```text
datasets/woodscape_raw/
├── rgb_images/
└── box_2d_annotations/
    ├── 00000_FV.txt
    └── ...
```

O script aceita os argumentos `--raw`, `--output`, `--train`, `--val`, `--seed` e `--mode`:

```bash
python scripts/prepare_woodscape.py \
  --raw datasets/woodscape_raw \
  --output datasets/woodscape
```

**Atenção:** este comando pode produzir uma divisão diferente da utilizada nos experimentos publicados. Antes de reinterpretar resultados ou executar novas comparações, recupere e confira os arquivos originais de cada split.

### Treinamento

Os experimentos originais foram executados com **NVIDIA GeForce RTX 5070 (~12 GB VRAM)**, Python 3.14.4, PyTorch 2.14.0+cu130 e Ultralytics 8.4.147.

| Parâmetro | RT-DETR-L | YOLO11m |
| --- | --- | --- |
| Pesos iniciais | `rtdetr-l.pt` | `yolo11m.pt` |
| Épocas máximas | 100 | 100 |
| Resolução de entrada | 640 | 640 |
| Batch size | 8 | 8 |
| Otimizador | AdamW | AdamW |
| Learning rate | 0.001 | 0.001 |
| Weight decay | 0.0005 | 0.0005 |
| Warm-up | 3 épocas | 3 épocas |
| Patience | 20 | 20 |
| Seed | 42 | 42 |
| AMP | Ativado | Ativado |

Execução:

```bash
python training/train_rtdetr.py
python training/train_yolo11.py
```

Ambos chegaram à época 91 e foram interrompidos por *early stopping*, com melhor checkpoint na época 71. No YOLO11m, houve uma interrupção por `CUDA_ERROR_LAUNCH_TIMEOUT` durante a época 27; o treinamento foi retomado a partir de `last.pt`.

### Avaliação dos baselines

O script `evaluation/evaluate_model.py` utiliza `split="test"` para a avaliação final. Exemplos com os caminhos empregados nos experimentos originais:

```bash
python evaluation/evaluate_model.py \
  --model-type rtdetr \
  --weights runs/rtdetr/woodscape_test-2/weights/best.pt \
  --name rtdetr_l_woodscape_test

python evaluation/evaluate_model.py \
  --model-type yolo \
  --weights runs/yolo11/woodscape_yolo11m_e100_b8/weights/best.pt \
  --name yolo11m_woodscape_test
```

Os checkpoints não estão incluídos no GitHub; esses comandos exigem os arquivos locais de pesos e o dataset preparado.

## Resultados de deep learning — WoodScape

### Validação dos baselines originais

A tabela a seguir preserva os números históricos dos dois primeiros experimentos, antes da inclusão de YOLO11s e YOLO11l.

| Métrica | RT-DETR-L | YOLO11m |
| --- | ---: | ---: |
| Precision | 0.655 | 0.672 |
| Recall | 0.593 | 0.580 |
| mAP@50 | 0.618 | 0.620 |
| mAP@50:95 | 0.394 | 0.408 |
| Inferência | 5.8 ms | 2.7 ms |
| Parâmetros | 31.99 M | 20.03 M |
| GFLOPs | 105.4 | 67.8 |
| Tamanho do modelo | 66.2 MB | 40.5 MB |

### Teste independente: quatro detectores

Avaliação padrão da Ultralytics nas **824 imagens de teste** do WoodScape (7.226 objetos anotados). Os tempos de inferência são os reportados nas respectivas execuções e dependem do ambiente e das condições de medição.

| Modelo | Precision | Recall | mAP@50 | mAP@50:95 | Inferência (ms/imagem) |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11s | 0.6548 | 0.5462 | 0.5894 | 0.3651 | 2.07 |
| YOLO11m | 0.6800 | 0.5604 | 0.6163 | 0.3976 | 4.70 |
| YOLO11l | 0.6650 | **0.5950** | **0.6300** | **0.4120** | 5.90 |
| RT-DETR-L | **0.6941** | 0.5876 | 0.6270 | 0.4012 | 10.01 |

O YOLO11l obteve o maior mAP@50:95 no teste, enquanto o YOLO11s apresentou o menor tempo de inferência registrado. Diferenças pequenas de métricas não demonstram, por si sós, superioridade estatisticamente significativa.

**mAP@50:95 por classe** (resultados disponíveis para YOLO11m, YOLO11l e RT-DETR-L):

| Classe | YOLO11m | YOLO11l | RT-DETR-L |
| --- | ---: | ---: | ---: |
| `vehicles` | 0.612 | 0.615 | 0.598 |
| `person` | 0.441 | 0.462 | 0.433 |
| `bicycle` | 0.309 | 0.312 | 0.317 |
| `traffic_light` | 0.288 | 0.306 | 0.301 |
| `traffic_sign` | 0.338 | 0.365 | 0.358 |

### Avaliação operacional

O script `evaluation/evaluate_operational.py` realiza correspondência por classe entre predições e anotações (*matching IoU* de 0,50), contabilizando TP, FP e FN. O parâmetro `--nms-iou` controla a supressão de predições sobrepostas e **não** o limiar de correspondência.

Os limiares de confiança foram escolhidos **na validação** para maximizar o F1 micro global e depois aplicados ao teste, sem ajuste baseado no teste:

| Modelo | Confiança selecionada | F1 micro (teste) |
| --- | ---: | ---: |
| YOLO11s | 0.25 | 0.6729 |
| YOLO11m | 0.40 | 0.7054 |
| YOLO11l | 0.40 | **0.7079** |
| RT-DETR-L | 0.50 | 0.6889 |

No YOLO11l, o limiar 0,40 obteve F1 micro **0,7151 na validação** (contra 0,7134 para 0,25 e 0,7038 para 0,50). No teste, o YOLO11l registrou P micro 0,7676, R micro 0,6568 e F1 micro 0,7079.

**Classe `person` no teste operacional:** o YOLO11l, com confiança 0,40, registrou TP=1.063, FP=282, FN=656, precisão=0,7903, recall=0,6184 e F1=0,6939. O recall operacional de `person` registrado para RT-DETR-L foi 0,7190 e, para YOLO11m, 0,5945. Portanto, a classificação pelo F1 global não substitui a análise específica de falsos negativos de pessoas.

Exemplo de reprodução da avaliação operacional do YOLO11l:

```bash
python evaluation/evaluate_operational.py \
  --model-type yolo \
  --weights runs/yolo11/woodscape_yolo11l_e100_b8/weights/best.pt \
  --data-dir datasets/woodscape \
  --name woodscape_yolo11l_test_conf0.40 \
  --split test --conf 0.40 --match-iou 0.50 --nms-iou 0.70 \
  --imgsz 640 --batch 8 --device 0
```

### Observações de qualidade dos dados

Durante os treinamentos originais, a Ultralytics identificou e removeu uma anotação duplicada em `03404_RV.png` (treinamento) e outra em `01866_MVL.png` (validação). Uma melhoria prevista é tratar essas duplicações durante a preparação.

## Dataset: Mines v4 e preparação do Mines Clean

Fonte: [Mines Dataset v4 — Roboflow Universe](https://universe.roboflow.com/vikasse2016/mines-f4fjb/dataset/4) (licença indicada no arquivo de origem: **CC BY 4.0**). O dataset foi exportado em formato YOLO, com **5.350 imagens** e **14 classes originais**.

A inspeção encontrou variantes de nomenclatura (`dumb_truck`, `dunp_truck`, `truck`) representando caminhões basculantes e **22 anotações poligonais** misturadas às caixas de detecção. O processamento foi realizado sem alterar os arquivos originais.

### Preparação reproduzível

Organize o download original em `datasets/mines/`, com `data.yaml` e os diretórios `train/`, `valid/` e `test/` (cada um contendo `images/` e `labels/`). Execute:

```bash
python scripts/inspect_mines_rare_classes.py
python scripts/prepare_mines_dataset.py \
  --source datasets/mines \
  --dest datasets/mines_clean
```

O script de preparação:

- converte polígonos em caixas delimitadoras, preservando as anotações de detecção;
- unifica `dumb_truck`, `dunp_truck` e `truck` em `dump_truck`, conforme inspeção visual das ocorrências raras;
- remapeia as classes para sete IDs consecutivos;
- separa **a imagem inteira** para revisão se houver `bull_dozer`, `front_end_loader`, `hip_of_soil` ou `road_sign`, evitando deixar esses objetos como fundo sem anotação;
- preserva os splits de origem, sem transferir imagens entre treino, validação e teste;
- produz `data.yaml`, `audit.json` e `review_images.csv` em `datasets/mines_clean/`.

O destino deve estar ausente antes da execução; o script não sobrescreve um dataset já preparado. O `data.yaml` gerado contém um caminho absoluto local em `path:` e precisa ser regenerado ou ajustado se o projeto for movido para outra máquina.

### Taxonomia do Mines Clean

| ID | Classe |
| ---: | --- |
| 0 | `dump_truck` |
| 1 | `excavator` |
| 2 | `grader` |
| 3 | `light_vehicle` |
| 4 | `person` |
| 5 | `safety_helmet` |
| 6 | `safety_jacket` |

### Resultado da preparação (15/10/2026)

| Split | Imagens originais | Imagens limpas | Para revisão | Polígonos convertidos | Instâncias limpas |
| --- | ---: | ---: | ---: | ---: | ---: |
| Treino | 3.806 | 3.802 | 4 | 13 | 9.901 |
| Validação | 1.037 | 1.037 | 0 | 7 | 1.530 |
| Teste | 507 | 507 | 0 | 2 | 881 |
| **Total** | **5.350** | **5.346** | **4** | **22** | **12.312** |

As quatro imagens separadas para revisão pertencem ao treino; nenhuma foi retirada da validação ou do teste. A auditoria detalhada permanece disponível localmente em `datasets/mines_clean/audit.json`. A licença e a atribuição da fonte original devem ser respeitadas em qualquer redistribuição autorizada.

### Experimentos de transferência para mineração

| Experimento | Inicialização | Dataset de ajuste | Estado |
| --- | --- | --- | --- |
| M1 | `yolo11m.pt` (COCO) | Mines Clean | Iniciado; métricas pendentes |
| M2 | YOLO11m ajustado no WoodScape | Mines Clean | Planejado |
| M3 | YOLO11l ajustado no WoodScape | Mines Clean | Planejado |

A comparação **M1 × M2** isola melhor o efeito do treinamento intermediário no WoodScape, mantendo arquitetura, splits e hiperparâmetros equivalentes. Como os conjuntos têm taxonomias distintas (WoodScape: cinco classes; Mines Clean: sete), a camada de saída do detector é adaptada no fine-tuning. A avaliação final deve ser realizada somente no teste do Mines Clean, após a seleção de hiperparâmetros na validação.

Comando usado para iniciar o M1:

```bash
yolo detect train \
  model=yolo11m.pt \
  data=datasets/mines_clean/data.yaml \
  epochs=100 imgsz=640 batch=8 device=0 \
  optimizer=AdamW lr0=0.001 seed=42 patience=100 \
  project=runs/yolo11 name=mines_yolo11m_coco_e100_b8 exist_ok=False
```

**Limitação:** o Mines contém imagens de domínio diferente do WoodScape, que usa câmeras fisheye. O desempenho no Mines não demonstra, isoladamente, robustez a distorções fisheye ou condições reais de operação em caminhões de mina.

## E5 — Aprendizado de máquina clássico (em desenvolvimento)

O experimento E5 investiga se uma representação visual mais compacta pode contribuir para reconhecer objetos de interesse em imagens fisheye. A proposta inicial considera:

1. Extração de recortes a partir das bounding boxes originais do WoodScape;
2. Descritores HOG (*Histogram of Oriented Gradients*);
3. Padronização das características (`StandardScaler`);
4. Redução dimensional com PCA (*Principal Component Analysis*);
5. Modelagem da classe `person` com One-Class SVM.

### E5.1 — Extração de recortes (concluída em amostra exploratória)

O script `classical_ml/extract_crops.py` gera recortes e um manifesto CSV com nome do recorte, imagem e captura originais, classe e coordenadas da bounding box.

```bash
python classical_ml/extract_crops.py \
  --raw datasets/woodscape_raw \
  --output datasets/woodscape_crops_test \
  --class-name person \
  --max-crops 50
```

**Resultado observado em 13/10/2026:** 50 recortes da classe `person`, nenhuma anotação inválida, nenhum arquivo de anotação sem imagem correspondente e inspeção visual satisfatória dos recortes. O manifesto foi criado em `datasets/woodscape_crops_test/person_manifest.csv`.

Este é um **teste funcional exploratório**, não uma avaliação de desempenho do classificador. Os 50 recortes ainda não foram separados por `train`, `val` e `test`.

### E5.2 — HOG + PCA (planejado)

Será implementada a extração de descritores HOG dos recortes, seguida de padronização e PCA para avaliar dimensionalidade e variância explicada. **Ainda não há implementação versionada nem resultados publicados para esta etapa.**

### E5.3 — One-Class SVM (planejado)

O classificador será ajustado sobre características da classe `person`. Outras classes poderão ser utilizadas como exemplos de comparação na avaliação, mantendo os dados de teste fora do ajuste do modelo e do pré-processamento.

### Limitações e protocolo futuro

- A entrada inicial do E5 são **recortes previamente localizados pelas anotações**, e não imagens completas. Portanto, E5 ainda não é um detector end-to-end.
- A comparação direta de mAP e de tempo total de detecção com YOLO11m e RT-DETR-L **não é válida nesta fase**: o E5 não inclui a etapa de localização dos objetos.
- Antes de calcular métricas finais, será necessário recuperar os splits originais, prevenir vazamento entre capturas e ajustar `StandardScaler`, PCA e One-Class SVM apenas com dados de treinamento.
- Variância explicada pelo PCA não equivale a desempenho de classificação. O desempenho será analisado em dados independentes e com métricas adequadas à tarefa proposta.

## Próximos passos

- [x] Preparar o WoodScape e avaliar YOLO11s, YOLO11m, YOLO11l e RT-DETR-L no teste.
- [x] Executar avaliação operacional com seleção de confiança na validação.
- [x] Inspecionar e preparar o Mines Clean com auditoria das anotações.
- [ ] Concluir M1 e executar M2/M3; comparar generalização e transferência de aprendizado.
- [ ] Investigar falsos negativos da classe `person` e classes pouco representadas no Mines.
- [x] Extrair e inspecionar uma amostra de 50 recortes `person` do WoodScape.
- [ ] Recuperar e registrar o manifesto da divisão original do WoodScape.
- [ ] Implementar e testar HOG + PCA e One-Class SVM.
- [ ] Medir custo de CPU, memória e desempenho em dispositivos de borda.

## Referências

- [WoodScape — ValeoAI](https://github.com/valeoai/WoodScape)
- [Mines Dataset v4 — Roboflow Universe](https://universe.roboflow.com/vikasse2016/mines-f4fjb/dataset/4)
- [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)

## Autor

Matheus Higino — projeto de pesquisa em visão computacional, detecção de objetos com câmeras fisheye e avaliação de custo computacional.
