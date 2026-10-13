# Fisheye Object Detection

Pesquisa em visão computacional para **detecção de objetos em imagens fisheye**, com ênfase na qualidade das detecções, no custo computacional e na viabilidade de execução em dispositivos de borda.

O projeto parte de dois detectores de objetos baseados em deep learning — **RT-DETR-L** e **YOLO11m** — avaliados sobre o **WoodScape**. Uma linha experimental complementar investiga **características HOG, redução dimensional com PCA e One-Class SVM**. Essa segunda linha ainda está em desenvolvimento e, inicialmente, trabalha com recortes de objetos anotados, não com detecção completa.

## Estado da pesquisa

| Frente | Situação |
| --- | --- |
| Preparação do WoodScape e divisão agrupada por captura | Concluída nos experimentos originais |
| Treinamento RT-DETR-L e YOLO11m | Concluído |
| Avaliação dos dois detectores no conjunto de teste | Concluída |
| Extração de recortes da classe `person` | Teste exploratório concluído (50 recortes) |
| HOG e PCA | Pendente |
| One-Class SVM | Pendente |
| Recuperação do manifesto da divisão original | Pendente |

> **Reprodutibilidade:** os dois detectores foram avaliados usando a mesma divisão original de dados. Antes de executar uma nova avaliação comparativa, será necessário recuperar os identificadores exatos de `train`, `val` e `test` da máquina em que os baselines foram treinados. A mesma proporção ou semente, isoladamente, não garante a mesma composição dos conjuntos.

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
│   └── extract_crops.py          # Recortes de objetos anotados (implementado)
├── scripts/
│   └── prepare_woodscape.py      # Conversão e preparação do WoodScape
├── training/
│   ├── train_rtdetr.py
│   └── train_yolo11.py
├── evaluation/
│   └── evaluate_model.py
├── docs/                          # Relatórios dos experimentos
├── datasets/                      # Dados locais, não versionados
├── runs/                          # Saídas locais, não versionadas
├── requirements.txt
└── README.md
```

Arquivos de dataset, checkpoints, saídas de treinamento e recortes não devem ser commitados.

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

## Resultados dos baselines

### Validação

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

### Teste independente

Resultados dos melhores checkpoints nas **824 imagens de teste** (7.226 objetos anotados):

| Métrica | RT-DETR-L | YOLO11m |
| --- | ---: | ---: |
| **Precision** | **0.6941** | 0.6800 |
| **Recall** | **0.5876** | 0.5604 |
| **mAP@50** | **0.6270** | 0.6163 |
| **mAP@50:95** | **0.4012** | 0.3976 |
| Inferência | 10.01 ms | **4.70 ms** |
| Parâmetros | 31.99 M | **20.03 M** |
| GFLOPs | 105.4 | **67.8** |

O RT-DETR-L obteve métricas agregadas de detecção ligeiramente superiores no teste, enquanto o YOLO11m teve menor custo de inferência na medição registrada. As latências são específicas do ambiente e das condições de avaliação utilizadas.

### mAP@50:95 por classe no teste

| Classe | RT-DETR-L | YOLO11m |
| --- | ---: | ---: |
| `vehicles` | 0.598 | **0.612** |
| `person` | 0.433 | **0.441** |
| `bicycle` | **0.317** | 0.309 |
| `traffic_light` | **0.301** | 0.288 |
| `traffic_sign` | **0.358** | 0.338 |

Resultados detalhados por classe e outras métricas estão documentados nos relatórios em `docs/`.

### Observações de qualidade dos dados

Durante o treinamento, a Ultralytics identificou e removeu uma anotação duplicada em `03404_RV.png` (treinamento) e outra em `01866_MVL.png` (validação). Uma melhoria prevista é tratar essas duplicações durante a preparação.

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

- [x] Preparar o WoodScape e executar os baselines RT-DETR-L e YOLO11m.
- [x] Avaliar ambos os detectores no conjunto independente de teste.
- [x] Extrair e inspecionar visualmente uma amostra de 50 recortes `person`.
- [ ] Recuperar e registrar os identificadores da divisão original dos baselines.
- [ ] Implementar e testar HOG + PCA.
- [ ] Implementar e avaliar One-Class SVM com protocolo independente.
- [ ] Examinar falsos positivos, falsos negativos e objetos pequenos nos baselines.
- [ ] Medir custo de CPU, memória e tempo dos métodos clássicos.
- [ ] Estudar validação cruzada agrupada, aumento de dados e modelos menores.
- [ ] Investigar exportação e desempenho em dispositivos de borda.

## Referências

- [WoodScape — ValeoAI](https://github.com/valeoai/WoodScape)
- [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)

## Autor

Matheus Higino — projeto de pesquisa em visão computacional, detecção de objetos com câmeras fisheye e avaliação de custo computacional.
