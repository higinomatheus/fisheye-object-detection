# Fisheye Object Detection

Projeto de pesquisa para **detecção de objetos em imagens fisheye**, utilizando o dataset **WoodScape** e modelos de detecção em tempo real.

Nesta etapa da pesquisa foram avaliadas duas arquiteturas:

* **RT-DETR-L**
* **YOLO11m**

O objetivo é construir um pipeline reprodutível para preparação dos dados, treinamento, validação, teste e comparação de modelos em imagens obtidas por câmeras fisheye, considerando tanto a qualidade da detecção quanto o custo computacional.

---

## Objetivos

Este projeto busca:

* preparar o dataset WoodScape para detecção de objetos;
* converter as anotações originais para o formato utilizado pela Ultralytics;
* manter divisões fixas de `train`, `val` e `test`;
* treinar e avaliar diferentes arquiteturas de detecção;
* comparar RT-DETR e YOLO utilizando os mesmos dados;
* avaliar desempenho por classe;
* analisar custo computacional e velocidade de inferência;
* investigar posteriormente execução em dispositivos de borda.

As principais métricas de interesse são:

* Precision;
* Recall;
* mAP@50;
* mAP@50:95;
* número de parâmetros;
* GFLOPs;
* uso de VRAM;
* latência;
* FPS;
* tamanho do modelo.

---

## Dataset

O projeto utiliza o [WoodScape](https://github.com/valeoai/WoodScape), dataset desenvolvido pela Valeo para tarefas de visão computacional com câmeras fisheye.

Para a tarefa de detecção são utilizadas cinco classes:

| ID | Classe          |
| -: | --------------- |
|  0 | `vehicles`      |
|  1 | `person`        |
|  2 | `bicycle`       |
|  3 | `traffic_light` |
|  4 | `traffic_sign`  |

O dataset preparado neste projeto contém:

| Split      |   Imagens |    Objetos |
| ---------- | --------: | ---------: |
| Train      |     6.587 |     57.719 |
| Validation |       823 |      7.118 |
| Test       |       824 |      7.226 |
| **Total**  | **8.234** | **72.063** |

> Os dados do WoodScape não são versionados neste repositório. Faça o download diretamente das fontes oficiais e respeite os termos de uso e a licença do dataset.

---

## Estrutura do projeto

```text
fisheye-object-detection/
├── datasets/
│   ├── woodscape_raw/
│   │   ├── rgb_images/
│   │   └── box_2d_annotations/
│   │
│   └── woodscape/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── woodscape.yaml
│
├── scripts/
│   └── prepare_woodscape.py
│
├── training/
│   ├── train_rtdetr.py
│   └── train_yolo11.py
│
├── evaluation/
│   └── evaluate_model.py
│
├── docs/
│   ├── rtdetr_l_woodscape_e3_b8_full.tex
│   └── rtdetr_l_woodscape_e100_b8_summary.tex
│
├── runs/
├── requirements.txt
├── .gitignore
└── README.md
```

As pastas `datasets/`, `runs/` e os arquivos de pesos `.pt` não devem ser enviados para o GitHub.

---

## Formato original das anotações

As bounding boxes do WoodScape utilizadas neste projeto seguem o formato:

```text
classe,class_id,xmin,ymin,xmax,ymax
```

Exemplo:

```text
vehicles,0,11,387,250,570
vehicles,0,1005,358,1243,522
person,1,865,295,915,375
```

O script de preparação converte essas coordenadas para o formato utilizado pela Ultralytics:

```text
class_id x_center y_center width height
```

com todas as coordenadas normalizadas entre `0` e `1`.

---

## Ambiente Python

Crie um ambiente virtual:

```bash
python3 -m venv .venv
```

Ative o ambiente:

```bash
source .venv/bin/activate
```

Atualize o `pip`:

```bash
python -m pip install --upgrade pip
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Dependências principais:

```text
ultralytics
Pillow
PyYAML
```

---

## Preparação do WoodScape

Coloque os dados originais em:

```text
datasets/woodscape_raw/
├── rgb_images/
└── box_2d_annotations/
```

Depois execute:

```bash
python3 scripts/prepare_woodscape.py \
    --raw datasets/woodscape_raw \
    --output datasets/woodscape
```

O script:

1. localiza imagens e anotações correspondentes;
2. lê as bounding boxes originais;
3. converte as coordenadas para o formato YOLO/Ultralytics;
4. agrupa imagens relacionadas pela captura;
5. divide os dados em `train`, `val` e `test`;
6. cria a estrutura de diretórios;
7. gera o arquivo `woodscape.yaml`.

Resultado obtido:

```text
Imagens encontradas: 8234
Anotações encontradas: 8234
Pares válidos: 8234

train: 6587 imagens | 57719 objetos
val:    823 imagens |  7118 objetos
test:   824 imagens |  7226 objetos
```

---

## Estratégia de divisão

As imagens possuem identificadores associados às diferentes câmeras:

```text
FV
RV
MVL
MVR
```

Exemplos:

```text
04346_FV.png
04346_RV.png
04346_MVL.png
04346_MVR.png
```

Durante a divisão do dataset, imagens associadas à mesma captura são agrupadas para reduzir o risco de vazamento de informação entre treinamento, validação e teste.

A divisão foi mantida fixa durante os experimentos com RT-DETR-L e YOLO11m.

---

## Hardware utilizado

Os experimentos foram executados com:

```text
GPU: NVIDIA GeForce RTX 5070
VRAM: ~12 GB
PyTorch: 2.14.0+cu130
CUDA PyTorch: 13.0
Python: 3.14.4
Ultralytics: 8.4.147
```

---

# Treinamento

## RT-DETR-L

O primeiro baseline completo utiliza o **RT-DETR-L**, carregado a partir dos pesos pré-treinados:

```text
rtdetr-l.pt
```

Execute:

```bash
python3 training/train_rtdetr.py
```

Configuração principal:

```text
Epochs máximas: 100
Image size:     640
Batch:          8
Optimizer:      AdamW
Learning rate:  0.001
Weight decay:   0.0005
Warm-up:        3 épocas
Patience:       20 épocas
Device:         CUDA:0
Workers:        8
Seed:           42
AMP:            True
```

O treinamento foi interrompido por `early stopping` na época 91, sendo a época 71 selecionada como melhor checkpoint.

---

## YOLO11m

O segundo baseline utiliza o **YOLO11m**, carregado a partir dos pesos:

```text
yolo11m.pt
```

Execute:

```bash
python3 training/train_yolo11.py
```

A configuração experimental foi mantida próxima à utilizada pelo RT-DETR-L:

```text
Epochs máximas: 100
Image size:     640
Batch:          8
Optimizer:      AdamW
Learning rate:  0.001
Weight decay:   0.0005
Warm-up:        3 épocas
Patience:       20 épocas
Device:         CUDA:0
Workers:        8
Seed:           42
AMP:            True
```

Durante a época 27 ocorreu uma interrupção causada por `CUDA_ERROR_LAUNCH_TIMEOUT`.

O treinamento foi retomado utilizando:

```bash
yolo train resume \
    model=runs/yolo11/woodscape_yolo11m_e100_b8/weights/last.pt
```

A execução prosseguiu normalmente até o `early stopping` na época 91. Assim como no RT-DETR-L, a melhor época foi a 71.

---

# Avaliação

A avaliação final dos modelos é realizada pelo script:

```text
evaluation/evaluate_model.py
```

O script utiliza explicitamente:

```python
split="test"
```

para avaliar os checkpoints selecionados nas 824 imagens reservadas para teste.

## RT-DETR-L

```bash
python3 evaluation/evaluate_model.py \
    --model-type rtdetr \
    --weights runs/rtdetr/woodscape_test-2/weights/best.pt \
    --name rtdetr_l_woodscape_test
```

## YOLO11m

```bash
python3 evaluation/evaluate_model.py \
    --model-type yolo \
    --weights runs/yolo11/woodscape_yolo11m_e100_b8/weights/best.pt \
    --name yolo11m_woodscape_test
```

---

# Resultados experimentais

## Validação

| Métrica    | RT-DETR-L |     YOLO11m |
| ---------- | --------: | ----------: |
| Precision  |     0.655 |   **0.672** |
| Recall     | **0.593** |       0.580 |
| mAP@50     |     0.618 |   **0.620** |
| mAP@50:95  |     0.394 |   **0.408** |
| Inferência |    5.8 ms |  **2.7 ms** |
| Parâmetros |   31.99 M | **20.03 M** |
| GFLOPs     |     105.4 |    **67.8** |
| Modelo     |   66.2 MB | **40.5 MB** |

Na validação, o YOLO11m apresentou mAP ligeiramente superior e menor custo computacional.

---

## Teste

Os dois melhores checkpoints foram posteriormente avaliados no mesmo conjunto de teste, contendo:

```text
824 imagens
7.226 instâncias
```

Resultados:

| Métrica    |  RT-DETR-L |     YOLO11m |
| ---------- | ---------: | ----------: |
| Precision  | **0.6941** |      0.6800 |
| Recall     | **0.5876** |      0.5604 |
| mAP@50     | **0.6270** |      0.6163 |
| mAP@50:95  | **0.4012** |      0.3976 |
| Inferência |   10.01 ms | **4.70 ms** |
| Parâmetros |    31.99 M | **20.03 M** |
| GFLOPs     |      105.4 |    **67.8** |

O RT-DETR-L apresentou desempenho global ligeiramente superior no conjunto de teste.

O YOLO11m, entretanto, apresentou desempenho muito próximo com custo computacional consideravelmente menor.

---

## Resultados por classe no conjunto de teste

### RT-DETR-L

| Classe        | Precision | Recall | mAP@50 | mAP@50:95 |
| ------------- | --------: | -----: | -----: | --------: |
| vehicles      |     0.776 |  0.737 |  0.796 |     0.598 |
| person        |     0.755 |  0.623 |  0.704 |     0.433 |
| bicycle       |     0.649 |  0.466 |  0.504 |     0.317 |
| traffic_light |     0.663 |  0.572 |  0.574 |     0.301 |
| traffic_sign  |     0.627 |  0.539 |  0.556 |     0.358 |

### YOLO11m

| Classe        | Precision | Recall | mAP@50 | mAP@50:95 |
| ------------- | --------: | -----: | -----: | --------: |
| vehicles      |     0.773 |  0.744 |  0.804 |     0.612 |
| person        |     0.782 |  0.634 |  0.710 |     0.441 |
| bicycle       |     0.608 |  0.442 |  0.504 |     0.309 |
| traffic_light |     0.612 |  0.453 |  0.522 |     0.288 |
| traffic_sign  |     0.625 |  0.529 |  0.542 |     0.338 |

O YOLO11m apresentou desempenho superior nas classes `vehicles` e `person` em mAP@50:95, enquanto o RT-DETR-L apresentou melhores resultados para `bicycle`, `traffic_light` e `traffic_sign`.

---

## Eficiência computacional

A comparação mostra uma diferença relevante entre as arquiteturas:

```text
RT-DETR-L
31.99 milhões de parâmetros
105.4 GFLOPs
~10.0 ms de inferência no teste

YOLO11m
20.03 milhões de parâmetros
67.8 GFLOPs
~4.7 ms de inferência no teste
```

O YOLO11m utiliza aproximadamente 37% menos parâmetros e apresenta custo computacional significativamente menor, mantendo desempenho de detecção próximo ao RT-DETR-L.

Esses resultados tornam a família YOLO particularmente interessante para os próximos experimentos relacionados a execução em dispositivos de borda.

---

## Artefatos gerados

Os treinamentos e avaliações geram, entre outros:

```text
weights/best.pt
weights/last.pt
results.png
labels.jpg
confusion_matrix.png
confusion_matrix_normalized.png
PR_curve.png
P_curve.png
R_curve.png
F1_curve.png
```

Os arquivos `.pt` e o diretório `runs/` são ignorados pelo Git.

---

## Observações sobre as anotações

Durante o treinamento, a Ultralytics identificou duas labels duplicadas:

```text
train/.../03404_RV.png: 1 duplicate labels removed
val/.../01866_MVL.png: 1 duplicate labels removed
```

A biblioteca removeu automaticamente essas duplicações durante o carregamento.

Como melhoria futura, o script de preparação poderá ser ajustado para eliminar labels duplicadas antes do treinamento.

---

## Próximas etapas

* [x] preparar o dataset WoodScape;
* [x] definir divisão fixa de `train`, `val` e `test`;
* [x] realizar smoke test com RT-DETR-L;
* [x] realizar treinamento completo do RT-DETR-L;
* [x] realizar treinamento completo do YOLO11m;
* [x] implementar avaliação padronizada;
* [x] avaliar RT-DETR-L no conjunto de teste;
* [x] avaliar YOLO11m no conjunto de teste;
* [x] comparar RT-DETR-L e YOLO11m;
* [ ] corrigir labels duplicadas durante a preparação;
* [ ] analisar curvas Precision-Recall, F1 e matrizes de confusão;
* [ ] analisar falsos positivos e falsos negativos;
* [ ] avaliar desempenho em objetos pequenos;
* [ ] treinar YOLO11s como alternativa mais leve;
* [ ] realizar experimentos com data augmentation;
* [ ] estudar diferentes resoluções de entrada;
* [ ] medir FPS em condições controladas;
* [ ] estudar exportação ONNX/TensorRT;
* [ ] avaliar execução em hardware de borda.

---

## Documentação

Os relatórios dos experimentos seguem o padrão:

```text
<modelo>_<dataset>_e<epochs>_b<batch>_<tipo>.tex
```

Exemplos:

```text
rtdetr_l_woodscape_e100_b8_full.tex
rtdetr_l_woodscape_e100_b8_summary.tex
yolo11m_woodscape_e100_b8_full.tex
```

Um relatório comparativo consolidado entre os modelos será produzido após a análise completa dos resultados.

---

## Referências

* WoodScape:
  https://github.com/valeoai/WoodScape

* Ultralytics RT-DETR:
  https://docs.ultralytics.com/models/rtdetr/

* Ultralytics YOLO11:
  https://docs.ultralytics.com/models/yolo11/

* RT-DETR — *DETRs Beat YOLOs on Real-time Object Detection*:
  https://arxiv.org/abs/2304.08069

---

## Autor

**Matheus Higino**

Projeto desenvolvido no contexto de pesquisa em visão computacional, detecção de objetos em imagens fisheye e processamento em ambientes de borda.
