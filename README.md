# Fisheye Object Detection

Projeto de pesquisa para **detecção de objetos em imagens fisheye**, utilizando o dataset **WoodScape** e modelos de detecção em tempo real, com foco inicial no **RT-DETR** e posterior comparação com arquiteturas da família **YOLO**.

O objetivo é construir um pipeline reprodutível para preparação dos dados, treinamento, validação e comparação de modelos em imagens obtidas por câmeras fisheye.

---

## Objetivos

Este projeto busca:

- preparar o dataset WoodScape para detecção de objetos;
- converter as anotações originais para o formato utilizado pela Ultralytics;
- manter divisões fixas de `train`, `val` e `test`;
- treinar e avaliar o RT-DETR;
- comparar RT-DETR com modelos YOLO utilizando os mesmos dados;
- avaliar métricas de precisão e desempenho computacional;
- investigar posteriormente execução em dispositivos de borda.

As principais métricas de interesse são:

- Precision;
- Recall;
- mAP@50;
- mAP@50:95;
- número de parâmetros;
- GFLOPs;
- uso de VRAM;
- latência;
- FPS;
- tamanho do modelo.

---

## Dataset

O projeto utiliza o [WoodScape](https://github.com/valeoai/WoodScape), dataset desenvolvido pela Valeo para tarefas de visão computacional com câmeras fisheye.

Para a tarefa de detecção são utilizadas cinco classes:

| ID | Classe |
|---:|---|
| 0 | `vehicles` |
| 1 | `person` |
| 2 | `bicycle` |
| 3 | `traffic_light` |
| 4 | `traffic_sign` |

O dataset preparado neste projeto contém:

| Split | Imagens | Objetos |
|---|---:|---:|
| Train | 6.587 | 57.719 |
| Validation | 823 | 7.118 |
| Test | 824 | 7.226 |
| **Total** | **8.234** | **72.063** |

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
│   └── train_rtdetr.py
│
├── docs/
│   └── relatorio_woodscape_rtdetr_parcial.tex
│
├── runs/
├── requirements.txt
├── .gitignore
└── README.md
```

As pastas `datasets/`, `runs/` e arquivos de pesos não devem ser enviados para o GitHub.

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
cd scripts

python3 prepare_woodscape.py     --raw ../datasets/woodscape_raw     --output ../datasets/woodscape
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

---

## Treinamento com RT-DETR

O treinamento atual utiliza o **RT-DETR-L** disponibilizado pela Ultralytics.

Execute a partir da raiz do projeto:

```bash
python3 training/train_rtdetr.py
```

Configuração utilizada no primeiro teste:

```text
Modelo:        rtdetr-l.pt
Epochs:        3
Image size:    640
Batch:         4
Device:        CUDA:0
Workers:       8
Seed:          42
Pretrained:    True
Deterministic: False
AMP:           True
```

O modelo pré-treinado é baixado automaticamente pela Ultralytics na primeira execução.

---

## Hardware utilizado

O primeiro treinamento em GPU foi realizado com:

```text
GPU: NVIDIA GeForce RTX 5070
VRAM: ~12 GB
PyTorch: 2.14.0+cu130
CUDA PyTorch: 13.0
Python: 3.14.4
Ultralytics: 8.4.147
```

---

## Resultados preliminares

Foi realizado inicialmente um **smoke test de 3 épocas** para validar todo o pipeline.

Resultados do melhor checkpoint no conjunto de validação:

| Métrica | Valor |
|---|---:|
| Precision | 0.604 |
| Recall | 0.552 |
| mAP@50 | 0.564 |
| mAP@50:95 | 0.347 |
| Inferência | ~5.5 ms/imagem |

Resultados por classe:

| Classe | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|
| vehicles | 0.674 | 0.762 | 0.765 | 0.559 |
| person | 0.696 | 0.639 | 0.683 | 0.391 |
| bicycle | 0.507 | 0.413 | 0.430 | 0.242 |
| traffic_light | 0.651 | 0.384 | 0.425 | 0.222 |
| traffic_sign | 0.491 | 0.561 | 0.518 | 0.324 |

> **Importante:** esses valores não representam o desempenho final do modelo. O treinamento teve apenas 3 épocas e a configuração utilizava `warmup_epochs=3`. O objetivo desse experimento foi apenas validar o funcionamento do pipeline.

---

## Artefatos gerados pelo treinamento

Os resultados são armazenados em:

```text
runs/rtdetr/woodscape_test/
```

Entre os principais arquivos gerados estão:

```text
weights/best.pt
weights/last.pt
labels.jpg
results.png
```

Os arquivos `.pt` e o diretório `runs/` são ignorados pelo Git.

---

## Observações sobre as anotações

Durante o primeiro treinamento, a Ultralytics identificou duas labels duplicadas:

```text
train/.../03404_RV.png: 1 duplicate labels removed
val/.../01866_MVL.png: 1 duplicate labels removed
```

A correção do script de preparação para eliminar duplicatas antes do treinamento está prevista como uma das próximas melhorias.

---

## Próximas etapas

- [ ] Remover labels duplicadas durante a preparação;
- [ ] regerar o dataset preparado;
- [ ] definir hiperparâmetros definitivos;
- [ ] realizar treinamento completo do RT-DETR-L;
- [ ] avaliar o melhor checkpoint;
- [ ] preservar o conjunto de teste para avaliação final;
- [ ] implementar treinamento com YOLO11;
- [ ] comparar YOLO11 e RT-DETR utilizando os mesmos splits;
- [ ] medir latência, FPS, parâmetros, GFLOPs e VRAM;
- [ ] estudar exportação ONNX/TensorRT;
- [ ] avaliar execução em hardware de borda.

---

## Resultados experimentais

Os resultados definitivos serão adicionados após a conclusão dos treinamentos completos.

A comparação será realizada mantendo, sempre que possível:

- o mesmo dataset;
- os mesmos splits;
- a mesma resolução de entrada;
- o mesmo protocolo de avaliação;
- as mesmas métricas.

Isso permite uma comparação mais justa entre diferentes arquiteturas.

---

## Documentação

Um relatório técnico parcial das etapas já realizadas está disponível em:

```text
docs/relatorio_woodscape_rtdetr_parcial.tex
```

---

## Referências

- WoodScape:  
  https://github.com/valeoai/WoodScape

- Ultralytics RT-DETR:  
  https://docs.ultralytics.com/models/rtdetr/

- RT-DETR — *DETRs Beat YOLOs on Real-time Object Detection*:  
  https://arxiv.org/abs/2304.08069

---

## Autor

**Matheus Higino**

Projeto desenvolvido no contexto de pesquisa em visão computacional, detecção de objetos em imagens fisheye e processamento em ambientes de borda.
