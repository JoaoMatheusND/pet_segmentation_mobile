# Pet Segmentation Mobile

Pipeline completo de segmentação de animais domésticos: treinamento de UNet em PyTorch → exportação para TFLite → app Android com inferência on-device.

## Visão Geral

O modelo segmenta imagens de pets em 3 classes por pixel:
- **Pet** (animal)
- **Background** (fundo)
- **Border** (borda/contorno)

Dataset: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (baixado automaticamente).

## Pré-requisitos e Instalação

### 1. Miniconda (gerenciador de ambiente Python)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Reiniciar o terminal após instalar
```

### 2. Java 17 (necessário para Gradle / build Android)

```bash
sudo apt install openjdk-17-jdk
```

> **Atenção:** Java 21 é incompatível com Gradle 8.x. Use Java 17.

### 3. Android SDK

**Opção A — Android Studio (recomendado):**
Baixar em [developer.android.com/studio](https://developer.android.com/studio). O SDK é instalado automaticamente.

**Opção B — Command Line Tools:**
```bash
# Baixar em https://developer.android.com/studio#command-tools
mkdir -p ~/Android/Sdk/cmdline-tools/latest
unzip commandlinetools-linux-*.zip -d ~/Android/Sdk/cmdline-tools/latest

export ANDROID_HOME=~/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin

sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"
```

Após instalar, defina o caminho do SDK no projeto:
```bash
echo "sdk.dir=$HOME/Android/Sdk" > android/local.properties
```

### 4. ADB (instalar APK no dispositivo)

Incluído no Android SDK (`platform-tools`). Para usar sem o SDK completo:
```bash
sudo apt install adb
```

### Opcional — GPU CUDA 11.8

Necessário apenas para acelerar o treinamento. Sem GPU, o treino roda na CPU (mais lento).

## Uso Rápido

```bash
# Pipeline completo: treino → export → APK
./build.sh

# Pular treino (reusar best.pth existente)
./build.sh --skip-train

# Só build Android (requer model.tflite já gerado)
./build.sh --android-only
```

O script cria automaticamente o ambiente conda `deepl` e instala as dependências.

## Pipeline Detalhado

### Fase 1 — Treinamento

```bash
conda activate deepl
cd training
python train.py       # 20 épocas → salva best.pth
python evaluate.py    # IoU e pixel accuracy no conjunto val
python visualize.py   # gera results.png com 4 amostras
```

Saída do treino:
```
Epoch 01/20 | train=0.8225 | val=0.8493 √
Epoch 02/20 | train=0.6915 | val=0.7364 √
...
```
`√` indica novo melhor val loss — `best.pth` atualizado.

### Fase 2 — Exportação

```bash
python export.py
# PyTorch (.pth) → ONNX → TF SavedModel → TFLite FP32
# Saída: model.tflite (~100+ MB)
```

### Fase 3 — App Android

```bash
cd android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Arquitetura do Modelo

**UNet** com encoder-decoder de 5 níveis e skip connections:

| Componente | Detalhe |
|---|---|
| Entrada | RGB 256×256, normalização ImageNet |
| Encoder | 64→128→256→512→1024 canais, max-pooling |
| Decoder | Upsampling bilinear + concatenação |
| Saída | Logits [B, 256, 256, 3] |
| Otimizador | Adam, lr=1e-3 |
| Loss | CrossEntropyLoss |

## App Android

Kotlin + TensorFlow Lite 2.14. O app permite capturar foto pela câmera ou galeria e exibe o resultado em painel duplo: imagem original + overlay colorido da segmentação.

```
MainActivity → Segmentor.kt → model.tflite (assets/)
     ↓              ↓
  UI + input    preprocess → infer → argmax → color overlay
```

## Estrutura do Projeto

```
pet_segmentation_mobile/
├── build.sh              # Script de orquestração end-to-end
├── training/
│   ├── model.py          # Arquitetura UNet
│   ├── dataset.py        # Loader Oxford-IIIT Pet
│   ├── train.py          # Loop de treinamento
│   ├── evaluate.py       # Métricas (IoU, pixel accuracy)
│   ├── visualize.py      # Plot de amostras
│   ├── export.py         # Conversão para TFLite
│   ├── requirements.txt  # Dependências Python
│   └── best.pth          # Checkpoint (gerado pelo treino)
└── android/
    └── app/src/main/
        ├── java/com/petseg/mobile/
        │   ├── MainActivity.kt   # Activity principal
        │   └── Segmentor.kt      # Wrapper de inferência TFLite
        └── assets/
            └── model.tflite      # Modelo (copiado pelo build.sh)
```
