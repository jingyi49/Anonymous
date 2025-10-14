# MelCap: A Unified Single-Codebook Neural Codec for High-Fidelity Audio Compression

[Audio samples](https://melcapdemo.github.io/Mel_cap_demo/) | [Paper](https://arxiv.org/html/2510.01903v1)

MelCap is a neural audio codec that compresses complex audio into a single codebook and reconstructs it with high fidelity.

## Installation

To use MelCap only in inference mode, install it using:

```bash
pip install cosmos-tokenizer
pip install vocos
```

If you wish to train the model, install it with additional dependencies:

```bash
pip install cosmos-tokenizer
pip install vocos[train]
```

## Usage

### First Stage: Mel-spectrogram Reconstruction
```bash
find $TRAIN_DATASET_DIR -name *.wav > filelist.train
find $VAL_DATASET_DIR -name *.wav > filelist.val
```

Train with only reconstruction loss:
```bash
python recon/train_96.py
```
Train with reconstruction loss and vgg loss:
```bash
python recon/train_96_vgg.py
```
If you want to fint-tuen with GM Loss:
```
python recon/train_96_post_gm.py
```


### Second Stage: From Discrete Mel Tokens to Waveform


- In [`configs/vocos_96_com.yaml`](configs/vocos_96_com.yaml), set the `vae_ckpt_path` to point to the checkpoint from your stage one training.

- In [`vocos/feature_extractors.py`](vocos/feature_extractors.py), ensure the parameters in the feature extractor are consistent with the configuration used in [`train_96.py`](recon/train_96.py) during stage one.
```bash
python train.py -c configs/vocos_96_com.yaml
```


### Inference
In infer.py, set the config_path and ckpt_path to point to the checkpoint from your stage two training.
Additionally, in infer.py, ensure the params are consistent with the configuration used in train_96.py during the first stage.

```bash
python infer.py 
```


## Pre-trained models
[Released at huggingface](https://huggingface.co/jingyi49/MelCap)
| Model | Training Loss (Stage 1) | Dataset | First Stage Checkpoint Path | Second Stage Checkpoint Path |
|-------|-------------------------|---------|-----------------|-----------------|
| **Original Vocos** | No First Stage | AudioSet |None|`logs96/lightning_logs/version_0` |
| **VQ Model** | Reconstruction | AudioSet |`cp_8_vq3/vae_00430000`|`logs96_decoder_vq/lightning_logs/version_1` |
| **VGG-enhanced Model** | VGG + Reconstruction | AudioSet |`cp_8_vgg3/vae_00620000` |`logs96_decoder_vgg/lightning_logs/version_8` |
| **Large Dataset Model** | VGG + Reconstruction | AudioSet + HQ-Conversation | `cp_8_vgg3/vae_00620000`|`logs96_decoder_ld/lightning_logs/version_0` |

## Acknowledgments
We would like to acknowledge the following projects where parts of the codes in MelCAp are derived from:
https://github.com/NVIDIA/Cosmos-Tokenizer
https://github.com/gemelo-ai/vocos

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
