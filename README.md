<h2 align="center">✨ Med-DisSeg: Dispersion-Driven Medical Image Segmentation</h2>

<p align="center">
  <b>
    Zhiquan Chen<sup>1,2</sup>, 
    Haitao Wang<sup>1,2</sup>, 
    Guowei Zou<sup>1,2</sup>, 
    Zhen Zhang<sup>4</sup><sup>*</sup>, 
    Xin Li<sup>3</sup><sup>*</sup>, 
    Hejun Wu<sup>1,2</sup><sup>*</sup>, 
    Weifeng Li<sup>3</sup><sup>*</sup>
  </b>
</p>

<p align="center">
  <sup>1</sup>School of Computer Science and Engineering, Sun Yat-sen University, Guangzhou, China<br>
  <sup>2</sup>Guangdong Key Laboratory of Big Data Analysis and Processing, Guangzhou, China<br>
  <sup>3</sup>Department of Emergency Medicine, Guangdong Provincial People’s Hospital (Guangdong Academy of Medical Sciences), Southern Medical University, Guangzhou, China<br>
  <sup>4</sup>School of Computer Science and Engineering, Huizhou University, Huizhou, China
</p>

<p align="center">
  <sup>*</sup>Corresponding authors:
  <a href="mailto:zzsjbme@sjtu.edu.cn">zzsjbme@sjtu.edu.cn</a>,
  <a href="mailto:sylixin@scut.edu.cn">sylixin@scut.edu.cn</a>,
  <a href="mailto:wuhejun@mail.sysu.edu.cn">wuhejun@mail.sysu.edu.cn</a>,
  <a href="mailto:liweifeng2736@gdph.org.cn">liweifeng2736@gdph.org.cn</a>
</p>

<p align="center">
  <a href="mailto:your_email@xxx.com">
    <img src="https://img.shields.io/badge/Contact-your__email%40xxx.com-blue.svg?style=flat-square" alt="Contact Author">
  </a>
</p>

---

## Overview 🔍

<p align="center">
  <a href="https://raw.githubusercontent.com/xiaoquan-1/Med-DisSeg/main/figs/compare1.pdf">
    <img src="https://raw.githubusercontent.com/xiaoquan-1/Med-DisSeg/main/figs/meddisseg_framework.png" width="90%">
  </a>
</p>

<p align="center">
  <b>Figure 1.</b> Overall architecture of the proposed <em>Med-DisSeg</em> framework.  
  Click the figure to view the high-resolution PDF.
</p>

---

## Abstract

Accurate medical image segmentation is a fundamental prerequisite for modern precision medicine.  
However, achieving robust and fine-grained delineation remains challenging under heterogeneous visual appearances. In practice, target regions and surrounding tissues often exhibit similar intensity distributions or texture patterns, leading to ambiguous activations, boundary leakage, and unreliable separation.

We observe that these failures are closely associated with two critical factors:  
(i) **representation collapse** in the encoder, where heterogeneous anatomical structures become insufficiently separated in the embedding space; and  
(ii) **scale-biased decoding**, which inadequately balances local detail preservation against global contextual modeling.

To address these issues, we propose **Med-DisSeg**, a dispersion-driven medical image segmentation framework that jointly enhances representation learning and attentive multi-scale reconstruction.  
At the representation level, a lightweight **Dispersive Loss** is introduced to explicitly enlarge inter-sample margins by treating all in-batch hidden representations as negative pairs, thereby promoting well-dispersed and boundary-aware embeddings with negligible computational overhead.  
At the architectural level, an **ELAT (Encoder-Level Attention with TeLU activation)** module adaptively reweights channel responses and multi-scale spatial saliency to suppress noise and enhance boundary-sensitive cues. Meanwhile, a multi-branch **CBAT (Channel-Balanced Adaptive Attention)** decoder effectively fuses features across different receptive fields to mitigate single-scale bias and improve the localization of structures with diverse sizes.

Extensive experiments conducted on five public datasets spanning three imaging modalities demonstrate that Med-DisSeg consistently achieves state-of-the-art performance.  
Moreover, despite not being specifically designed for multi-organ CT segmentation, the proposed framework attains competitive results on this benchmark, indicating its strong cross-task generalization capability.

---

## Datasets 📚

We evaluate **Med-DisSeg** on five public medical image segmentation benchmarks covering three imaging modalities:

| Dataset        | Modality        | Target            |
|----------------|-----------------|-------------------|
| Kvasir-SEG     | Endoscopy       | Polyp             |
| Kvasir-Sessile | Endoscopy       | Polyp             |
| GlaS           | Histopathology  | Gland             |
| ISIC-2016      | Dermoscopy      | Skin lesion       |
| ISIC-2017      | Dermoscopy      | Skin lesion       |

For **Kvasir-SEG**, we follow the official protocol and adopt a split of 880/120 for training and validation.  
**Kvasir-Sessile**, a challenging subset of Kvasir-SEG, uses the widely adopted 156/20/20 split for training, validation, and testing.  
For **GlaS**, the official split of 85/80 is used for training and validation.  
For **ISIC-2016**, we employ the official split of 900/379, while **ISIC-2017** follows the recommended 2000/150/600 split for training, validation, and testing.

---

## Experimental Results 🏆

<p align="center">
  <img src="https://raw.githubusercontent.com/xiaoquan-1/Med-DisSeg/main/figs/table1.png" width="80%">
</p>

<p align="center">
  <b>Table 1.</b> Quantitative comparison with state-of-the-art methods on Kvasir-Sessile, Kvasir-SEG, and GlaS datasets.
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/xiaoquan-1/Med-DisSeg/main/figs/table2.png" width="45%">
</p>

<p align="center">
  <b>Table 2.</b> Quantitative comparison on ISIC-2016 and ISIC-2017 datasets.
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/xiaoquan-1/Med-DisSeg/main/figs/table3.png" width="45%">
</p>

<p align="center">
  <b>Table 3.</b> Ablation studies of individual components in Med-DisSeg.
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/xiaoquan-1/Med-DisSeg/main/figs/chaocanshuxiaorong.png" width="96%">
</p>

<p align="center">
  <b>Figure 2.</b> Dispersive loss parameter ablation experiment.
</p>
---

<p align="center">
  <img src="https://raw.githubusercontent.com/xiaoquan-1/Med-DisSeg/main/figs/compare1.png" width="96%">
</p>

<p align="center">
  <b>Figure 3.</b> Qualitative comparison with representative state-of-the-art segmentation methods.
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/xiaoquan-1/Med-DisSeg/main/figs/compare2.png" width="50%">
</p>

<p align="center">
  <b>Figure 4.</b> Qualitative comparison between Med-DisSeg and SAM3.
</p>

---

## Getting Started 🚀

### Data Preparation

Taking **Kvasir-SEG** as an example, the dataset should be organized as follows:

```text
Kvasir-SEG
├── images
├── masks
├── train.txt
├── val.txt
```


### Pretrained Weights 📦

We provide pretrained weights for **Med-DisSeg** on **Kvasir-SEG**, following the two-stage training strategy described in the paper.

- **Stage 1 (Representation Learning)**  
  The first-stage model is trained to learn well-dispersed and discriminative feature representations using the proposed *Dispersive Loss* and the ELAT module.  
  [Download Stage 1 weights (checkpoint1.pth)](https://github.com/xiaoquan-1/Med-DisSeg/releases/download/v1/checkpoint1.pth)

- **Stage 2 (Final Segmentation)**  
  The second-stage model further fine-tunes the network for accurate segmentation by integrating the *Dispersive Loss* and CBAT decoders.  
  [Download Stage 2 weights (checkpoint2.pth)](https://github.com/xiaoquan-1/Med-DisSeg/releases/download/v1/checkpoint2.pth)

> **Note:** The Stage 2 model is initialized from the corresponding Stage 1 checkpoint.  
> For reproducibility of the reported results, we recommend loading the Stage 1 weights before training or evaluating Stage 2.
> The weights of the second stage can also be directly used in the testing stage.

### Training
- To train the first stage of Med-DisSeg, run: `train_stage1.py`.
- To train the second stage of Med-DisSeg, add the weights of the first stage to the `train.py` script and run it.

### Evaluation
- To evaluate the model and generate the prediction results, run: `test.py`.