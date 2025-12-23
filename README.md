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
  <!-- arXiv Badge (optional, add later if available) -->
  <!--
  <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg?style=flat-square" alt="arXiv Paper">
  </a>
  -->
  <!-- Contact Badge -->
  <a href="mailto:your_email@xxx.com" target="_blank">
    <img src="https://img.shields.io/badge/Contact-your__email%40xxx.com-blue.svg?style=flat-square" alt="Contact Author">
  </a>
</p>

---

## Overview 🔍
<p align="center">
  <a href="/home/d501/data/czq/ConDSeg-main/zhanshitu/模型图final.pdf">
    <img src="/home/d501/data/czq/ConDSeg-main/zhanshitu/meddisseg_framework.png" width="90%">
  </a>
</p>

**Figure 1. Framework of the proposed Med-DisSeg.**  
Click the figure to view the high-resolution PDF.



---

## Abstract
Accurate medical image segmentation is fundamental to precision medicine, yet achieving robust and fine-grained delineation remains difficult under heterogeneous appearances. In practice, targets and surrounding tissues often share similar intensity or texture, leading to ambiguous activations, boundary leakage, and unreliable separation. We find that these failures are closely associated with two factors: (i) representation collapse in the encoder, where heterogeneous anatomical structures become insufficiently separated in the embedding space, and (ii) scale-biased decoding that inadequately balances local detail against global context.

To address these issues, we propose **Med-DisSeg**, a dispersion-driven segmentation framework that jointly strengthens representation learning and improves attentive multi-scale reconstruction. At the representation level, we introduce a lightweight **Dispersive Loss** that treats all in-batch hidden representations as negative pairs, explicitly enlarging inter-sample margins and promoting well-dispersed, boundary-aware embeddings with negligible overhead. At the architectural level, an **ELAT (Encoder-Level Attention with TeLU activation)** module reweights channel responses and multi-scale spatial saliency to suppress noise and enhance boundary-sensitive cues. Meanwhile, a multi-branch **CBAT (Channel-Balanced Adaptive Attention)** decoder adaptively fuses features across receptive fields to mitigate single-scale bias and improve localization of structures with diverse sizes.

Extensive experiments on five datasets spanning three imaging modalities demonstrate consistent state-of-the-art performance. Moreover, although not tailored for multi-organ CT segmentation, Med-DisSeg achieves competitive results on this benchmark, supporting its cross-task applicability.

---

## Datasets 📚
We evaluate Med-DisSeg on five public medical image segmentation datasets across three imaging modalities:

| Dataset        | Modality        | Target            |
|----------------|-----------------|-------------------|
| Kvasir-SEG     | Endoscopy       | Polyp             |
| Kvasir-Sessile | Endoscopy       | Polyp             |
| GlaS           | Histopathology  | Gland             |
| ISIC-2016      | Dermoscopy      | Skin lesion       |
| ISIC-2017      | Dermoscopy      | Skin lesion       |

For Kvasir-SEG, we followed the official recommendation, using a split of 880/120 for training and validation. Kvasir-Sessile, a challenging subset of Kvasir-SEG, adopted the widely used split of 156/20/20 for training, validation, and testing as in [TGANet](https://github.com/nikhilroxtomar/TGANet), [TGEDiff](https://www.sciencedirect.com/science/article/pii/S0957417424004147), etc. For GlaS, we used the official split of 85/80 for training and validation. For ISIC-2016, we utilized the official split of 900/379 for training and validation. For ISIC-2017, we also followed the official recommendation, using a split of 2000/150/600 for training, validation and testing.

## Experimental Results🏆

[//]: # (![img.png]&#40;figures/comp_1.png&#41;)

[//]: # (![img.png]&#40;img.png&#41;)

**Table 1. Quantitative comparison of ConDSeg with state-of-the-art methods on Kvasir-Sessile, Kvasir-SEG and GlaS datasets.**
<div>
    <img src="/home/d501/data/czq/ConDSeg-main/zhanshitu/表1.png" width="80%" height="96%">
</div>

**Table 2. Quantitative comparison of ConDSeg with state-of-the-art methods on ISIC-2016 and ISIC-2017 datasets.**
<div>
    <img src="/home/d501/data/czq/ConDSeg-main/zhanshitu/表2.png" width="45%" height="40%">
</div>

**Table 3. The ablation diagrams of each module.**
<div>
    <img src="/home/d501/data/czq/ConDSeg-main/zhanshitu/表3.png" width="45%" height="40%">
</div>

<br> </br>

<div>
    <img src="/home/d501/data/czq/ConDSeg-main/zhanshitu/分割图.png" width="96%" height="96%">
</div>

**Figure 2. Visualization of results comparing with other methods.**

<div>
    <img src="/home/d501/data/czq/ConDSeg-main/zhanshitu/sam3对比.png" width="96%" height="96%">
</div>

**Figure 3. Visualization of results comparing with SAM3.**

## Getting Started 🚀

### Data Preparation
Taking Kvasir-SEG as an example, the dataset should be organized as:
```text
Kvasir-SEG
├── images
├── masks
├── train.txt
├── val.txt
```

### Training
- To train the first stage of Med-DisSeg, run: `train_stage1.py`.
- To train the second stage of Med-DisSeg, add the weights of the first stage to the `train.py` script and run it.

### Evaluation
- To evaluate the model and generate the prediction results, run: `test.py`.