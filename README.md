# BMPT: Bidirectional Mutual Prompt Tuning for Cross-Domain Action Recognition

## 🚀  Getting Started

### 🛠️ Environment Setup

```bash
conda create -n BMPT python=3.10
conda activate BMPT
pip install -r requirements.txt
```

### 📂 Dataset Preparation
Supported datasets:

```bash
* Kinetics-400
* UCF-101
* HMDB-51
* Something-Something V2
```
Please follow the instructions provided by [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) for data preparation.  
The knowledge features required for each dataset are stored in the `./knowledge` directory.  
In particular, the extracted knowledge features for the SSv2 dataset can be downloaded from the following link: **https://pan.baidu.com/s/1_EiaMV3xAxYwy3-WjYlsag?pwd=fn6d**.


# Model Zoo
<p><b>NOTE:</b> All models in our experiments below use the publicly available ViT/B-16 based CLIP model pretrained on Kinetics-400. The pretrained weights are available at the following links:</p>

<ul>
  <li><a href="https://pan.baidu.com/s/1mVSy8elu5qUJpENrCgjkOA?pwd=ehir">Baiduyun</a></li>
</ul>


### 🔬 Base-to-novel generalization results
<h3>
</h3>

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Base</th>
      <th>Novel</th>
      <th>HM</th>
      <th>model weights</th>
    </tr>
  </thead>
  <tbody>
  <tr>
    <td><b>HMDB-51</b></td>
    <td align="center">80.2</td>
    <td align="center">63.7</td>
    <td align="center">71.0</td>
    <td align="center">
      <a href="https://pan.baidu.com/s/1nVOt_h7mRPtmhFbPcKomXg?pwd=rvjd">Baiduyun</a>
    </td>
  </tr>

  <tr>
    <td><b>UCF-101</b></td>
    <td align="center">97.5</td>
    <td align="center">89.6</td>
    <td align="center">93.9</td>
    <td align="center">
      <a href="https://pan.baidu.com/s/1ZkYIMlfL5zTHu0qLvWoL0w?pwd=9cbe">Baiduyun</a>
    </td>
  </tr>

  <tr>
    <td><b>SSv2</b></td>
    <td align="center">22.2</td>
    <td align="center">17.5</td>
    <td align="center">19.6</td>
    <td align="center">
      <a href="https://pan.baidu.com/s/1TTExXmYnwF49gIeWWyb0gA?pwd=vrtq">Baiduyun</a>
    </td>
  </tr>
</tbody>

</table>

### 📊 Few-shot Action Recognition Results
<h3>
</h3>
<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="4">K-shot</th>
      <th rowspan="2">Model Weights</th>
    </tr>
    <tr>
      <th>2-shot</th>
      <th>4-shot</th>
      <th>8-shot</th>
      <th>16-shot</th>
    </tr>
  </thead>
 <tbody>
  <tr>
    <td><b>HMDB-51</b></td>
    <td align="center">67.6</td>
    <td align="center">69.6</td>
    <td align="center">74.3</td>
    <td align="center">76.2</td>
    <td align="center">
      <a href="">Baiduyun</a>
    </td>
  </tr>
  <tr>
    <td><b>UCF-101</b></td>
    <td align="center">93.8</td>
    <td align="center">95.5</td>
    <td align="center">96.7</td>
    <td align="center">97.3</td>
    <td align="center">
      <a href="">Baiduyun</a>
    </td>
  </tr>
  <tr>
    <td><b>SSv2</b></td>
    <td align="center">8.9</td>
    <td align="center">10.6</td>
    <td align="center">13.3</td>
    <td align="center">18.5</td>
    <td align="center">
      <a href="">Baiduyun</a>
    </td>
  </tr>
</tbody>
</table>

### 🔁 Cross-dataset Action Recognition Results
<h3>

</h3>

<table>
  <thead>
    <tr>
      <th>U → H</th>
      <th>Model Weights</th>
      <th>H → U</th>
      <th>Model Weights</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td align="center"><b>94.0</b></td>
      <td align="center">
        <a href="https://pan.baidu.com/s/1XunV27lkm1xcU2-fUe1NBQ?pwd=g2ju">Baiduyun</a>
      </td>
      <td align="center"><b>95.6</b></td>
      <td align="center">
        <a href="https://pan.baidu.com/s/1DDA2dax8Qfg1zAIlSRsi1g?pwd=fbdq">Baiduyun</a>
      </td>
    </tr>
  </tbody>
</table>

### 🏋️‍♂️ Training


```bash
# Modefied the daset path in configs

# Pretraining on HMDB51 
cd scripts/pre_training
bash bmpt_b2n_hmdb51.sh
```
**Note:**
- We recommend keeping the total batch size as mentioned in respective config files. Please use `--accumulation-steps` to maintain the total batch size. Specifically, here the effective total batch size is 2(`GPUs_NUM`) x 4(`TRAIN.BATCH_SIZE`) x 8(`TRAIN.ACCUMULATION_STEPS`) = 64.
- 
### 🙏 Acknowledgements
Our project is partially based on the open-source projects [ViFiCLIP](https://github.com/muzairkhattak/ViFi-CLIP) and [BDC-CLIP](https://github.com/Fei-Long121/BDC-CLIP). We sincerely acknowledge their contributions.
