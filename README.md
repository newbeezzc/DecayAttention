# Abstract

Long sequence time series data forecasting based on deep learning has been applied in many practical scenarios. However, the time series data sequences obtained in real
world inevitably contain missing values due to the failures of sensors or the network fluctuations. Current research works
dedicate to impute the incomplete time series data sequence during the data preprocessing stage, which will lead to the
problems of unsynchronized prediction and error accumulation. We propose an improved multi-headed self-attention mechanism,
DecayAttention, which can be applied to the existing X-former models to handle the missing values in the time series data
sequences without decreasing of prediction accuracy. We apply DecayAttention to two state-of-the-art X-former models, with
13.0% average prediction accuracy improvement.

## DecayAttention

We propose a portable attention mechanism DecayAttention that can adapt to incomplete datasets. DecayAttention can be used in any Transformer-based model by replacing its original attention layers.
<p align="center">
<img src=".\pic\DecayAttn-decayAttention架构图.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall architecture of DecayAttention.
</p>


We enable the model to generate robust forecasts close to the true values without going through an extra imputation step in the data preprocessing stage.
<p align="center">
<img src=".\pic\DecayAttn-模型预览图.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Model structure diagram after applying our method with Transformer.
</p>

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).
3. Train the model. We provide the experiment scripts of all benchmarks and all three baseline models under the folder `./scripts` as examples. You can reproduce the experiment results using commands like:

```bash
bash ./scripts/ETT_script/Transformer_Delay_ETTh1.sh
bash ./scripts/ECL_script/Transformer_Delay.sh
bash ./scripts/Exchange_script/Transformer_Delay.sh
bash ./scripts/Traffic_script/Transformer_Delay.sh
bash ./scripts/Weather_script/Transformer_Delay.sh
bash ./scripts/ILI_script/Transformer_Delay.sh
```
We only list scripts here that training the improved Transformer. You can see the folder `./scripts` for others.

## Main Results

We improved the Transformer, Informer and Autoformer and conducted comparative experiments with the unimproved models. Our approach gives Transformer a robust performance improvement of 13% on average.
<p align="center">
<img src=".\pic\results.png" height = "550" alt="" align=center />
</p>

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{zhang2023,
  title={An improved self-attention for long sequence timeseries data forecasting with missing values},
  author={Zhi-cheng ZHANG, Yong WANG, Jian-jian PENG, Jun-ting DUAN},
  year={2023}
}
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

