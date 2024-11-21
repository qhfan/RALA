# RALA
Implementation of "[Breaking the Low-Rank Dilemma of Linear Attention](https://arxiv.org/abs/2411.07635)"

The Softmax attention mechanism in Transformer models is notoriously computationally expensive, particularly due to its quadratic complexity, posing significant challenges in vision applications. In contrast, linear attention provides a far more efficient solution by reducing the complexity to linear levels. However, compared to Softmax attention, linear attention often experiences significant performance degradation. Our experiments indicate that this performance drop is due to the low-rank nature of linear attention's feature map, which hinders its ability to adequately model complex spatial information. In this paper, to break the low-rank dilemma of linear attention, we conduct rank analysis from two perspectives: the KV buffer and the output features. Consequently, we introduce Rank-Augmented Linear Attention (RALA), which rivals the performance of Softmax attention while maintaining linear complexity and high efficiency. Based on RALA, we construct the Rank-Augmented Vision Linear Transformer (RAVLT). Extensive experiments demonstrate that RAVLT achieves excellent performance across various vision tasks. Specifically, without using any additional labels, data, or supervision during training, RAVLT achieves an 84.4% Top-1 accuracy on ImageNet-1k with only 26M parameters and 4.6G FLOPs. This result significantly surpasses previous linear attention mechanisms, fully illustrating the potential of RALA. 
![RALA](https://github.com/qhfan/RALA/blob/main/RALA.png)

## Image Classification
|Model|Params(M)|FLOPs(G)|Acc(%)|log|ckpt|
|-|-|-|-|-|-|
|RAVLT-T|15|2.4|82.8|[RAVLT-T](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_T_log.txt)|[RAVLT-T](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_T.pth)
|RAVLT-S|26|4.6|84.4|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_S_log.txt)|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_S.pth)
|RAVLT-B|48|9.9|85.5|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_B_log.txt)|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_B.pth)
|RAVLT-L|95|16.0|85.8|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_L_log.txt)|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_L.pth)

## Object Detection and Semantic Segmentation
### Retinanet 1x
|Backbone|Params(M)|FLOPs(G)|AP<sup>b</sup>|AP<sup>b</sup><sub>50</sub>|AP<sup>b</sup><sub>75</sub>|AP<sup>b</sup><sub>S</sub>|AP<sup>b</sup><sub>M</sub>|AP<sup>b</sup><sub>L</sub>|ckpt|
|-|-|-|-|-|-|-|-|-|-|
|RAVLT-T|24 |201 |45.9 |67.4 |49.4 |28.5 |50.1 |60.8|[RAVLT-T](https://huggingface.co/aldjalkdf/RAVLT/blob/main/retinanet_t_1x_12_epoch.pth)|
|RAVLT-S|34 |244 |48.3 |69.8 |52.1 |32.7 |52.8 |63.6|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/retinanet_s_1x_12_epoch.pth)|
|RAVLT-B|57 |353 |49.8 |71.2 |54.0 |34.0 |54.3 |64.9|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/retinanet_b_1x_12_epoch.pth)|
|RAVLT-L|104 |482 |50.9 |72.2 |55.0 |34.7 |55.7 |65.4|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/retinanet_l_1x_12_epoch.pth)|
