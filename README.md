# [CVPR2025]Breaking the Low-Rank Dilemma of Linear Attention
Our work is accepted by CVPR2025! The code will be released soon (The author is busy for ICCV. The code may be released within one month)

Implementation of "[Breaking the Low-Rank Dilemma of Linear Attention](https://arxiv.org/abs/2411.07635)"

## Image Classification
|Model|Params(M)|FLOPs(G)|ckpt|
|-|-|-|-|
|RAVLT-T|15|2.4|[RAVLT-T](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_T.pth)|
|RAVLT-S|26|4.6|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_S.pth)|
|RAVLT-B|48|9.9|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_B.pth)|
|RAVLT-L|95|16.0|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/RAVLT_L.pth)|

## Object Detection and Instance Segmentation
### Retinanet 1x
|Backbone|Params(M)|FLOPs(G)|AP<sup>b</sup>|AP<sup>b</sup><sub>50</sub>|AP<sup>b</sup><sub>75</sub>|AP<sup>b</sup><sub>S</sub>|AP<sup>b</sup><sub>M</sub>|AP<sup>b</sup><sub>L</sub>|ckpt|
|-|-|-|-|-|-|-|-|-|-|
|RAVLT-T|24 |201 |45.9 |67.4 |49.4 |28.5 |50.1 |60.8|[RAVLT-T](https://huggingface.co/aldjalkdf/RAVLT/blob/main/retinanet_t_1x_12_epoch.pth)|
|RAVLT-S|34 |244 |48.3 |69.8 |52.1 |32.7 |52.8 |63.6|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/retinanet_s_1x_12_epoch.pth)|
|RAVLT-B|57 |353 |49.8 |71.2 |54.0 |34.0 |54.3 |64.9|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/retinanet_b_1x_12_epoch.pth)|
|RAVLT-L|104 |482 |50.9 |72.2 |55.0 |34.7 |55.7 |65.4|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/retinanet_l_1x_12_epoch.pth)|

### Mask R-CNN 1x
|Backbone|Params(M)|FLOPs(G)|AP<sup>b</sup>|AP<sup>b</sup><sub>50</sub>|AP<sup>b</sup><sub>75</sub>|AP<sup>m</sup>|AP<sup>m</sup><sub>50</sub>|AP<sup>m</sup><sub>75</sub>|ckpt|
|-|-|-|-|-|-|-|-|-|-|
|RAVLT-T|33 |219 |47.3 |69.1 |51.9 |42.7 |66.2 |46.0|[RAVLT-T](https://huggingface.co/aldjalkdf/RAVLT/blob/main/maskrcnn_t_1x_12_epoch.pth)|
|RAVLT-S|44 |262 |49.8 |71.3 |54.5 |44.6 |68.5 |48.2|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/maskrcnn_s_1x_12_epoch.pth)|
|RAVLT-B|67 |372 |51.2 |72.7 |56.4 |45.7 |69.9 |49.5|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/maskrcnn_b_1x_12_epoch.pth)|
|RAVLT-L|114 |501 |52.3 |73.8 |57.3 |46.4 |71.1 |50.4|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/maskrcnn_l_1x_12_epoch.pth)|

### Mask R-CNN 3x
|Backbone|Params(M)|FLOPs(G)|AP<sup>b</sup>|AP<sup>b</sup><sub>50</sub>|AP<sup>b</sup><sub>75</sub>|AP<sup>m</sup>|AP<sup>m</sup><sub>50</sub>|AP<sup>m</sup><sub>75</sub>|ckpt|
|-|-|-|-|-|-|-|-|-|-|
|RAVLT-S|44 |262 |51.4 |72.3 |56.5 |45.5 |69.7 |48.8|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/maskrcnn_s_3x_36_epoch.pth)|
|RAVLT-B|67 |372 |52.7 |73.5 |57.7 |46.4 |70.6 |50.2|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/maskrcnn_b_3x_36_epoch.pth)|
|RAVLT-L|114 |501 |53.6 |74.4 |58.9 |47.3 |71.6 |51.2|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/maskrcnn_l_3x_36_epoch.pth)|

### Cascade Mask R-CNN  3x
|Backbone|Params(M)|FLOPs(G)|AP<sup>b</sup>|AP<sup>b</sup><sub>50</sub>|AP<sup>b</sup><sub>75</sub>|AP<sup>m</sup>|AP<sup>m</sup><sub>50</sub>|AP<sup>m</sup><sub>75</sub>|ckpt|
|-|-|-|-|-|-|-|-|-|-|
|RAVLT-S|82 |741 |54.2 |72.9 |58.7 |46.8 |70.5 |50.9|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/Casmaskrcnn_s_3x_36_epoch.pth)|
|RAVLT-B|105 |851 |55.3 |73.8 |60.1 |47.7 |71.4 |52.1|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/Casmaskrcnn_b_3x_36_epoch.pth)|
|RAVLT-L|152 |979 |55.6 |74.1 |60.5 |48.0 |71.8 |52.3|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/Casmaskrcnn_l_3x_36_epoch.pth)|

## Semantic Segmentation
### Semantic FPN 1x
|Backbone|Params(M)|FLOPs(G)|mIoU(%)|ckpt|
|-|-|-|-|-|
|RAVLT-T|18 |136 |47.9|[RAVLT-T](https://huggingface.co/aldjalkdf/RAVLT/blob/main/fpn_t_1x.pth) |
|RAVLT-S|28 |180 |49.5|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/fpn_s_1x.pth) |
|RAVLT-B|51 |292 |51.9|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/fpn_b_1x.pth) |
|RAVLT-L|98 |424 |52.6|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/fpn_l_1x.pth) | 

### UperNet 2x
|Backbone|Params(M)|FLOPs(G)|mIoU(%)|ckpt|
|-|-|-|-|-|
|RAVLT-S|55 |937 |50.7|[RAVLT-S](https://huggingface.co/aldjalkdf/RAVLT/blob/main/uper_s_2x.pth)|
|RAVLT-B|77 |1050 |52.5|[RAVLT-B](https://huggingface.co/aldjalkdf/RAVLT/blob/main/uper_b_2x.pth)|
|RAVLT-L|125 |1182 |53.2|[RAVLT-L](https://huggingface.co/aldjalkdf/RAVLT/blob/main/uper_l_2x.pth)|

### Citation

```bibtex
@inproceedings{fan2024breakinglowrank,
      title={Breaking the Low-Rank Dilemma of Linear Attention},
      author={Qihang Fan and Huaibo Huang and Ran He },
      year={2025},
      booktitle={CVPR},
}
```

The Softmax attention mechanism in Transformer models is notoriously computationally expensive, particularly due to its quadratic complexity, posing significant challenges in vision applications. In contrast, linear attention provides a far more efficient solution by reducing the complexity to linear levels. However, compared to Softmax attention, linear attention often experiences significant performance degradation. Our experiments indicate that this performance drop is due to the low-rank nature of linear attention's feature map, which hinders its ability to adequately model complex spatial information. In this paper, to break the low-rank dilemma of linear attention, we conduct rank analysis from two perspectives: the KV buffer and the output features. Consequently, we introduce Rank-Augmented Linear Attention (RALA), which rivals the performance of Softmax attention while maintaining linear complexity and high efficiency. Based on RALA, we construct the Rank-Augmented Vision Linear Transformer (RAVLT). Extensive experiments demonstrate that RAVLT achieves excellent performance across various vision tasks. Specifically, without using any additional labels, data, or supervision during training, RAVLT achieves an 84.4% Top-1 accuracy on ImageNet-1k with only 26M parameters and 4.6G FLOPs. This result significantly surpasses previous linear attention mechanisms, fully illustrating the potential of RALA. 
