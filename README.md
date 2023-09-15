# Improved YOLOv5s for Small Ship Detection with Optical Remote Sensing Images
A small ship detection method based on spanning connections, Hybrid Spatial Pyramid Pooling, Coordinate Attention mechanism and EIOU (Improved YOLOv5s) is proposed.

We created a new ship dataset called OSSD, which has 10,133 images and corresponding label files. 
Link: https://pan.baidu.com/s/1lI9A1fh0uU_ADdy7DdBO3Q 
Code: inav

# Partial experimental results
#
                                                   P	   R	  AP
          YOLOv5s	                93.65	87.78	87.36
+ Weighted-Add-IPNH	93.30	90.12	90.51
+ Add-IPNH	                92.37	91.04	91.44
+ Weighted-Concat-IPNH	92.40	90.24	90.89
+ Concat-IPNH (Our)	95.16	92.86	92.40

Notes: Weighted-Add-IPNH denotes the skip connections adopted the weighted fusion approaches and sum features up. Add-IPNH denotes the skip connections directly sum features up. Weighted-Concat-IPNH denotes the skip connections adopted the weighted fusion approaches and concatenated features. Concat-IPNH denotes the skip connections without weighted concatenated features.

Cite: Z. Liu, W. Zhang, H Yu, S. Zhou, W. Qi, Y. Guo and C. Li. "Improved YOLOv5s for Small Ship Detection with Optical Remote Sensing Images". IEEE Geoscience and Remote Sensing Letters (under consideration by GRSL)