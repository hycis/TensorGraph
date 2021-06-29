## Level 1 - HeteroNet
<img src="img/level1.png" height="500">

## Level 2
HeteroNet contains single_encoder and merged_encoder.<br>
Single_encoder served as image_feature_extraction from individual MRI series.<br>
It is a single encoder that process all input modalitlies (T1,T2,T1+C) in cropped ROI region of 4x320x320.<br>
Merged_encoder merges all features from all inputs for further classification.<br>
<img src="img/level2.png" height="500">

## Level 3 - Single Encoder
<img src="img/level3_1.png" height="500">

## Level 3 = Merged Encoder
<img src="img/level3_2.png" height="500">