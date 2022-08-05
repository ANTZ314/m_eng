# Models tested


## ViT
| Model 	 	| Locally		| Colab 	| NOTE 	| Fit Time	|
| -			| -			| -		| -		| - 		|
| vt_adl		| GPU 		| FAIL 	| ERROR	| none 	|
| ViT_1 	 	| NO 		| YES  	| Classifier | Custom Image |
| ViT_2 	 	| GPU 		| YES 	| Classifier |1h53min |
| ViT_3 	 	| NO 		| YES 	| cifar only | ~1h	|
| ViT_4 		| NO 		| YES 	| cifar only |1h15	|
| ViT_5 	 	| GPU 		| YES  	| no classify	| ~1h |
| ViT_6 	 	| NO 		| NO  	| unfit 	| ?? |
| ViT_7? 	 	| NO 		| NO  	| unfit 	| ?? |

<br>
## Anomaly Detection
| Model 	 	| Locally		| Colab 	| NOTE 	|
| -			| -			| -		| -		|
| AE_1 		| NO 		| NO 	| Enc_Dec |
| AE_2 		| NO 		| NO 	| Denoise	|
| AE_3		| YES 		| NO 	| Rec_Anom |
| mvtec_ad	| GPU  		| PASS 	| Anomaly |
| novelty_det1	| GPU 		| NO		| Anomaly |
| novelty_det1	| GPU 		| NO		| AE Anomaly |

<br>
## Transfer Classification
| Model 	 	| Locally		| Colab 	| NOTE 	|
| -			| -			| -		| -		|
| Class_2.py 	| YES		| NO		| Classifier |
| Class_5.py 	| YES		| NO		| Classifier |
| Class_6.py 	| YES		| NO		| Classifier |
| Class_8.py 	| YES		| NO		| Classifier |
| Class_10.py	| YES		| NO		| Classifier |
| Class_x.py 	| YES		| NO		| Classifier |

