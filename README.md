
_____________________________________________________________________________________## To train YOLOv3/v4 on darknet ##_____________________________________________________________________________________


a) Create a directory having three things:

	1.--> train directory (Having all the data for training + validation)

	2.--> classes.txt file (Having names of all the classes)

	3.--> .cfg file
_____________________________________________________________________________________________________________________________________________________________________________________________________________
b) Make all the required changes in cfg file for data.
	e.g.:

	1.--> Set desired input image size by changing: 
		width=832 or 608 or 416
		height=832 or 608 or 416 (NOTE: image size must be a multiple of 32)

	2.--> Set learning rate at:
		learning_rate=0.00261

	3.--> Set the number of classes in [yolo] layers at:
		classes=1

	4.--> Set filters in [convolutional] layers ("before [yolo] layers only") according to the classes using formula:
		filters=(classes + 5) * 3

	5.--> if OOM(Out Of Memory) Error ==> Increse "subdivisions=8" (Double the existing value )

	6.--> For other changes and more information check "https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects"
_____________________________________________________________________________________________________________________________________________________________________________________________________________
c) Now to train YOLO model:
	
	1.--> Open terminal in "darknet" directory.

	2.--> Run command "python3 yolo-training.py --EnvDir /...path to/ the directory/ created in step (a) --split 0.2 --init_weight path/to/weight file/ --gpus gpu_no(0,1,2)"

		Here:
			-->> --EnvDir: Path to the directory having training data, classes.txt file and .cfg file
			-->> --split: % of validation data, default value 0.1 (for 20% -> 0.2, for 30% -> 0.3 etc)
			-->> --init_weight: if want to start training with pre-defined weights, give the path to the weight file.
			-->> --gpus: (default: --gpus 0)In case of multiple gpus define gpu on which you want to run yolo training like: 
										(for two gpus: '--gpus 0,1', to use one of them: '--gpus 0 or --gpus 1').



_____________________________________________________________________________________________________________________________________________________________________________________________________________
d) Check Accuracy of Yolo Model:

	1.--> Open terminal in "darknet" directory.
	
	2.-->Run command "./darknet detector map /...path to/obj.data/ ./path/to/cfg file/ ./path/to/weight file/
			
		 ie:  ./darknet detector map ./data/Bang_ocr/obj.data ./data/Bang_ocr/yolov4_tiny-Custum_230123.cfg ./data/Bang_ocr/yolov4_tiny-Custum_230123_best.weights -thresh 0.25


---------example-----
python3 yolo-training.py --EnvDir ./data/Seatbelt/ --split 0.2 --init_weight ./data/Seatbelt/yolov4-tiny_seatbelt_081223.weights  --gpus 0,1
