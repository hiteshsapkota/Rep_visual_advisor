# Rep_visual_advisor
Steps:
(a) Image feature extraction: 
     Description: Extracts the image feature (for training, validation, and test images)
     using "resnet-18" pretrained model
     
     Steps:
     1. First download the "train2017", "val2017", and "test2017" images  from the Colbert server with location 
     "/home/soccr/datasets/explainable_privacy/images" and store them to the directory "images"
     
     2. Use the feat-extract.py program with the following paramters:
         i. File name for which you want to extract the image features. This can be train2017, val2017, or test2017
         ii. location of file conaaining annotation file path. In our case it is either train2017.txt, val2017.txt, or      
         test2017.txt
        Eg: To extract feature for images under "train2017" use following command: python feat-extract.py "train2017" 
        "train2017.txt"
      3. This extracts the features and stores to Features directory with corresponding file name eg, "train2017.npy". In the 
      file each row corresponds to the image and column corresponds to the corresponding features for that image.
    "Note:" In this case you are free to experiment with other pretrained model. Modify the code "imgfeat.py" to do so.

 (b) Privacy attribute prediction:
     Description: Trains the 68 different SVM models with the features("train2017.npy" and "dev2017.npy") as an input  and attribute wise binary output stored in the directory "binary_output". Then, each model predicts the binary output using "test2017.npy" features. The prediction (series of 0 and 1) is stored in prediction directory. Also, per attribute accuracy is also stored. To make prediction use main.py program with following command:  python main.py
     Note: SVM code is provided in the svc.py. 

     



