python preprocess.py data.dataset=Amazon_Musical_Instruments_14 data.type=glove
python preprocess.py data.dataset=Amazon_Digital_Music_14 data.type=glove
python preprocess.py data.dataset=Amazon_Office_Products_14 data.type=glove

python preprocess.py data.dataset=Amazon_Musical_Instruments_14 data.type=bert experiment.device=0
python preprocess.py data.dataset=Amazon_Digital_Music_14 data.type=bert experiment.device=1
python preprocess.py data.dataset=Amazon_Office_Products_14 data.type=bert experiment.device=2

# test
python preprocess.py data.dataset=Amazon_All_Beauty_18 data.type=glove 
python preprocess.py data.dataset=Amazon_All_Beauty_18 data.type=bert