import numpy as np
import matplotlib.pyplot as plt

filenames = [
'1_NoPretrainEmbd/Main_Seg_NoAtt/Segment_1_No_Attention_GRU.csv',
'1_NoPretrainEmbd/Main_Seg_NoAtt/Segment_1_No_Attention_LSTM.csv',
'1_NoPretrainEmbd/Main_Seg_NoAtt/Segment_1_No_Attention_biGRU.csv',
'1_NoPretrainEmbd/Main_Seg_NoAtt/Segment_1_No_Attention_biLSTM.csv',
'1_NoPretrainEmbd/Main_Seg_Att_1L/Segment_2_Attention_1L_GRU.csv',
'1_NoPretrainEmbd/Main_Seg_Att_1L/Segment_2_Attention_1L_LSTM.csv',
'1_NoPretrainEmbd/Main_Seg_Att_1L/Segment_2_Attention_1L_biGRU.csv',
'1_NoPretrainEmbd/Main_Seg_Att_1L/Segment_2_Attention_1L_biLSTM.csv',
'1_NoPretrainEmbd/Main_Seg_Att_2L/Segment_2_Attention_2L_GRU.csv',
'1_NoPretrainEmbd/Main_Seg_Att_2L/Segment_2_Attention_2L_LSTM.csv',
'1_NoPretrainEmbd/Main_Seg_Att_2L/Segment_2_Attention_2L_biGRU.csv',
'1_NoPretrainEmbd/Main_Seg_Att_2L/Segment_2_Attention_2L_biLSTM.csv',
'2_PretrainEmbd/1_NoAtt/Segment_1_No_Attention_GRU.csv',
'2_PretrainEmbd/1_NoAtt/Segment_1_No_Attention_LSTM.csv',
'2_PretrainEmbd/1_NoAtt/Segment_1_No_Attention_biGRU.csv',
'2_PretrainEmbd/1_NoAtt/Segment_1_No_Attention_biLSTM.csv',
'2_PretrainEmbd/2_Att_1L/Segment_2_Attention_1L_GRU.csv',
'2_PretrainEmbd/2_Att_1L/Segment_2_Attention_1L_LSTM.csv',
'2_PretrainEmbd/2_Att_1L/Segment_2_Attention_1L_biGRU.csv',
'2_PretrainEmbd/2_Att_1L/',
'2_PretrainEmbd/3_Att_2L/Segment_2_Attention_2L_GRU.csv',
'2_PretrainEmbd/3_Att_2L/Segment_2_Attention_2L_LSTM.csv',
'2_PretrainEmbd/3_Att_2L/Segment_2_Attention_2L_biGRU.csv',
'2_PretrainEmbd/3_Att_2L/Segment_2_Attention_2L_biLSTM.csv']


results = list()
for i in np.arange(6):
  results.append(np.genfromtxt(filenames[i], delimiter=','))
  
line_labels = ['No Att GRU','No Att LSTM','No Att biGRU','No Att biLSTM','Att 1L GRU','Att 1L LSTM','Att 1L biGRU','Att 1L biLSTM',]
lines={}
figure_titles = ['training loss', 'testing loss', 'precision', 'recall', 'f1']
plt.figure(figsize=(12,8))
for i in np.arange(5):   # loop over training, testing, precision, recall, f1
  plt.subplot(2,3,i+1)
  plt.title(figure_titles[i])
  for j in np.arange(6):  #loop over all configurations
    lines[j],=plt.plot(results[j][:,i])
  plt.xlabel('epochs')
  plt.ylabel(figure_titles[i])
  plt.legend([lines[k] for k in np.arange(6)], line_labels)
  plt.ylim((0.1))
  
plt.tight_layout()
plt.show()
