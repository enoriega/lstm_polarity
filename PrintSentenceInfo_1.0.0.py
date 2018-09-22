import pandas as pd
import numpy as np
import json
import csv

N_valid_sentences = 0
N_invalid_sentences = 0
N_consistent_result = 0

sentences = pd.read_csv("opposing_reach_stmts.csv", encoding = "ISO-8859-1")
# shape = (10992,6), sentences from 0 to 10991

shapeSentences = sentences.shape

with open('SentencesInfo.csv', 'w', newline='') as csvfile:
  sentenceInfoWriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
  sentenceInfoWriter.writerow(["sentence text","event interval start","event interval end","trigger","rule","polarity","controller","controlled" ])
  for row in np.arange(shapeSentences[0]):  # loop over all sentences: shapeSentences[0]
    # read json file as lists and dicts
    if (row)%100==0:
      print('processing sentence ',row+1,' ...')
    filenameJson = "annotations/entry_"+str(row)+".json"
    with open(filenameJson) as json_data:
      dictJson = json.load(json_data)

    lemmas = list(dictJson['documents'].values())[0]['sentences'][0]['words']

    # loop over mentions to find events, until found matched entities in the current sentence
    flag_entity_match=0
    mention = 0
    while flag_entity_match==0 and mention<len(dictJson["mentions"]):
      dictCurr = dictJson["mentions"][mention]
      if 'trigger' in dictCurr.keys():   #find events
        if "controlled" in dictCurr["arguments"].keys():  #find events that have controller and controlled
          # get controller and controlled name from json

          controlled = dictCurr["arguments"]["controlled"][0]["text"]
          controller = dictCurr["arguments"]["controller"][0]["text"]
          tokens = dictCurr
          # get controller and controlled name from csv
          subj = str(sentences.iloc[row]["SubjText"])
          obj = str(sentences.iloc[row]["ObjText"])
          # compare the names from json and from csv
          if controller.lower()==subj.lower() and controlled.lower()==obj.lower():
            N_valid_sentences+=1
            flag_entity_match=1
            sentenceText = sentences.iloc[row]["Sentence"]
            eventIntervalStart = dictCurr["tokenInterval"]["start"]
            eventIntervalEnd = dictCurr["tokenInterval"]["end"]
            eventTrigger = dictCurr["trigger"]["text"]
            inferenceRule = dictCurr["foundBy"]
            polarity = dictCurr["displayLabel"]
            sentenceInfoWriter.writerow([' '.join(lemmas), eventIntervalStart, eventIntervalEnd, eventTrigger, inferenceRule, polarity, '"'+controller+'"', '"'+controlled+'"'])
      mention+=1
    if flag_entity_match==0:
      N_invalid_sentences+=1

print("valid sentences:",N_valid_sentences)
print("invalid sentences:", N_invalid_sentences)
print("total sentence:", N_valid_sentences+N_invalid_sentences)
