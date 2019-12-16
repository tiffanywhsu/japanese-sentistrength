# coding: utf-8
import numpy as np
import pandas as pd
import csv
import emoji
import re
from jNlp.jTokenize import jTokenize
from tinysegmenter import *
import io
import subprocess
import shlex
import os.path
import sys


def processLine(linewrite, termsList):
    '''
    Remove all spaces within a segmented term in the input string.
    Input:
    	linewrite: input string
        termsList: list of terms from dictionary files
    Output:
    	linewrite: processed string
    '''
    for term in termsList:
        if linewrite.find(term) != -1:
            newterm = term.replace(' ','')
            linewrite = linewrite.replace(term, newterm)
    return linewrite


def run_jp_sentistrength(input_file_path, output_folder_path, sentistrength_path, sentistrength_dictionary_path):
    '''
    Uses Japanese SentiStrength to tag texts for sentiment.
    Output file will be saved under [processed_path]0.txt.
    If number of texts is larger, wait a little for program to complete.

    Input:
        input_file_path: file path to the file with raw text (REQUIRES ending with '.csv')
        output_folder_path: file path to folder where you want to save your processed text file (REQUIRES ending with '/')
        sentistrength_path: file path to the Japanese SentiStrength program
        sentistrength_dictionary_path: file path to the folder of the Japanese SentiStrength dictionary files
    Output:
        None
    '''

    # Load tweets dataframe
    dfG = pd.read_csv(input_file_path, encoding='utf8')


    ################ PROCESSING APPLICABLE TO JP SENTISTRENGTH ################################################
    # Add all segmented dictionary terms to a list
    termsList = []
    for filename in ['NegatingWordListSeg.txt', 'QuestionWordsSeg.txt', 'BoosterWordListSeg.txt', 
                     'SentimentLookupTableSeg.txt', 'NegationExceptionListSeg.txt']:
        with io.open(sentistrength_dictionary_path + "segmented/" + filename, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line[0:line.find('\t')]
                line = line.replace('\n','').replace('\r','')
                termsList.append(line)
            f.close()


    # Process each tweet to segment the tweet into Japanese 'words', recognize and process the dictionary terms
    # And remove newlines and other unnecessary characters
    i=0
    with io.open(output_folder_path + "output_messages.txt" ,mode='a', encoding='utf-8') as fw:
        for m in dfG['message']:
            segmenter = TinySegmenter()
            linewrite = re.sub(r'\s+', '', m)
            linewrite.rstrip()
            linewrite.replace('~',' ')
            linewrite = ' '.join(segmenter.tokenize(linewrite))
            linewrite = processLine(linewrite, termsList)
            linewrite = re.sub(r" +", " ", linewrite)
            linewrite = linewrite+'\n'
            fw.write(linewrite)
            i+=1
    fw.close()



    ######################## APPLY SENTISTRENGTH to each status ####################################
    ## Modified from: http://sentistrength.wlv.ac.uk/jkpop/ClassifyCommentSentiment.py

    FileToClassify = output_folder_path + "output_messages.txt"

    #Test file locations and quit if anything not found
    if not os.path.isfile(sentistrength_path):
        print("SentiStrength not found at: ", sentistrength_path)
        sys.exit()
    if not os.path.isdir(sentistrength_dictionary_path):
        print("SentiStrength langauge files folder not found at: ", sentistrength_dictionary_path)
        sys.exit()
    if not os.path.isfile(FileToClassify):
        print("File to classify not found at: ", FileToClassify)

    print("Running SentiStrength on file " + FileToClassify + " with command:")
    cmd = 'java -jar "' + sentistrength_path + '" sentidata "' + sentistrength_dictionary_path + '" input "' + FileToClassify + '" negatingWordsOccurAfterSentiment maxWordsAfterSentimentToNegate 1 negatingWordsDontOccurBeforeSentiment maxWordsAfterBoosters 1'
    print(cmd)
    p = subprocess.Popen(shlex.split(cmd),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    #classifiedSentimentFile = "/Users/tiffanyhsu/projects/jpsentistrength/test_output.txt"
    classifiedSentimentFile = os.path.splitext(FileToClassify)[0] + "_out.txt"
    print("Finished! The results will be in:\n" + classifiedSentimentFile)


if __name__ == '__main__':
    run_jp_sentistrength(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

