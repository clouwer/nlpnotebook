
@echo off

cd C:\Users\kenshinpg\Anaconda3\Scripts

bert-serving-start -model_dir C:\Jupyter\Kenshinpg\BERT\chinese_L-12_H-768_A-12 -num_worker=2

pause