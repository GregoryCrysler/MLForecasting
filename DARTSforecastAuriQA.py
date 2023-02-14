
                                                                                                 import sys
import gc
import pandas as pd
import numpy as np
import sqlalchemy
import pandas.io.sql as psql
import pyodbc
import matplotlib.pyplot as plt
from datetime import datetime

import joblib
import pickle
import os
#import darts
import darts.utils as utl
from darts import TimeSeries
from darts.metrics import mape, smape, mae
from darts.models import NBEATSModel, TransformerModel
#from darts.utils.data.sequential_dataset import SequentialDataset
#from darts.utils.data.horizon_based_dataset import HorizonBasedDataset
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler
from darts.timeseries import TimeSeries

from torch.nn import L1Loss
import torch
import torch.optim as optim

from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer , QuantileTransformer
from sklearn.impute import SimpleImputer

#import Settings
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)
#######################################################################################




#CreateMLSeries_TEST(SRC, 'Train', 'Sold_uncapped', TimeBegin), CreateMLSeries(SRC, 'Valid', 'Sold_uncapped', TimeBegin)
#df = SRC
#MlType = 'Valid'
#ValType = 'Sold_uncapped'
#Keys = ForecastableSeries
#BeginDT =   TimeBegin
#training_set_TOTAL_IMP.columns, valid_set_TOTAL_IMP.columns = pd.DatetimeIndex([y for x,y in training_set_TOTAL_IMP.columns.values]), pd.DatetimeIndex([y for x,y in valid_set_TOTAL_IMP.columns.values])

_no_value = object()
def CreateMLSeries_TEST (df, MlType, ValType, BeginDT, Keys=_no_value):
    if Keys is _no_value:
        output  = pd.pivot_table(df.loc[(df.FlagDataType == MlType) & (df.StartDate >= BeginDT) & (df.ZndSelloutSyncAvails > 0),[ValType, 'AnalysisKey','StartDate']]
                    , values=[ValType], fill_value=0, index=['StartDate'],columns=['AnalysisKey'], aggfunc=np.max)            
        output.columns =  [y for x,y in output.columns.values]
    else:
        output  = pd.pivot_table(df.loc[(df.FlagDataType == MlType) & (df.StartDate >= BeginDT) & (df.ZndSelloutSyncAvails > 0) & (df.AnalysisKey.isin(Keys)),[ValType, 'AnalysisKey','StartDate']]
                        , values=[ValType], fill_value=0, index=['StartDate'],columns=['AnalysisKey'], aggfunc=np.max)            
        output.columns =  [y for x,y in output.columns.values]
    return output


#_no_value = object()
#def CreateMLSeries (df, MlType, ValType, BeginDT, Keys=_no_value):
#    if Keys is _no_value:
#        output  = pd.pivot_table(df.loc[(df.FlagDataType == MlType) & (df.StartDate >= BeginDT),[ValType, 'AnalysisKey','StartDate']]
#                    , values=[ValType], fill_value=0, index=['StartDate'],columns=['AnalysisKey'], aggfunc=np.max)            
#        output.columns =  [y for x,y in output.columns.values]
#    else:
#        output  = pd.pivot_table(df.loc[(df.FlagDataType == MlType) & (df.StartDate >= BeginDT) & (df.AnalysisKey.isin(Keys)),[ValType, 'AnalysisKey','StartDate']]
#                        , values=[ValType], fill_value=0, index=['StartDate'],columns=['AnalysisKey'], aggfunc=np.max)            
#        output.columns =  [y for x,y in output.columns.values]
#    return output

#trainDF = training_set_Sold_uncapped
#validDF = valid_set_Sold_uncapped
#VarThreshold = 5
def FindForecastableSeries (trainDF, validDF, DateThreshold, VarThreshold=5):
    pop = SRC.loc[SRC.StartDate <= DateThreshold, ].groupby(['AnalysisKey'])['Sold_uncapped'].var() > VarThreshold
    keys = set(pop.loc[pop.values==True].index.values)
    #ValidSeries = list(set(trainDF.index.values).intersection(set(validDF.index.values)).intersection(keys))
    ValidSeries = list(set(trainDF.columns.values).intersection(set(validDF.columns.values)).intersection(keys))
    return ValidSeries



#train = training_set_TOTAL_IMP
#val =   valid_set_TOTAL_IMP
#scltype = 'IMP'
def TransformerData (train,val,scltype):
    #scl = Scaler()                                                                                                                                         
    scl = Scaler(scaler=QuantileTransformer(n_quantiles=100, output_distribution = 'normal'), name=scltype)        #, output_distribution = 'uniform'          #n_quantiles=100
    train_scaled = scl.fit_transform(train)
    val_scaled = scl.transform(val)                                                    
    return    train_scaled,  val_scaled   , scl

#arr, date_index = training_set_Avails, training_set_Avails.index
#i = 0
##training_set_Avails_Cov =    array_to_seq(training_set_Avails, training_set_Avails.index)
def array_to_seq(arr, date_index):
    ts_sequence = []
    for i in tqdm(range(len(arr))):
        #ts_sequence.append(TimeSeries.from_times_and_values(date_index, arr.iloc[i, :]))
        ts_sequence.append(TimeSeries.from_times_and_values(date_index, arr.iloc[:, i]))
    return ts_sequence
    

def mae_loss(forecast, target):
    return torch.mean(torch.abs(forecast - target))

def mink_loss(forecast, target,p=1.15):
    return torch.mean(torch.cdist(forecast, target, p=p))

def RMSLE_loss(forecast, target):
    return torch.sqrt(torch.mean(   torch.cdist(torch.log(forecast+1), torch.log(target+1), p=2)))

#Ad Hoc Export (pre-method to concat)
#df_numpy = X1
#scl = IMP_scl 
#PredCols = ['StartDate','AnalysisKey','TOTAL_IMP_Pred']
#PrepForExport(X1, IMP_scl, ['StartDate','AnalysisKey','TOTAL_IMP_Pred'])
#tmp = scl.inverse_transform(X1[:,1])

#def PrepForExport(df_numpy, scl, PredCols):
#    tmp = pd.DataFrame(df_numpy,index=valid_set_Sold_uncapped.index[:CHUNK_OUTPUT_LENGTH],columns=ForecastableSeries)
#    post_proc_pred = scl.inverse_transform(TimeSeries(tmp)).pd_dataframe()#.reset_index()
#    post_proc_pred.columns =  ForecastableSeries
#    post_proc_pred.reset_index(inplace=True)
#    output = pd.melt(post_proc_pred,id_vars=['StartDate'], value_vars=post_proc_pred.columns[1:])
#    output.columns = PredCols
#    return output


#df_numpy = X1
#RawCols = rawcols #pred.pd_dataframe().columns
#PredCols = ['StartDate','AnalysisKey','TOTAL_IMP_Pred']

def PrepForExportTest(df_numpy, PredCols, RawCols):    
    #post_proc_pred = pd.DataFrame(df_numpy,index=valid_set_Sold_uncapped.index[:CHUNK_OUTPUT_LENGTH],columns=RawCols)
    post_proc_pred = pd.DataFrame(df_numpy,index=pd.Series(np.arange(TimeEnd+  pd.Timedelta('1W'), ValidationTimeEnd+  pd.Timedelta('1W'), timedelta(days=7))),columns=RawCols)
    post_proc_pred.reset_index(inplace=True)
    #output = pd.melt(post_proc_pred,id_vars=['StartDate'], value_vars=post_proc_pred.columns[1:])
    output = pd.melt(post_proc_pred,id_vars=['index'], value_vars=post_proc_pred.columns[1:])
    output.columns = PredCols
    return output

#InsertToSQL(outputTotal_IMP, tblname, typ, mdl, writedate)
#tsdf = post_proc_pred_DF
#mdl = my_model
#cols = SQLCols
def InsertToSQL(tsdf, tblname, typ, mdl, writedate, cols):
    tsdf['StartDate'] = tsdf['StartDate'].apply(str)
    #tsdf.to_string(columns=['StartDate'],col_space=100)
    writedate = str(writedate)
    mdldesc = 'batch_size = '+str(mdl.batch_size)+', N_EPOCHS= '+str(mdl.n_epochs)+', num_encoder_layers= '+str(mdl.num_encoder_layers)+', nhead= '+str(mdl.nhead)
    tsdf['ExecutionType'], tsdf['ModelType'], tsdf['ExecutionDate'] = typ, mdldesc, writedate
    tsdf[cols].to_sql(name=tblname.split('.')[1], con=engine,if_exists='append',schema=tblname.split('.')[0],index=False, method=None) #,chunksize=10



#tsdf=post_proc_pred_DF
#tblname = TBL+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
def ExportToSQL(tsdf,tblname):
    tsdf['StartDate'] = tsdf['StartDate'].apply(str)    
    #tsdtype      = {'StartDate': 'str', 'AnalysisKey': 'int','TOTAL_IMP_Pred':'float'}
    tsdf.to_sql(tblname.split('.')[1], con=engine,if_exists='replace',schema=tblname.split('.')[0],chunksize=1000, index=False)#, dtype=tsdtype)



# log_message(source, start_time, log_type, log_msg, rows)
# start_ts = start_time 
# rowcount = rows
def log_message(source, start_ts, log_type, log_msg, rowcount=0):    
    # Slice the datetime to only keep milliseconds "2019-09-21 05:30:00.123" = 23 chars
    log_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[0:23]
    start_dt = start_ts[0:23]
    load_log_id = 42
    print(log_type, log_msg)
    cursor.execute('''EXECUTE '''+ProcessLogSP+'''
    @source=?, @start_dt=?, @log_dt=?, @log_message=?, @rowcount=?, @load_log_id=?, @log_type=? ''',
            (source, start_dt, log_dt, log_msg, rowcount, load_log_id, log_type))
    cnxn.commit()

##############################################################################################################################################################################

os.chdir('D:/VSWorkingDir/')                                                         #App SErver has only a 120 GB C Drive;  may not be necessary in QA/Prod

cnxn = pyodbc.connect('Driver={SQL Server};'
                        'Server=172.31.75.52;'
                        'Database=InnVision_QA;'
                        'UID=gcrysler;'
                        'PWD=Innovar!21;'
                        'Trusted_Connection=no;')
cursor = cnxn.cursor()
engine = sqlalchemy.create_engine('mssql+pyodbc://gcrysler:Innovar!21@AWSDEVSQLN1/InnVision_QA?driver=ODBC Driver 17 for SQL Server', fast_executemany=True)


TBL = 'Adaptive.BaseAnalysis_AURI'
Cols = 'AnalysisKey,StartDate,FlagDataType ,WeekIDX,BroadcastWeekKey,ZoneCode,NetworkCode,DayPart, Sold_uncapped, TOTAL_IMP, AUR, ZndSelloutSyncAvails'
SRC = psql.read_sql('SELECT '+Cols+' FROM '+TBL, cnxn)#.fillna(0)
SRC['StartDate'] = pd.DatetimeIndex(SRC['StartDate'])   #, freq='w')
#SRC.set_index('StartDate',inplace=True) #,drop=True)
                                                                                #SRC = SRC.loc[SRC.Sold_uncapped >0, ]
########################################################################################
                                                                                      #Start

LOOKBACK = 2 #3
LOOKAHEAD = 3
#CHUNK_OUTPUT_FINAL_LENGTH = 66
CHUNK_OUTPUT_LENGTH = 66    #15
CHUNK_INPUT_LENGTH = CHUNK_OUTPUT_LENGTH*LOOKBACK
Pacing_Length = 15
buffer= 26 #0  #53

TimeEnd = pd.to_datetime((SRC.loc[SRC.FlagDataType == 'Train',['StartDate']].max().values[0]))
TimeBegin = pd.to_datetime((SRC.loc[SRC.FlagDataType == 'Train',['StartDate']].min().values[0]))+ pd.Timedelta(str(buffer)+'W') 
ValidationTimeEnd = TimeEnd + pd.Timedelta(str(CHUNK_OUTPUT_LENGTH)+'W') 
#ValidationTimeBegin =   TimeEnd + pd.Timedelta(str(1)+'W')   #ValidationTimeEnd - pd.Timedelta(str(CHUNK_OUTPUT_LENGTH)+'W')                #ValidationTimeEnd - pd.Timedelta(str((LOOKBACK + LOOKAHEAD )*CHUNK_OUTPUT_LENGTH)+'W')    
ValidationTimeBegin =   TimeEnd - pd.Timedelta(str(Pacing_Length)+'W') 
DateThreshold = TimeEnd - pd.Timedelta(str(Pacing_Length)+'W') 


########################################################################################
#Create Pivoted Dataset (AnalysisKey on cols, Dates on rows)

##(1) for Sold_uncapped
###(df, MlType, ValType, BeginDT, Keys=_no_value)
training_set_Sold_uncapped, valid_set_Sold_uncapped  =  CreateMLSeries_TEST(SRC, 'Train', 'Sold_uncapped', TimeBegin), CreateMLSeries_TEST(SRC, 'Train', 'Sold_uncapped', ValidationTimeBegin)
ForecastableSeries = FindForecastableSeries(training_set_Sold_uncapped,valid_set_Sold_uncapped,DateThreshold)
training_set_Sold_uncapped_COV, valid_set_Sold_uncapped_COV = TimeSeries.from_dataframe(training_set_Sold_uncapped[ForecastableSeries]), TimeSeries.from_dataframe(valid_set_Sold_uncapped[ForecastableSeries])

##(2) for TOTAL_IMP (using ForecastableSeries found from Sold_uncapped)
training_set_TOTAL_IMP, valid_set_TOTAL_IMP  =  TimeSeries.from_dataframe(CreateMLSeries_TEST(SRC, 'Train', 'TOTAL_IMP', TimeBegin,ForecastableSeries)), TimeSeries.from_dataframe(CreateMLSeries_TEST(SRC, 'Train', 'TOTAL_IMP', ValidationTimeBegin,ForecastableSeries))

##(3) for AUR    (using ForecastableSeries found from Sold_uncapped)
training_set_AUR, valid_set_AUR  =  TimeSeries.from_dataframe(CreateMLSeries_TEST(SRC, 'Train', 'AUR', TimeBegin,ForecastableSeries)), TimeSeries.from_dataframe(CreateMLSeries_TEST(SRC, 'Train', 'AUR', ValidationTimeBegin,ForecastableSeries))


##(4) for Avails    (using ForecastableSeries found from Sold_uncapped)
training_set_Avails, valid_set_Avails =  TimeSeries.from_dataframe(CreateMLSeries_TEST(SRC, 'Train', 'ZndSelloutSyncAvails', TimeBegin,ForecastableSeries)), TimeSeries.from_dataframe(CreateMLSeries_TEST(SRC, 'Train', 'ZndSelloutSyncAvails', ValidationTimeBegin,ForecastableSeries))
########################################################################################


ProcessLogSP = 'Adaptive.up_ProcessLogV5'
source =    'DARTSStrongVarModel_Impressions|AUR_Dev.py'
start_time = str(datetime.now())
log_type='Transformer, TOTAL_IMP/AUR'
#params = ''
log_msg = 'Training Period = '+str(training_set_Sold_uncapped.shape[0])                                 
rows = len(ForecastableSeries)
log_message(source, start_time, log_type, log_msg, rows)



########################################################################################
#Scale / Impute
training_set_Sold_uncapped_COV_scl, valid_set_Sold_uncapped_COV_scl, Sold_scl  = TransformerData(training_set_Sold_uncapped_COV, valid_set_Sold_uncapped_COV, 'Sold')
training_set_TOTAL_IMP_scl, valid_set_TOTAL_IMP_scl, IMP_scl  = TransformerData(training_set_TOTAL_IMP, valid_set_TOTAL_IMP, 'IMP')
training_set_AUR_scl, valid_set_AUR_scl, AUR_scl     = TransformerData(training_set_AUR, valid_set_AUR, 'AUR')
training_set_Avails_scl, valid_set_Avails_scl, Avails_scl     = TransformerData(training_set_Avails, valid_set_Avails, 'ZndSelloutSyncAvails')
########################################################################################


################################################################################################################################################################################
gc.collect()

BATCH_SAMPLE_RATE = 0.5
N_EPOCHS = 50 #100 #40 #1 #30                  #100    #11                                                       #80
NUM_LAYERS= 2 #15
#LAYER_WIDTH=[1024,1024]                    #256 (small)#If List, Len Must be equal to Num Stacks[1...i]
MODEL_NAME = 'Transformer'+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'.dump'
BATCH_SIZE =    int(BATCH_SAMPLE_RATE * training_set_Sold_uncapped.shape[0])

#https://pytorch.org/docs/stable/generated/torch.optim.Adam.html?highlight=torch%20optim%20adam#torch.optim.Adam 
kwards = {'lr': 1e-4}  
start_time = str(datetime.now())
########################################################################################


my_model = TransformerModel(
    input_chunk_length = int(CHUNK_INPUT_LENGTH/2.25),  #int(CHUNK_INPUT_LENGTH/1.5),  #int(CHUNK_INPUT_LENGTH/2.25),    #,  #int(CHUNK_INPUT_LENGTH/1.25),      
    output_chunk_length =  CHUNK_OUTPUT_LENGTH,
    batch_size = BATCH_SIZE,
    n_epochs = N_EPOCHS,
    model_name = MODEL_NAME,
    log_tensorboard=   False,   #True,  
    nr_epochs_val_period = 9,
    d_model = 1024,                                                         #The Encoder is far more complicated than the decoder
    nhead = 4,
    num_encoder_layers = NUM_LAYERS+1,                                        #The Encoder is far more complicated than the decoder
    num_decoder_layers = NUM_LAYERS, #int(NUM_LAYERS/1.5),                               
    dim_feedforward = 512,                                                  #The Encoder is far more complicated than the decoder
    dropout = 0.15,                      #0.15   
    activation = "relu"                 #"relu"    
    , loss_fn = mink_loss   #RMSLE_loss  #mae_loss
    , optimizer_kwargs  = kwards                   
    #,random_state=42,
    , force_reset=True
)


#print('Started @ '+start_time)
##my_model.fit(series=[training_set_TOTAL_IMP_scl, training_set_AUR_scl], val_series=[valid_set_TOTAL_IMP_scl, valid_set_AUR_scl], verbose=True)
#my_model.fit(series=training_set_TOTAL_IMP_scl,  verbose=True)
#finish_time = str(datetime.now())
#print('Finished @ '+finish_time)


#training_set_Avails =          CreateMLSeries(SRC, 'Train', 'ZndSelloutSyncAvails',ForecastableSeries)
#training_set_Avails_Cov =    array_to_seq(training_set_Avails, training_set_Avails.index)


print('Started @ '+start_time)
my_model.fit(series=[training_set_TOTAL_IMP_scl, training_set_AUR_scl], verbose=True, past_covariates= [training_set_Avails_scl, training_set_Sold_uncapped_COV_scl])
finish_time = str(datetime.now())
print('Finished @ '+finish_time)
                                                                                #DOH - re-run the preds (wrong covars)
my_model._save_model('C:/ModelData/',MODEL_NAME,N_EPOCHS)
pred = my_model.predict(n=CHUNK_OUTPUT_LENGTH,series=[training_set_TOTAL_IMP_scl, training_set_AUR_scl], past_covariates= [training_set_Avails_scl, training_set_Sold_uncapped_COV_scl])           #, CHUNK_OUTPUT_LENGTH
#X1, X2 = np.array([p.values().squeeze() for p in pred[0].all_values()]), np.array([p.values().squeeze() for p in pred[1]])
X1, X2 = IMP_scl.inverse_transform(pred[0]).all_values()[:,:,-1], AUR_scl.inverse_transform(pred[1]).all_values()[:,:,-1]


                                                                                ################################################################################
rawcols = pred[0].pd_dataframe().columns
post_proc_pred_DF = PrepForExportTest(X1, ['StartDate','AnalysisKey','TOTAL_IMP_Pred'], rawcols)
SQLCols  = ['AnalysisKey','StartDate',  'TOTAL_IMP_Pred', 'ExecutionType', 'ModelType', 'ExecutionDate']
tblname = 'Adaptive.ForecastHistory_Total_IMP'
typ = 'Transformer'
writedate = finish_time
InsertToSQL(post_proc_pred_DF, tblname, typ, my_model, writedate, SQLCols)
#ExportName = TBL+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")   #+'_mae_loss'
#ExportToSQL(post_proc_pred_DF,ExportName)


post_proc_pred_DF =    PrepForExportTest(X2, ['StartDate','AnalysisKey','AUR_Pred'], rawcols)
SQLCols  = ['AnalysisKey','StartDate',  'AUR_Pred', 'ExecutionType', 'ModelType', 'ExecutionDate']
tblname = 'Adaptive.ForecastHistory_AUR'
typ = 'Transformer'
writedate = finish_time
InsertToSQL(outputTotal_IMP, tblname, typ, my_model, writedate, SQLCols)
#post_proc_pred_DF =    PrepForExport(X2, AUR_scl, ['StartDate','AnalysisKey','AUR_Pred'])
#ExportName = TBL+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")   #+'_mae_loss'
#ExportToSQL(post_proc_pred_DF,ExportName)
################################################################################################################################################################
##InsertToSQL(outputTotal_IMP, tblname, typ, mdl, writedate)
##tsdf = post_proc_pred_DF
#def InsertToSQL(tsdf, tblname, typ, mdl, writedate):
#    tsdf['StartDate'] = tsdf['StartDate'].apply(str)
#    #tsdf.to_string(columns=['StartDate'],col_space=100)
#    writedate = str(writedate)
#    #if mdl.damped == True:
#    #    mdldesc= 'Damped'
#    #else:
#    #    mdldesc= 'Non-Damped'
#    mdldesc = 'Damped' + ', ETS method = '+mdl['method']+', smoothing_level '+str(mdl['smoothing_level'])
#    tsdf['ExecutionType'], tsdf['ModelType'], tsdf['ExecutionDate'] = typ, mdldesc, writedate
#    tsdf.to_sql(name=tblname.split('.')[1], con=engine,if_exists='append',schema=tblname.split('.')[0],index=False, method=None) #,chunksize=10
# #def PrepForExportTest(df_numpy, PredCols):
# #   post_proc_pred = pd.DataFrame(df_numpy,index=valid_set_Sold_uncapped.index[:66],columns=ForecastableSeries)
# #   #post_proc_pred = scl.inverse_transform(TimeSeries(tmp)).pd_dataframe()#.reset_index()
# #   post_proc_pred.columns =  ForecastableSeries
# #   post_proc_pred.reset_index(inplace=True)
# #   output = pd.melt(post_proc_pred,id_vars=['StartDate'], value_vars=post_proc_pred.columns[1:])
# #   output.columns = PredCols
# #   return output
#################################################################################



##tmp = pd.DataFrame(X2,index=valid_set_Sold_uncapped.index[:CHUNK_OUTPUT_LENGTH],columns=ForecastableSeries)
####post_proc_pred = IMP_scl.inverse_transform(tmp)
##post_proc_pred = AUR_scl.inverse_transform(TimeSeries(tmp)).pd_dataframe()#.reset_index()
##post_proc_pred.columns =  ForecastableSeries
##post_proc_pred.reset_index(inplace=True)
###ExportCols =   ForecastableSeries.copy()
###ExportCols.insert(0,'StartDate')     #post_proc_pred.columns = ForecastableSeries.insert('StartDate') 
###post_proc_pred[post_proc_pred<=0.0] = 0                                                    #Still Trouble
###post_proc_pred_DF = pd.DataFrame(post_proc_pred,index=valid_set_Sold_uncapped.index[:CHUNK_OUTPUT_LENGTH],columns=ForecastableSeries)#.reset_index() 
###post_proc_pred_DF = pd.DataFrame(post_proc_pred,index=valid_set_Sold_uncapped.index[:CHUNK_OUTPUT_LENGTH],columns=ForecastableSeries)#.reset_index() 
##post_proc_pred_DF = pd.melt(post_proc_pred,id_vars=['StartDate'], value_vars=post_proc_pred.columns[1:])
##post_proc_pred_DF.columns = ['StartDate','AnalysisKey','AUR_Pred']
###post_proc_pred_DF['StartDate'] = pd.DatetimeIndex(post_proc_pred_DF['StartDate']) 
###post_proc_pred_DF['StartDate'] = str(post_proc_pred_DF['StartDate']) 

##ExportName = TBL+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")   #+'_mae_loss'
##ExportToSQL(post_proc_pred_DF,ExportName)

#########################################################################################
##tmp = pd.DataFrame(X1,index=valid_set_Sold_uncapped.index[:CHUNK_OUTPUT_LENGTH],columns=ForecastableSeries)
####post_proc_pred = IMP_scl.inverse_transform(tmp)
##post_proc_pred = IMP_scl.inverse_transform(TimeSeries(tmp)).pd_dataframe()#.reset_index()
##post_proc_pred.columns =  ForecastableSeries
##post_proc_pred.reset_index(inplace=True)
###ExportCols =   ForecastableSeries.copy()
###ExportCols.insert(0,'StartDate')     #post_proc_pred.columns = ForecastableSeries.insert('StartDate') 
###post_proc_pred[post_proc_pred<=0.0] = 0                                                    #Still Trouble
###post_proc_pred_DF = pd.DataFrame(post_proc_pred,index=valid_set_Sold_uncapped.index[:CHUNK_OUTPUT_LENGTH],columns=ForecastableSeries)#.reset_index() 
###post_proc_pred_DF = pd.DataFrame(post_proc_pred,index=valid_set_Sold_uncapped.index[:CHUNK_OUTPUT_LENGTH],columns=ForecastableSeries)#.reset_index() 
##post_proc_pred_DF = pd.melt(post_proc_pred,id_vars=['StartDate'], value_vars=post_proc_pred.columns[1:])
##post_proc_pred_DF.columns = ['StartDate','AnalysisKey','TOTAL_IMP_Pred']
###post_proc_pred_DF['StartDate'] = pd.DatetimeIndex(post_proc_pred_DF['StartDate']) 
###post_proc_pred_DF['StartDate'] = str(post_proc_pred_DF['StartDate']) 

##ExportName = TBL+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")   #+'_mae_loss'
##ExportToSQL(post_proc_pred_DF,ExportName)

