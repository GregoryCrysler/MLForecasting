import sys
import pandas as pd
import numpy as np
import sqlalchemy
import pandas.io.sql as psql
import pyodbc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import joblib
import pickle
import os
import Settings

import darts.utils as utl
from darts import TimeSeries
from darts.metrics import mape, smape, mae
from darts.models import NBEATSModel
#from darts.utils.data.sequential_dataset import SequentialDataset
from darts.utils.data.horizon_based_dataset import HorizonBasedDataset

from torch.nn import L1Loss
import torch
import torch.optim as optim

from tqdm import tqdm
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer , robust_scale, RobustScaler
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', 20)

##################################################################################################################################################################


#trainDF = training_set
#validDF = validation_set
#VarThreshold = 5
def FindForecastableSeries (trainDF, validDF, DateThreshold, VarThreshold=5):
    pop = SRC.loc[SRC.StartDate <= DateThreshold, ].groupby(['AnalysisKey'])['Sold_uncapped'].var() > VarThreshold
    keys = set(pop.loc[pop.values==True].index.values)
    ValidSeries = list(set(trainDF.index.values).intersection(set(validDF.index.values)).intersection(keys))
    return ValidSeries


def array_to_seq(arr, date_index):
    ts_sequence = []
    for i in tqdm(range(len(arr))):
        ts_sequence.append(TimeSeries.from_times_and_values(date_index, arr.iloc[i, :]))
        #ts_sequence.append(TimeSeries.from_times_and_values(date_index, arr[i, :]))
    return ts_sequence


def mae_loss(forecast, target):
    return torch.mean(torch.abs(forecast - target))

def pen_mae_loss(forecast, target, penalty=2):
    if torch.mean(forecast - target) > 0:
        ls = torch.mean(torch.abs(forecast - target))
    else:
        ls = penalty * torch.mean(torch.abs(forecast - target))
    return ls

def smape_loss(forecast, target):
    return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                          torch.abs(forecast.data) + torch.abs(target.data)))

def divide_no_nan(a, b):
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

def display_forecast(pred_series, ts_transformed, forecast_type, start_date=None):
    plt.figure(figsize=(8,5))
    if (start_date):
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label='actual')
    pred_series.plot(label=('historic ' + forecast_type + ' forecasts'))
    plt.title('R2: {}'.format(r2_score(ts_transformed.univariate_component(0), pred_series)))
    plt.legend();


#tsdf=post_proc_pred_DF
#tblname = TBL+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
def ExportToSQL(tsdf,tblname):
    tsdf.to_sql(tblname.split('.')[1], con=engine,if_exists='replace',schema=tblname.split('.')[0],chunksize=1000)

#tsdf = post_proc_pred_DF
def InsertToSQL(tsdf, tblname, typ, mdl, writedate):
    tsdf['StartDate'] = tsdf['StartDate'].apply(str)    
    writedate = str(writedate)
    mdldesc = 'input_chunk = ' + str(mdl.input_chunk_length) + ', num_blocks = '+str(mdl.num_blocks)+ ', num_layers = '+str(mdl.num_layers)+ ', batch_size = '+str(model.batch_size)
    tsdf['ExecutionType'], tsdf['ModelType'], tsdf['ExecutionDate'] = typ, mdldesc, writedate
    tsdf.to_sql(name=tblname.split('.')[1], con=engine,if_exists='append',schema=tblname.split('.')[0],index=False, method=None) #,chunksize=10


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
##################################################################################################################################################################
#X = training_set

class QuantileScaler(TransformerMixin):

    def __init__(self, minimum=None):
        self.minimum = minimum

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.q1_ = np.quantile(X,0.1,axis=1).reshape(-1,1)
        self.q9_ = np.quantile(X,0.9,axis=1).reshape(-1,1)
        return self

    def transform(self, X, y=None):
        return np.divide(X - self.q1_, self.q9_ - self.q1_ , out=np.zeros_like(X), where=(self.q9_ - self.q1_) != 0.0)

    def inverse_transform(self, X):
        return (X * (self.q9_ - self.q1_ )) + self.q1_

class LocalStdScaler(TransformerMixin):

    def __init__(self, minimum=None):
        self.minimum = minimum

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.avg_ = X.mean(axis=1).reshape(-1,1)
        self.std_ = X.std(axis=1).reshape(-1, 1)
        return self

    def transform(self, X, y=None):
        return np.divide(X - self.avg_, self.std_, out=np.zeros_like(X), where=(self.std_) != 0.0)

    def inverse_transform(self, X):
        return (X * self.std_) + self.avg_

class LocalMinMaxScaler(TransformerMixin):

    def __init__(self, minimum=None):
        self.minimum = minimum

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
              X = X.values

        self.min_ = X.min(axis=1).reshape(-1, 1) if self.minimum is None else self.minimum
        self.max_ = X.max(axis=1).reshape(-1, 1)

        return self

    def transform(self, X, y=None):
        return np.divide(X - self.min_, self.max_ - self.min_, out=np.zeros_like(X), where=(self.max_ - self.min_) != 0.0)

    def inverse_transform(self, X):
        return X * (self.max_ - self.min_) + self.min_

class VariableSizeImputer(TransformerMixin):
    def __init__(self, fill_value=0.0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill_value)

    def inverse_transform(self, X):
        return X

##################################################################################################################################################################
#Testing Vars & Misc Pre-Production Steps
ProcessLogSP ='Adaptive.up_ProcessLogV5'
tblname='Adaptive.ForecastHistory'

##################################################################################################################################################################


#cnxn = pyodbc.connect('DSN=InnovarQA')
cnxn = pyodbc.connect('Driver={SQL Server};'
                        'Server=172.31.75.52;'
                        'Database=InnVision_QA;'
                        + Settings.UIDSet
                        + Settings.PWDSet
                        + 'Trusted_Connection=no;')
cursor = cnxn.cursor()
engine = sqlalchemy.create_engine('mssql+pyodbc://'+Settings.UIDSet[4:-1]+':'+Settings.PWDSet[4:-1]+'@AWSDEVSQLN1/InnVision_QA?driver=ODBC Driver 17 for SQL Server', fast_executemany=True)


#TBL = 'Adaptive.BaseAnalysis_Charlotte_vw'
TBL = 'Adaptive.BaseAnalysis'
Cols = 'AnalysisKey,StartDate,FlagDataType ,WeekIDX,BroadcastWeekKey,ZoneCode,NetworkCode,DayPart ,Avails, Sold_uncapped'
SRC = psql.read_sql('SELECT '+Cols+' FROM '+TBL, cnxn)#.fillna(0)
#########################################################################
#Formula for Required Overlap in Train/Validation (yes, it's weird)
##(self.lookback + self.max_lh) * self.output_chunk_length  

LOOKBACK = 5 #3
LOOKAHEAD = 3
CHUNK_OUTPUT_FINAL_LENGTH = 66 #15 #66
CHUNK_OUTPUT_LENGTH = 15
CHUNK_INPUT_LENGTH = CHUNK_OUTPUT_LENGTH*LOOKBACK
Pacing_Length = 15


#Horizon = 15                            #For Initial Testing (required Horizon = 66)
#TimeEnd = pd.to_datetime((SRC.loc[SRC.FlagDataType == 'Train',['StartDate']].max().values[0]))
TimeEnd = pd.to_datetime(SRC['StartDate'].max())                                                                #Prod
FcastEnd =   TimeEnd +  pd.Timedelta(str(CHUNK_OUTPUT_FINAL_LENGTH)+'W')                                        #Prod
#ValidationTimeEnd = TimeEnd + pd.Timedelta(str(CHUNK_OUTPUT_FINAL_LENGTH)+'W') 
#ValidationTimeBegin =   ValidationTimeEnd - pd.Timedelta(str((LOOKBACK + LOOKAHEAD )*CHUNK_OUTPUT_LENGTH)+'W') 
ValidationTimeEnd = TimeEnd                                                                                     #Prod
ValidationTimeBegin =   TimeEnd -  pd.Timedelta(str(Pacing_Length)+'W')                                         #Prod

training_set =  pd.pivot_table(SRC[['Sold_uncapped','AnalysisKey','StartDate']]
               , values=['Sold_uncapped'], fill_value=0, index=['AnalysisKey'],columns=['StartDate'], aggfunc=np.max)
training_set.columns = pd.DatetimeIndex([y for x,y in training_set.columns.values])



validation_set  =  pd.pivot_table(SRC.loc[(SRC.StartDate >= ValidationTimeBegin) ,['Sold_uncapped','AnalysisKey','StartDate']]
                , values=['Sold_uncapped'], fill_value=0, index=['AnalysisKey'],columns=['StartDate'], aggfunc=np.max)
validation_set.columns = pd.DatetimeIndex([y for x,y in validation_set.columns.values])

#DateThreshold = TimeEnd - pd.Timedelta(str(CHUNK_OUTPUT_LENGTH)+'W')+pd.Timedelta('1W') 
DateThreshold = TimeEnd - pd.Timedelta(str(Pacing_Length)+'W') 
ForecastableSeries = FindForecastableSeries(training_set,validation_set,DateThreshold)


##################################################################################################################################################################

training_set_NBEATS, validation_set_NBEATS = training_set.loc[ForecastableSeries], validation_set.loc[ForecastableSeries]
imputer =   VariableSizeImputer() #SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0)
scaler =    QuantileScaler()
#scaler =  LocalStdScaler()
pipe = make_pipeline(imputer,scaler)
pipe.fit(training_set_NBEATS)

training_sequence=array_to_seq(pipe.transform(training_set_NBEATS), training_set_NBEATS.columns)
validation_sequence=array_to_seq(pipe.transform(validation_set_NBEATS), validation_set_NBEATS.columns)



training_dataset = HorizonBasedDataset(target_series = training_sequence,
                                       output_chunk_length=CHUNK_OUTPUT_LENGTH,
                                       lh=(1,LOOKAHEAD),
                                       lookback=LOOKBACK)

validation_dataset = HorizonBasedDataset(target_series = validation_sequence,
                                         output_chunk_length=CHUNK_OUTPUT_LENGTH,
                                         lh=(1,LOOKAHEAD),
                                         lookback=LOOKBACK)

##################################################################################################################################################################

#Stop (reload model in ScratchPad unless re-training)

source =    'DARTSforecast.py'
start_time = str(datetime.now())
log_type='DARTS'
#params = ''
log_msg = 'Training Period = '+str(training_set.shape[1])
rows = training_set.shape[0]
log_message(source, start_time, log_type, log_msg, rows)
##################################################################################################################################################################

BATCH_SAMPLE_RATE = 0.33
N_EPOCHS =30                                                #80
NUM_STACKS=2                                                    #2 required if         generic_architecture=False
NUM_BLOCKS=5
NUM_LAYERS=5
LAYER_WIDTH=[1024,1024]                    #256 (small)#If List, Len Must be equal to Num Stacks[1...i]
MODEL_NAME = 'NBEATS'
BATCH_SIZE =    int(BATCH_SAMPLE_RATE * training_set_NBEATS.shape[0])

mdl_name = 'nbeats_run_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'.dump'

model = NBEATSModel(input_chunk_length=CHUNK_OUTPUT_LENGTH*LOOKBACK,
                    output_chunk_length=CHUNK_OUTPUT_LENGTH,
                    nr_epochs_val_period=5,
                    num_stacks=NUM_STACKS,
                    num_blocks=NUM_BLOCKS,
                    num_layers=NUM_LAYERS,
                    layer_widths=LAYER_WIDTH,
                    generic_architecture=False,
                    model_name=mdl_name,
                    batch_size=BATCH_SIZE,
                    #log_tensorboard=True,                              #Tensorboard to log the different parameters
                    #torch_device_str='coda:0',
                    n_epochs=N_EPOCHS,
                    loss_fn= mae_loss,                              #mae_loss,
                    trend_polynomial_degree = 2)                        #3

# model fitting
print('Started @ '+str(datetime.now()))
#model.fit_from_dataset(train_dataset=training_dataset, val_dataset=validation_dataset,verbose=True) 
model.fit_from_dataset(train_dataset=training_dataset, verbose=True)                                        #Prod
print('Finished @ '+str(datetime.now()))

model._save_model('C:/ModelData/',mdl_name,N_EPOCHS)
finish_time = str(datetime.now())
#(self.lookback + self.max_lh) * self.output_chunk_length       ("`(lookback + max_lh) * H` (0-th)")
#(4 + 2) * 15
##################################################################################################################################################################

pred = model.predict(n=CHUNK_OUTPUT_FINAL_LENGTH, series=training_sequence)
X = np.array([p.values().squeeze() for p in pred]) #X = pred.pd_series()
post_proc_pred = pipe.inverse_transform(X)
post_proc_pred[post_proc_pred<=0.0] = 0
#post_proc_pred_DF = pd.DataFrame(post_proc_pred,index=training_set_NBEATS.index, columns=validation_set_NBEATS.columns[len(validation_set_NBEATS.columns)-CHUNK_OUTPUT_FINAL_LENGTH:]).reset_index() 
post_proc_pred_DF = pd.DataFrame(post_proc_pred,index=training_set_NBEATS.index, columns = pd.Series(np.arange(TimeEnd+  pd.Timedelta('1W'), FcastEnd+  pd.Timedelta('1W'), timedelta(days=7)))).reset_index() 

post_proc_pred_DF = pd.melt(post_proc_pred_DF,id_vars=['AnalysisKey'], value_vars=post_proc_pred_DF.columns[1:])
post_proc_pred_DF.columns = ['AnalysisKey','StartDate','Sold_uncapped_Pred']


#ExportToSQL(post_proc_pred_DF,ExportName)          #Error
#post_proc_pred_DF.to_csv('C:/ModelData/'+ExportName+'.csv')

tblname='Adaptive.ForecastHistory'
typ = 'NBEATS'
mdl = model
writedate = finish_time
InsertToSQL(post_proc_pred_DF, tblname, typ, mdl, writedate)



##################################################################################################################################################################
#mdl_name = 'checkpoint_50.pth.tar'
#os.getcwd()
#'C:\\Users\\P3018584\\source\\repos\\DARTSforecast\\DARTSforecast'
#dir = '\\nbeats_run_04_15_2021_16_22_15.dump\\'
#model_load = model.load_from_checkpoint(mdl_name,os.getcwd())