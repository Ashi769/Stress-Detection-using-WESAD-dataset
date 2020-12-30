import numpy as np
import pandas as pd
import os
import pickle
from scipy import stats
import scipy.stats
from scipy import stats
import scipy.signal as scisig
import scipy.stats
import cvxEDA
import math

d1 = {}
features_names = ['ACC','BVP','EDA','TEMP','label','Resp']
features_val = [32,64,4,4,700,700]
for ind in range(6):
    d1[features_names[ind]]=features_val[ind]
feat_names = []

def output_the_sample(counts):
    print('Number of samples per class:')
    for label, nn in zip(counts.index, counts.values):
        if label == 0:
             print(f'amusement : {nn}')
        elif label ==1:
            print(f'baseline : {nn}')
        else:
            print(f'stress : {nn}')

def get_allin_list(lis):
    data_file_list = []
    for ss in lis:
        data_file = pd.read_csv(f'S_{ss}.csv')
        data_file['subject']=ss
        data_file_list.append(data_file)
    return data_file_list  
    
def cal_x_y(ww,w,label):
    ww[0] = 'net_acc'
    w.columns = ww
    wstats = find_window_stats(w,label)
    xx = pd.DataFrame(wstats)
    x = xx.drop('label',axis=0)
    y = x['label'][0]
    x.drop('label', axis=1, inplace=True)
    return w,x,y

def lis_ofind_col(x):
    arr = []
    mydict = []
    for i in x.index:
        for j in x.columns:
            mydict.append((i,j))
    for (i,j) in mydict:
        arr.append('_'.join([i,j]))
    return arr

def cal_wdata_file(x,y,w,feat_names):
    xx = x.values.flatten()
    wdata_file = pd.DataFrame(xx)
    wdata_file = wdata_file.T
    wdata_file.columns = feat_names
    yy = pd.DataFrame({'label': y}, index=[0])
    wdata_file = pd.concat([wdata_file,yy], axis=1)
    wdata_file['BVP_peak_freq'],wdata_file['TEMP_slope'] = find_max_amp_freq(w['BVP'].dropna()),find_the_m(w['TEMP'].dropna())
    return wdata_file

def get_lis1(data,n_w,label,lenn):
    global feat_names
    ans = []
    for i in range(n_w):
        st = lenn*i
        lt = lenn*(i+1)
        w = data[st:lt]
        dd = np.sqrt((w['ACC_x']*w['ACC_x'] + w['ACC_y']*w['ACC_y'] + w['ACC_z']*w['ACC_z']))
        w = pd.concat([w,dd])
        w,x,y = cal_x_y(list(w.columns),w,label)
        if len(feat_names)==0:
            feat_names = lis_ofind_col(x)
        ans.append(cal_wdata_file(x,y,w,feat_names))
    return ans
    
def find_all_sample(lis):
    ans = []
    for (data, n_w, label) in lis:
        samples = get_lis1(data,n_w,label,21000)
        ans.append(pd.concat(samples))
    return ans[0],ans[1],ans[2]

def cal_data_frame(x,y):
    return pd.DataFrame(x,columns=y)

def calF(di,lab):
    my_keys = [di['TEMP'],di['BVP'],di['ACC'],di['Resp'],lab,di['EDA']]
    my_vals = [['TEMP'],['BVP'],['ACC_x','ACC_y','ACC_z'],['Resp'],['label'],['EDA']]
    liis = [] 
    for key,val in zip(my_keys,my_vals):
        liis.append(cal_data_frame(key,val))
    for i in liis[2].columns:
        liis[2][i]=filterSignalFIR(liis[2].values)
    liis[5]['EDA']=butter_lowpass_filter(liis[5]['EDA'],1.0,d1['EDA'],6)
    te = []
    nnn = d1['BVP']
    for i in range(len(liis[1])):
        te.append((1/nnn)*i)
    liis[1].index=pd.to_datetime(te,unit='s')
    te = []
    nnn = d1['ACC']
    for i in range(len(liis[2])):
        te.append((1/nnn)*i)
    liis[2].index=pd.to_datetime(te,unit='s')
    te = []
    nnn = d1['Resp']
    for i in range(len(liis[3])):
        te.append((1/nnn)*i)
    liis[3].index=pd.to_datetime(te,unit='s')
    te = []
    nnn = d1['label']
    for i in range(len(liis[4])):
        te.append((1/nnn)*i)
    liis[4].index=pd.to_datetime(te,unit='s')
    te = []
    nnn = d1['EDA']
    for i in range(len(liis[5])):
        te.append((1/nnn)*i)
    liis[5].index=pd.to_datetime(te,unit='s')
    te = []
    nnn = d1['TEMP']
    for i in range(len(liis[0])):
        te.append((1/nnn)*i)
    liis[0].index=pd.to_datetime(te,unit='s')
    liis[5]['EDA_phasic'],liis[5]['EDA_smna'],liis[5]['EDA_tonic'],_,_,_,_=EDA(liis[5]['EDA'])
    df = join_all_liframe(liis[5], liis[1], liis[0],liis[2], liis[3], liis[4])
    df['label']=df['label'].fillna(method='bfill')
    df.reset_index(drop=True,inplace=True)
    gr=df.groupby('label')
    return gr,gr.get_group(1),gr.get_group(2),gr.get_group(3)

def join_all_liframe(a,b,c,d,e,f):
    lis = [c,d,e,f]
    dd=a.join(b,how='outer')
    for i in lis:
        dd = dd.join(i,how='outer')
    return dd

def butter_lowpass_filter(data, c, features, o=5):
    b, a = scisig.butter(o,c/(0.5*features),btype='low',analog=False)
    return scisig.lfilter(b, a, data)

def find_the_m(s):
    return scipy.stats.linregress(np.arange(len(s)),s)[0]

def find_window_stats(data, label=-1):
    return {'mean': np.mean(data), 'std': np.std(data), 'min': np.amin(data), 'max': np.amax(data),'label': label}

def find_max_amp_freq(x):
    f,ff = scisig.periodogram(x,fs=8)
    max_freq = -1
    max_amp = -1000000
    for amp, freq in zip(ff, f):
        if max_amp < amp:
            max_amp = amp
            max_freq = freq
    return max_freq

def filterSignalFIR(eda, cutoff=0.4, num=64):
    f = cutoff / (d1['ACC'] / 2.0)
    return scisig.lfilter(scisig.firwin(num,f), 1, eda)

def EDA(y):
    ff = d1['EDA']
    yy = (y - y.mean())/y.std()
    return cvxEDA.cvxEDA(yy,1/ff)

def get_my_val(x,y,z):
    return int(len(x)/21000),int(len(y)/21000),int(len(z)/21000)

def save_tocsv(x,y,z,i):
    all_s = pd.concat([x, y, z])
    xx = all_s.drop('label', axis=1)
    yy = pd.get_dummies(all_s['label'])
    all_s = pd.concat([xx,yy], axis=1)
    all_s.to_csv(f'S_{i}.csv')
    
def check_type(s):
    return type(s)==str
    
def start_preprocess(n):
    directory_path = '/home/shubham/Desktop/MINI'
    feature_path = os.path.join("",'subject_feats')
    check_for_dirs(directory_path,feature_path)
    for i in range(2,n):
        if i==12:
            continue
        s = 'S'+str(i)
        print('Ongoing preprocessing for',s,'...')
        the_csv_file_path=os.path.join(directory_path,'WESAD')
        with  open(os.path.join(os.path.join(the_csv_file_path, f'S{i}') , f'S{i}.pkl'), 'rb') as file:
            data1 = pickle.load(file,encoding='latin1')
            dict1 = data1['signal']['wrist']
            dict1.update({'Resp': data1['signal']['chest']['Resp']})
            gg, bb, sts, amu = calF(dict1, data1['label'])
            n_bb,n_sts,n_amu = get_my_val(bb,sts,amu)
            bb_samples,sts_samples,amu_samples = find_all_sample([(bb,n_bb,1),(sts,n_sts,2),(amu,n_amu,0)])
            save_tocsv(bb_samples,sts_samples,amu_samples,i)
    all_files = [i for i in range(2,18) if i!=12]
    data_file_list = get_allin_list(all_files)
    data_file = pd.concat(data_file_list)
    s = data_file['0'].astype(str) + data_file['1'].astype(str) + data_file['2'].astype(str)
    #print(type(s))
    if check_type(s) and s.index('1'):
        data_file['label']=s.index('1')
    else:
        data_file['label'] = s.apply(lambda x: x.index('1'))
      
    data_file.drop(['0', '1', '2'], axis=1, inplace=True)
    data_file.reset_index(drop=True, inplace=True)
    data_file.to_csv('combined.csv')
    counts = data_file['label'].value_counts()
    output_the_sample(counts)
    
def check_for_dirs(curr,csv_file):
    if not os.path.exists(curr):
        os.makedirs(curr)

start_preprocess(18)

