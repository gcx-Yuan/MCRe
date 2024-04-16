import numpy as np
import pandas as pd
import torch
import random
from PIL import Image
import pywt

def wave_tran_4(x):
    wave_1, wave_4 = pywt.dwt(x, 'db16')
    wave_1, wave_3 = pywt.dwt(wave_1, 'db16')
    wave_1, wave_2 = pywt.dwt(wave_1, 'db16')
    return wave_1, wave_2, wave_3, wave_4


class Data_Process():
    #print(data)
    def __init__(self, data, train=True, transform=None, target_transform=None, 
                 noise_type=None, INCV_b = 0.2, INCV_c = 0, random_state=0 ):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        
        self.noise_type=noise_type
        
        if self.train:
            self.train_data = torch.tensor(data.drop(['Label'], axis=1).values)
            self.train_labels = torch.tensor(data[['Label']].values)
                        
            df_train_YY_noise =  data[['Label']].copy()
            df_train_YY_noise.rename(columns = {'Label':'Label-noise'},inplace = True)
            ##依据INCV_b和INCV_c重置数据标签
            
            YY_counts = data[['Label']].value_counts(sort=False).tolist()
            YY_change = np.zeros(len(YY_counts))
            for i in range(len(YY_counts)):
                if i == 0:
                    YY_change[i] = int(YY_counts[i]*INCV_b)
                else:
                    YY_change[i] = int(YY_counts[i]*INCV_c)
            
            ####
            rslt = []
            index = []
            
            for i in range(len(YY_counts)):
                rslt.append(df_train_YY_noise[df_train_YY_noise['Label-noise'] == i].index.tolist())
                #print(df_train_YY_noise)
                #print(rslt[i])
                #print(int(YY_change[i]))
                index.append(random.sample(rslt[i], int(YY_change[i])))
    
            for i in range(len(YY_counts)):
                if i == 0:
                    for idx in index[i]:
                        df_train_YY_noise.loc[idx,"Label-noise"] = random.randint(1,(len(YY_counts)-1))#（这里要换成一个随机数）
                else:
                    for idx in index[i]:
                        df_train_YY_noise.loc[idx,"Label-noise"] = 0
            
            
            self.train_noisy_labels = torch.tensor(df_train_YY_noise.values)
            #self.actual_noise_rate = 
            self.noise_or_not = torch.tensor([self.train_noisy_labels[i]==self.train_labels[i] 
                                 for i in range(self.train_noisy_labels.shape[0])])
        
        else:
            self.test_data = torch.tensor(data.drop(['Label'], axis=1).values)
            self.test_labels = torch.tensor(data[['Label']].values)
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_noisy_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        #img = img.numpy()
        
        if self.transform is not None:
            img = self.transform(img)
            #print(img.shape)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)