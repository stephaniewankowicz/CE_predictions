#!/usr/bin/env python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils import data
from torch import nn
device = torch.device("cpu")


# In[217]:


#PREP THE DATA


# In[218]:


RES_MAX_ACC = {'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0,'C': 167.0, 'Q': 225.0, 'E': 223.0, 'G': 104.0,'H': 224.0, 'I': 197.0, 'L': 201.0, 'K': 236.0,                 'M': 224.0, 'F': 240.0, 'P': 159.0, 'S': 155.0,                 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0} 



def create_AH_key(AH_pairs):
    AH_key1 = AH_pairs[['Apo']]
    AH_key2 = AH_pairs[['Holo']]
    AH_key2.columns = ['PDB']
    AH_key1.columns = ['PDB']
    AH_key1['Apo_Holo'] = 'Apo'
    AH_key2['Apo_Holo'] = 'Holo'
    AH_key = pd.concat([AH_key1, AH_key2])
    return AH_key

def merge_apo_holo_df(df):
	df = df.merge(AH_key, on=['PDB'])
	df_holo = df[df['Apo_Holo'] == 'Holo']
	df_apo = df[df['Apo_Holo'] == 'Apo']
	test = df_holo.merge(AH_pairs, left_on='PDB', right_on='Holo')
	df_merged = test.merge(df_apo, left_on=['Apo', 'chain', 'resi'], right_on=['PDB', 'chain', 'resi'])  
	df_merged = df_merged.drop_duplicates()
	return df_merged

pairs = pd.read_csv('/Users/stephaniewanko/Downloads/temp/recheck/final_qfit_pairs.txt', sep='\t')
AH_pairs = pairs.drop_duplicates()
AH_key = create_AH_key(AH_pairs)

os.chdir('/Users/stephaniewanko/Downloads/temp/recheck/normalized')

all_files = glob.glob("*_methyl.out") #read in full protein files

li = []
pdb_remove =[]

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, sep=',', header=0)
    df['PDB'] = filename[0:4]
    if len(df.index) < 30:
        print(filename[54:58])
        print(len(df.index))
        pdb_remove.append(filename[54:58])
    li.append(df)

order_all = pd.concat(li, axis=0, ignore_index=True)


order_all = order_all[order_all['PDB'].isin(AH_key['PDB'])]
order_all = order_all[~order_all['PDB'].isin(pdb_remove)]



merged_all = merge_apo_holo_df(order_all)


order_all_sum = pd.DataFrame()
n = 1
for i in merged_all['Holo'].unique():
  order_all_sum.loc[n,'Holo'] = i
  order_all_sum.loc[n,'length'] = len(merged_all[merged_all['Holo']==i].index)
  order_all_sum.loc[n, 'Min_OP'] = merged_all[merged_all['Holo']==i]['s2calc_x'].min()
  order_all_sum.loc[n, 'Max_OP'] = merged_all[merged_all['Holo']==i]['s2calc_x'].max()
  order_all_sum.loc[n, 'Quartile1'] = merged_all[merged_all['Holo']==i]['s2calc_x'].quantile(0.25)
  order_all_sum.loc[n, 'Quartile3'] = merged_all[merged_all['Holo']==i]['s2calc_x'].quantile(0.75)
  order_all_sum.loc[n, 'Median'] = merged_all[merged_all['Holo']==i]['s2calc_x'].median()
  n += 1 

os.chdir('/Users/stephaniewanko/Downloads/temp/recheck/normalized')
all_files = glob.glob("*_10.0_order_param_subset.csv")
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, sep=',', header=0)
    df['PDB'] = filename[0:4]
    li.append(df)
order_10 = pd.concat(li, axis=0, ignore_index=True)

df = order_10.merge(AH_key, on=['PDB'])
df_holo = df[df['Apo_Holo'] == 'Holo']
df_apo = df[df['Apo_Holo'] == 'Apo']
test = df_holo.merge(AH_pairs, left_on='PDB', right_on='Holo')
df_merged = test.merge(df_apo, left_on=['Apo', 'chain', 'resi'], right_on=['PDB', 'chain', 'resi'])  
merged_10 = df_merged.drop_duplicates()

os.chdir('/Users/stephaniewanko/Downloads/temp/recheck/normalized')
all_files = glob.glob("*_5.0_order_param_subset.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, sep=',', header=0)
    df['PDB'] = filename[0:4]
    li.append(df)
order_5 = pd.concat(li, axis=0, ignore_index=True)


df = order_5.merge(AH_key, on=['PDB'])
df_holo = df[df['Apo_Holo'] == 'Holo']
df_apo = df[df['Apo_Holo'] == 'Apo']
test = df_holo.merge(AH_pairs, left_on='PDB', right_on='Holo')
df_merged = test.merge(df_apo, left_on=['Apo', 'chain', 'resi'], right_on=['PDB', 'chain', 'resi'])  
merged_5 = df_merged.drop_duplicates()

order_far = pd.merge(order_all, order_10, how='left', indicator=True).query("_merge == 'left_only'")


df = order_far.merge(AH_key, on=['PDB'])
df_holo = df[df['Apo_Holo'] == 'Holo']
df_apo = df[df['Apo_Holo'] == 'Apo']
test = df_holo.merge(AH_pairs, left_on='PDB', right_on='Holo')
df_merged = test.merge(df_apo, left_on=['Apo', 'chain', 'resi'], right_on=['PDB', 'chain', 'resi'])  
merged_far = df_merged.drop_duplicates()



merged_far['Difference'] = merged_far['s2calc_x'] - merged_far['s2calc_y']
merged_5['Difference'] = merged_5['s2calc_x'] - merged_5['s2calc_y']
merged_10['Difference'] = merged_10['s2calc_x'] - merged_10['s2calc_y']
merged_all['Difference'] = merged_all['s2calc_x'] - merged_all['s2calc_y']


philic =['ASP', 'GLU', 'LYS', 'ARG', 'GLN', 'ASN', 'HIS', 'SER', 'THR', 'CYS']
phobic =['ALA', 'ILE', 'LEU', 'MET', 'PHE', 'VAL', 'PRO', 'GLY']
both = ['MET', 'TYR', 'MET']


order5_summary = pd.DataFrame()
n = 1
for i in merged_all['Holo'].unique():
    tmp = merged_5[merged_5['Holo'] == i]
    for a in tmp['Apo'].unique():
        tmp2 = tmp[tmp['Apo'] == a]
        order5_summary.loc[n, 'Holo'] = i
        order5_summary.loc[n, 'Apo'] = a
        order5_summary.loc[n, 'Average_5_diff'] = tmp2['Difference'].mean()
        order5_summary.loc[n, 'Average_5_holo_OP'] = tmp2['s2calc_x'].mean()
        order5_summary.loc[n, 'Num_residues_5'] = len(tmp2.index)
        order5_summary.loc[n, 'Num_phobic'] = len(tmp2[tmp2['resn_x'].isin(phobic)])
        order5_summary.loc[n, 'Num_phylic'] = len(tmp2[tmp2['resn_x'].isin(philic)])
        n += 1

os.chdir('/Users/stephaniewanko/Downloads/temp/recheck/other_files')
all_files = glob.glob("*_qFit_sasa.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, sep=',', header=0)
    df['PDB'] = filename[0:4]
    if len(df.index) < 30:
    	print(filename[54:58])
    	print(len(df.index))
    li.append(df)
sasa = pd.concat(li, axis=0, ignore_index=True)

for index, row in sasa.iterrows():
 	sasa.loc[index,'RSA'] = float(row['ss'])/RES_MAX_ACC[row['aa']]


sasa['Solvent_Exposed'] = sasa['RSA'].apply(lambda x: np.where(x < 0.20, 'Not Solved Exposed', 'Solvently Exposed'))

order10_summary = pd.DataFrame()
n = 1
for i in merged_all['Holo'].unique():
    tmp = merged_10[merged_10['Holo'] == i]
    for a in tmp['Apo'].unique():
        tmp2 = tmp[tmp['Apo'] == a]
        order10_summary.loc[n, 'Holo'] = i
        order10_summary.loc[n, 'Apo'] = a
        order10_summary.loc[n, 'Average_10_diff'] = tmp2['Difference'].mean()
        order10_summary.loc[n, 'Average_10_holo_OP'] = tmp2['s2calc_x'].mean()
        order10_summary.loc[n, 'Num_residues_10'] = len(tmp2.index)
        n += 1

orderall_summary = pd.DataFrame()
n = 1
for i in merged_all['Holo'].unique():
    tmp = merged_all[merged_all['Holo'] == i]
    for a in tmp['Apo'].unique():
    	tmp2 = tmp[tmp['Apo'] == a]
    	orderall_summary.loc[n, 'Holo'] = i
    	orderall_summary.loc[n, 'Apo'] = a
    	orderall_summary.loc[n, 'Average_all_diff'] = tmp2['Difference'].mean()
    	orderall_summary.loc[n, 'Average_all_holo_OP'] = tmp2['s2calc_x'].mean()
    	n += 1


orderfar_summary = pd.DataFrame()
n = 1
for i in merged_far['Holo'].unique():
    tmp = merged_far[merged_far['Holo'] == i]
    for a in tmp['Apo'].unique():
    	tmp2 = tmp[tmp['Apo'] == a]
    	orderfar_summary.loc[n, 'Holo'] = i
    	orderfar_summary.loc[n, 'Apo'] = a
    	orderfar_summary.loc[n, 'Average_far_diff'] = tmp2['Difference'].mean()
    	orderfar_summary.loc[n, 'Average_far_holo_OP'] = tmp2['s2calc_x'].mean()
    	n += 1


sasa_exposed = sasa[sasa['Solvent_Exposed']=='Solvently Exposed']
sasa_notexposed = sasa[sasa['Solvent_Exposed']=='Not Solved Exposed']

far_not_solvent = pd.DataFrame()
for i in merged_far['Holo'].unique():
    tmp = merged_far[merged_far['Holo']==i]
    tmp_sasa = sasa_notexposed[sasa_notexposed['PDB']==i]
    tmp2 = tmp[tmp['resi'].isin(tmp_sasa['resnum']) & tmp['chain'].isin(tmp_sasa['chain'])]
    far_not_solvent = far_not_solvent.append(tmp2, ignore_index=True)


orderfar_solvent_sum = pd.DataFrame()
n = 1
for i in far_not_solvent['Holo'].unique():
    tmp = far_not_solvent[far_not_solvent['Holo'] == i]
    for a in tmp['Apo'].unique():
        tmp2 = tmp[tmp['Apo'] == a]
        orderfar_solvent_sum.loc[n, 'Holo'] = i
        orderfar_solvent_sum.loc[n, 'Apo'] = a
        orderfar_solvent_sum.loc[n, 'Average_far_diff'] = tmp2['Difference'].mean()
        orderfar_solvent_sum.loc[n, 'Average_far_holo_OP'] = tmp2['s2calc_x'].mean()
        orderfar_solvent_sum.loc[n, 'Num_residues_far'] = len(tmp2.index)
        n += 1


combined_summary = orderall_summary.merge(order5_summary, on=['Apo', 'Holo'])
combined_summary1 = combined_summary.drop_duplicates()

combined_summary = combined_summary.drop_duplicates()
combined_summary = combined_summary.merge(orderfar_summary, on=['Apo', 'Holo'])

combined_summary = combined_summary.drop_duplicates()
combined_summary['Difference_all_5'] = combined_summary['Average_all_diff'] - combined_summary['Average_5_diff']
combined_summary['Difference_far_5'] = combined_summary['Average_far_diff'] - combined_summary['Average_5_diff']
combined_summary['Difference_Holo_far_5'] = combined_summary['Average_far_holo_OP'] - combined_summary['Average_5_holo_OP']
combined_summary['Difference_Holo_all_5'] = combined_summary['Average_all_holo_OP'] - combined_summary['Average_5_holo_OP']

combined_summary = combined_summary[combined_summary['Apo'] !='1uj4']

combined_summary.to_csv('combined_summary_paper.csv', index=False)


not_sol_far_5 = orderfar_solvent_sum.merge(order5_summary, on=['Apo', 'Holo'])
not_sol_far_5 = not_sol_far_5.drop_duplicates()
not_sol_far_5['Difference_far_5'] = not_sol_far_5['Average_far_diff'] - not_sol_far_5['Average_5_diff']
not_sol_far_5['Difference_Holo_far_5'] = not_sol_far_5['Average_far_holo_OP'] - not_sol_far_5['Average_5_holo_OP']


#combined_summary['Difference_far_5'] - combined_summary['Average_5_diff'] 

#This needs to be pairs
NN_input_df = pd.DataFrame()
NN_input_df['PDB'] = order5_summary['Holo']
NN_input_df['Average_5'] = order5_summary['Average_5_holo_OP']
NN_input_df['Binding_hydrophobic'] = order5_summary['Num_phobic']
NN_input_df['Binding_hydrophilic'] = order5_summary['Num_phylic']

NN_input_df['Protein_size'] = order_all_sum['length']
NN_input_df['Median_OP'] = order_all_sum['Median']
NN_input_df['Q1_OP'] = order_all_sum['Quartile1']
NN_input_df['Q3_OP'] = order_all_sum['Quartile3']

NN_input_df['Num_distant'] = order10_summary['Num_residues_10']
NN_input_df['Predict'] = combined_summary['Difference_far_5'] - combined_summary['Average_5_diff'] 
NN_input_df.head()

scaler = StandardScaler()
NN_input_df[['Average_5', 'Num_phobic', 'Num_phylic', 'Protein_size', 'Median_OP', 'Q1_OP', 'Q3_OP', 'Num_distant']] = scaler.fit_transform(NN_input_df[['Average_5', 'Num_phobic', 'Num_phylic', 'Protein_size', 'Median_OP', 'Q1_OP', 'Q3_OP', 'Num_distant']])
NN_out = NN_input_df[['Predict', 'PDB']]
NN_out[['Predict']] = scaler.fit_transform(NN_input_df[['Predict']])


#define variables like input size, hidden unit, output size, batch size, and the learning rate.
n_input, n_hidden, n_out, batch_size, learning_rate = 8, 15, 1, 100, 0.01


tmp_x = NN_input_df.drop(['Predict','PDB'], axis =1).values
tmp_y = NN_out.drop(['PDB'], axis=1).values.reshape(-1,1) 

tmp_x_tensor = torch.tensor(tmp_x, dtype=torch.float32)
tmp_y_tensor = torch.tensor(tmp_y, dtype=torch.float32)


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


data_iter = load_array((tmp_x_tensor, tmp_y_tensor), 20)



net = nn.Linear(8, 1)
#net.weight.data.normal_(0, 0.01)
#net.bias.data.fill_(0)
loss = nn.MSELoss()
# implements a stochastic gradient descent optimization method
trainer = torch.optim.SGD(net.parameters(), lr=0.01)


net = nn.Sequential(nn.Linear(n_input, 10),
                      nn.ReLU(),
                      nn.Linear(10, 10),
                      nn.ReLU(),
                      nn.Linear(10, n_out),
                      nn.Sigmoid())


num_epochs = 100
losses = []
running_loss = 0.0
for epoch in range(num_epochs):
 
    for X, y in data_iter:
 
        l = loss(net(X) ,y)
 
        trainer.zero_grad() #sets gradients to zero
 
        l.backward() # back propagation
 
        trainer.step() # parameter update
        losses.append(l.item())
        print(f'epoch {epoch + 1}, loss {l:f}')


plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
#plt.show()


class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = torch.nn.Linear(8, 10)  # 8-(10-10)-1
    self.hid2 = torch.nn.Linear(10, 10)
    self.oupt = torch.nn.Linear(10, 1)

    torch.nn.init.xavier_uniform_(self.hid1.weight)
    torch.nn.init.zeros_(self.hid1.bias)
    torch.nn.init.xavier_uniform_(self.hid2.weight)
    torch.nn.init.zeros_(self.hid2.bias)
    torch.nn.init.xavier_uniform_(self.oupt.weight)
    torch.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = torch.relu(self.hid1(x))
    z = torch.relu(self.hid2(z))
    z = self.oupt(z)  # no activation
    return z


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(8, 10)  # 8-(10-10)-1
        self.hid2 = torch.nn.Linear(10, 10)
        self.oupt = torch.nn.Linear(10, 1)


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(13, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)


model = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, n_out),
                      nn.Sigmoid())




# plt.plot(l)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.title("Learning rate %f"%(learning_rate))
# plt.show()




