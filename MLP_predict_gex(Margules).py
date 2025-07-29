import matplotlib.pyplot as plt
import torch.cuda
import numpy as np
import sklearn
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from toolboxandexpiredproject import standarize
from sklearn.model_selection import train_test_split
import datetime
import os
from tqdm import tqdm
import itertools

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.hidden_layer1 = nn.Linear(2,50).to(torch.double)
        self.relu1 = nn.Tanh()
        self.hidden_layer2 = nn.Linear(50,50).to(torch.double)
        self.relu2 = nn.Tanh()
        self.hidden_layer3 = nn.Linear(50,25).to(torch.double)
        self.relu3 = nn.Tanh()
        self.output_layer = nn.Linear(25,1).to(torch.double)
    def forward(self, T,x):
        inputs = torch.cat((T.unsqueeze(1), x.unsqueeze(1)),axis=1) # combined arrays 
        layer1_out = self.hidden_layer1(inputs)
        layer1_out = self.relu1(layer1_out)
        layer2_out = self.hidden_layer2(layer1_out)
        layer2_out = self.relu2(layer2_out)
        layer3_out = self.hidden_layer3(layer2_out)
        layer3_out = self.relu3(layer3_out)
        output = self.output_layer(layer3_out) 
        output = torch.reshape(output,(-1,))
        return output
    
def train(model, dataloader, criterion, optimizer, device, sacle_data,log_vars,ablation_mask):
    model.train()
    total_loss = 0
    total_loss_target = 0
    total_loss_d1 = 0
    total_loss_dT = 0
    total_loss_d2 = 0
    
    x_mean, x_std = sacle_data['x_mean'], sacle_data['x_std']
    T_R_mean, T_R_std = sacle_data['T_R_mean'], sacle_data['T_R_std']

    for batch in tqdm(dataloader, desc='Training', ncols=100):
        T, x, labels, derivative_rx, derivative_rT,derivative_rx_2nd = [b.to(device) for b in batch]
        x.requires_grad_(True)
        T.requires_grad_(True)

        outputs = model(T, x)
        target_loss = criterion(outputs.to(torch.double), labels.to(torch.double))

        derivative = torch.autograd.grad(outputs=model(T, x),inputs=x,grad_outputs=torch.ones_like(outputs),create_graph=True)[0]

        derivativeT = torch.autograd.grad(outputs=model(T, x),inputs=T,grad_outputs=torch.ones_like(outputs), create_graph=True)[0] / T_R_std

        derivative2nd = torch.autograd.grad(outputs=derivative,inputs=x,grad_outputs=torch.ones_like(derivative),create_graph=True)[0]

        loss_d = torch.mean((derivative - derivative_rx)**2)
        loss_d2 = torch.mean((derivative2nd - derivative_rx_2nd)**2)
        loss_dT = torch.mean((derivativeT - derivative_rT)**2)
        loss = ablation_mask[0]*torch.exp(-log_vars[1]) * loss_d + ablation_mask[1]*torch.exp(-log_vars[2]) * loss_dT + torch.exp(-log_vars[0]) *target_loss + ablation_mask[2] * torch.exp(-log_vars[3]) *loss_d2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_target += target_loss.item()
        total_loss_d1 += loss_d.item()
        total_loss_dT += loss_dT.item()
        total_loss_d2 += loss_d2.item()

    torch.save(model, f'{path}/DNN_{dataname}_s_all.pkl')

    return (total_loss / len(dataloader),
            total_loss_target / len(dataloader),
            total_loss_d1 / len(dataloader),
            total_loss_dT / len(dataloader),
            total_loss_d2 / len(dataloader),log_vars) 
def predict(model, T,x, device):
    data.to(device)
    output=model(T,x)
    return output
def real_mixing(x):
    return 2/x[:, [0]]*x[:, [1]] * (1- x[:, [1]])+x[:, [1]] * np.log(x[:, [1]]) + (1-x[:, [1]]) * np.log(1-x[:, [1]])

def compute_derivative(model, T, x, device,x_mean,x_std,T_R_std):
    T, x = T.to(device), x.to(device)
    x.requires_grad_(True)
    T.requires_grad_(True)
    output = model(T, x)
    grad = torch.autograd.grad(outputs=output, inputs=x, grad_outputs=torch.ones_like(output), create_graph=True,retain_graph=True)[0]/ x_std
    gradT = torch.autograd.grad(outputs=output, inputs=T, grad_outputs=torch.ones_like(output), create_graph=True,retain_graph=True)[0] / T_R_std
    grad_d2 = torch.autograd.grad(outputs=grad, inputs=x, grad_outputs=torch.ones_like(output), create_graph=True,retain_graph=True)[0]/ x_std
    return grad,gradT, grad_d2
def ac_coefficient1(x,T,x_mean,x_std,T_mean,T_std):
    x_x=standarize.unstandardize_tensor(x,x_mean,x_std)
    x_T=standarize.unstandardize_tensor(T,T_mean,T_std)
    return 2/x_T*(1-2*x_x)#+torch.log(x_x/(1-x_x))
def ac_coefficient2(x,T,x_mean,x_std,T_mean,T_std):
    x_x=standarize.unstandardize_tensor(x,x_mean,x_std)
    x_T=standarize.unstandardize_tensor(T,T_mean,T_std)
    derivative2=-2*x_x*(1-x_x)/x_T**2
    return derivative2

def acoe1(x,x_mean,x_std,T_mean,T_std):
    x_x=standarize.unstandardize_tensor(x[:, [1]],x_mean,x_std)
    x_T=standarize.unstandardize_tensor(x[:, [0]],T_mean,T_std)
    return 2/ x_T* (1-x_x) **2
def acoe2(x,x_mean,x_std,T_mean,T_std):
    x_x=standarize.unstandardize_tensor(x[:, [1]],x_mean,x_std)
    x_T=standarize.unstandardize_tensor(x[:, [0]],T_mean,T_std)
    return 2/ x_T* x_x**2
def gex(x,pred,x_mean,x_std):
    x_x=standarize.unstandardize_tensor(x[:, [1]],x_mean,x_std)
    return pred.view(-1, 1) #- x_x * torch.log(x_x) - (1 - x_x) * torch.log(1 - x_x)
def acoe1_pred(x,gex,gex_d,x_mean,x_std):
    x_x=standarize.unstandardize_tensor(x[:, [1]],x_mean,x_std)
    return gex + (1-x_x) * gex_d.view(-1, 1)
def acoe2_pred(x,gex,gex_d,x_mean,x_std):
    x_x=standarize.unstandardize_tensor(x[:, [1]],x_mean,x_std)
    return gex - (1-x_x) * gex_d.view(-1, 1)
def acoe1_pred_dd(T, x,gmix,x_mean,x_std):
    outputs_ones=torch.ones(gmix.size(), device=device)
    gmix_d=torch.autograd.grad(outputs=gmix, inputs=x, grad_outputs=outputs_ones, create_graph=True)[0]
    gmix_dd=torch.autograd.grad(outputs=gmix_d, inputs=x, grad_outputs=outputs_ones, create_graph=True)[0]
    zz= gmix + (1-x) * ( gmix_d )
    # zz = gmix_d
    acoe1_d=torch.autograd.grad(outputs=zz, inputs=x, grad_outputs=torch.ones(zz.size(), device=device), create_graph=True)[0]
    return zz,acoe1_d.view(-1,1)
def acoe2_pred_dd(T, x,gmix,x_mean,x_std):
    outputs_ones=torch.ones(gmix.size(), device=device)
    gmix_d=torch.autograd.grad(outputs=gmix, inputs=x, grad_outputs=outputs_ones, create_graph=True)[0]
    gmix_dd=torch.autograd.grad(outputs=gmix_d, inputs=x, grad_outputs=outputs_ones, create_graph=True)[0]
    # zz = gmix_d
    zz= gmix   - (x) * ( gmix_d )
    acoe2_d=torch.autograd.grad(outputs=zz, inputs=x, grad_outputs=torch.ones(zz.size(), device=device), create_graph=True)[0]
    return zz,acoe2_d.view(-1,1)
def GD_equation(x,gmix):    
    outputs_ones = torch.ones(gmix.size(), device=device)
    gmix_d = torch.autograd.grad(outputs=gmix, inputs=x, grad_outputs=outputs_ones, create_graph=True)[0]
    gmix_dd = torch.autograd.grad(outputs=gmix_d, inputs=x, grad_outputs=outputs_ones, create_graph=True)[0]
    zz1=gmix  + (1-x) * (gmix_d)
    acoe1_d =  torch.autograd.grad(outputs=zz1, inputs=x, grad_outputs=torch.ones(zz1.size(), device=device), create_graph=True)[0]
    zz2=gmix  - (x) * (gmix_d)
    acoe2_d = torch.autograd.grad(outputs=zz2, inputs=x, grad_outputs=torch.ones(zz2.size(), device=device), create_graph=True)[0]
    GD_R = x * acoe1_d + (1-x) * acoe2_d
    return GD_R
def gex_r(x,x_mean,x_std,T_mean,T_std):
    x_x=standarize.unstandardize_tensor(x[:, [1]],x_mean,x_std)
    x_T=standarize.unstandardize_tensor(x[:, [0]],T_mean,T_std)
    return 2/x_T*x_x * (1- x_x)
def gex_rd(x,x_mean,x_std,T_mean,T_std):
    x_x=standarize.unstandardize_tensor(x[:, [1]],x_mean,x_std)
    x_T=standarize.unstandardize_tensor(x[:, [0]],T_mean,T_std)
    return 2/x_T * ((1- 2*x_x))
def gex_dm(x,gmix,x_mean,x_std):
    x_x=standarize.unstandardize_tensor(x[:, [1]],x_mean,x_std).view(-1)
    return (gmix)
def output_2_dataframe(all_data,T,x,model,device,sacle_data):
                T_R_std=sacle_data['T_R_std']
                x.requires_grad_(True)
                T.requires_grad_(True)
                all_data.requires_grad_(True)
                pred = predict(model, T,x, device)
                pred.requires_grad_(True)
                de_acoe1_p,de_acoe1_P_d=acoe1_pred_dd(T,x,pred,0,1)
                de_acoe1_r=acoe1(all_data,0,1,0,1)
                de_acoe2_p,de_acoe2_P_d=acoe2_pred_dd(T,x,pred,0,1)
                de_acoe2_r=acoe2(all_data,0,1,0,1)
                de_GD_resudual = GD_equation(x,pred)
                de_derivative,de_derivativeT,de_derivative_d2 = compute_derivative(model, T, x, device,0,1,T_R_std)
                de_Gmix_test_d_r=ac_coefficient1(x,T,0,1,0,1)
                de_Gex_p = gex(all_data,pred,0,1)
                de_Gex__r= gex_r(all_data,0,1,0,1)
                de_gmix_dd=torch.autograd.grad(outputs=de_derivative, inputs=x, grad_outputs=torch.ones_like(de_derivative), create_graph=True,)[0]/ x_std
                de_gex_d_r=gex_rd(all_data,0,1,0,1)
                de_gex_d_x=gex_dm(all_data,pred,0,1)
                de_Gmix_test_dT_r=ac_coefficient2(x,T,0,1,0,1)
                de_derivativeT=torch.autograd.grad(outputs=pred, inputs=T, grad_outputs=torch.ones_like(pred), create_graph=True,)[0]/T_R_std
                return pred, de_acoe1_p, de_acoe1_r, de_acoe2_p, de_acoe2_r, de_GD_resudual, de_derivative, de_Gmix_test_d_r, de_Gex_p, de_Gex__r, de_gmix_dd, de_gex_d_r, de_gex_d_x, de_acoe1_P_d, de_acoe2_P_d,de_Gmix_test_dT_r,de_derivativeT,de_derivative_d2
def equidistant_sampling_by_index(arr, step):
  return arr[::step]

learning_rates = [0.01,0.005, 0.001,0.0005, 0.0001]
# learning_rates = 0.001
ablation_masks_list = list(itertools.product([0, 1], repeat=3))
for ablation_mask in range(1):
    for l in range(1):
        ablation_mask = [1,1,1]
        print("Training with ablation_mask =", ablation_mask)
        path = f'C:/Users/Wonglab/Desktop/WONG/backup_result/PINN/KAN_gmix_RG_val_training/DNN_v{2}_Margules_lr={0.0001}_opwe_{ablation_mask}'
        if not os.path.isdir(path):
            os.mkdir(path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.set_printoptions(threshold=np.inf)
        dataname='Gex_data_RG_V8_1001_601T_difference3'
        data=pd.read_csv(f'C:/Users/Wonglab/Desktop/WONG/朱-專題研究/Project/tests/data/{dataname}.csv')


        filtered_data = data[(data['T'] != 0.8) & (data['T'] != 1.72)& (data['T'] != 0.58)].reset_index(drop=True)
        T_R_mean=np.mean(filtered_data['T'])
        T_R_std=np.std(filtered_data['T'])
        filtered_data['T_std'] = (filtered_data['T'] - T_R_mean) / T_R_std
        T_R=T_R_flat = filtered_data["T"].drop_duplicates().reset_index(drop=True)
        x = filtered_data['x'].drop_duplicates().reset_index(drop=True)

        T_R_grid, x_grid = np.meshgrid(T_R, x)
        T_R_flat_sort = np.sort(T_R_grid.flatten())[::-1]
        x_flat = x_grid.flatten()
        T_R=T_R_flat = filtered_data["T"].drop_duplicates().reset_index(drop=True)
        x = filtered_data['x'].drop_duplicates().reset_index(drop=True)

        test_data_RG = filtered_data[(data['T'] == 0.8) | (data['T'] == 1.72)].reset_index(drop=True)
        T_num = 101
        index_step = len(T_R_flat_sort) // (T_num-1)  # 例如，如果想近似取出 10 個點
        if index_step == 0:
            index_step = 1 # 避免除以零
        selected_T_R_levels = equidistant_sampling_by_index(T_R_flat_sort, index_step)

        T_R_mean=np.mean(selected_T_R_levels)
        T_R_std=np.std(selected_T_R_levels)
        filtered_data['T_std'] = (filtered_data['T'] - T_R_mean) / T_R_std

        T_R=T_R_flat = filtered_data["T_std"].drop_duplicates().reset_index(drop=True)
        x = filtered_data['x'].drop_duplicates().reset_index(drop=True)

        T_R_grid, x_grid = np.meshgrid(T_R, x)
        T_R_flat_sort = np.sort(T_R_grid.flatten())[::-1]
        x_flat = x_grid.flatten()
        

       
        test_T_R= test_data_RG["T_std"]
        x_test = test_data_RG["x"]

        val_data_RG = data[(data['T'] == 0.58)].reset_index(drop=True)
        val_T_R= val_data_RG["T"]
        x_val = val_data_RG["x"]

        
        T_R_grid, x_grid = np.meshgrid(T_R, x)
        T_R_flat = T_R_grid.flatten()
        x_flat = x_grid.flatten()

        
        x_mean=0
        x_std=1

        TR_flat = filtered_data['T_std']
        xR_flat = filtered_data['x']
        data = np.vstack((TR_flat, xR_flat)).T

        data=torch.tensor(data)

        TR_flat_test = test_data_RG["T_std"]
        xR_flat_test = test_data_RG["x"]

        R2_v=[]
        R2_t=[]
        R2=[]
        ii=[]
        jj=[]
        epoch_n=[]
        MSE=[]
        MSE_val=[]
        MSE_test=[]
        Ts=[]
        xs=[]
        time=[]
        scale_data={}

        # num=[11,21,41,51,101,126,201,251,501,1001]
        T_num=[101,201,301]
        num=[21,51,101,126]
        for i in range(1):
            for j in range(1):  
                
                n = 1001
                m = 601
                x_interval=np.linspace(0, 1, n)
                x_interval=np.around(x_interval,5)
                
                np.random.seed(1*24+1)
                index_step = len(T_R_flat) // (m-1)  
                if index_step == 0:
                    index_step = 1 # 避免除以零
                selected_T_R_levels = equidistant_sampling_by_index(T_R_flat, index_step)


                print(m,n)
                selected_data = []
                gmix_select = []
                onest_derivative_select = []
                onest_derivative_2nd_select = []
                onest_derivative_T_select = []
                all_gmix=filtered_data['Gmix'].tolist()
                all_onest_derivative=filtered_data['derivative_x'].tolist()
                all_onest_derivative_2nd=filtered_data['2nd_derivative_x'].tolist()
                all_onest_derivative_T=filtered_data['derivative_T'].tolist()
                test_derivative=torch.tensor(test_data_RG['derivative_x'].tolist()).to(device)
                test_derivative_T=torch.tensor(test_data_RG['derivative_T'].tolist()).to(device)
                test_derivative_2nd=torch.tensor(test_data_RG['2nd_derivative_x'].tolist()).to(device)
                x_indice=[]

                for x_R_level in x_interval:
                    selected_x_indices = np.where(xR_flat == x_R_level)[0]
                    for idx in selected_x_indices:
                        x_indice.append(idx)
                x_indice = np.asarray(x_indice)

                for T_R_level in selected_T_R_levels:
                    T_R_indices = np.where(TR_flat == T_R_level)[0]
                    selected_x_indices = np.intersect1d(x_indice,T_R_indices)
                    # selected_x_indices = np.random.choice(T_R_indices, n, replace=False)
                    selected_data.append(data[selected_x_indices])
                    gmix_select.extend([all_gmix[b] for b in selected_x_indices])
                    onest_derivative_select.extend([all_onest_derivative[b] for b in selected_x_indices])
                    onest_derivative_T_select.extend([all_onest_derivative_T[b] for b in selected_x_indices])
                    onest_derivative_2nd_select.extend([all_onest_derivative_2nd[b] for b in selected_x_indices])

                selected_data = np.vstack(selected_data)
                selected_data=torch.tensor(selected_data).to(device)
                gmix_select=torch.tensor(gmix_select).to(device)
                onest_derivative_select=torch.tensor(onest_derivative_select).to(device)
                onest_derivative_2nd_select=torch.tensor(onest_derivative_2nd_select).to(device)
                onest_derivative_T_select=torch.tensor(onest_derivative_T_select).to(device)
                print(selected_data.shape)
                scale_data['x_mean'],scale_data['x_std'],scale_data['T_R_mean'],scale_data['T_R_std']=x_mean,x_std,T_R_mean,T_R_std

                test_T_R= test_data_RG["T_std"]
                x_test = test_data_RG["x"]
                val_T_R= val_data_RG["T"]
                x_val = val_data_RG["x"]
                test_data=np.concatenate((np.array(test_T_R).reshape(-1,1),np.array(x_test).reshape(-1,1)),axis=1)
                test_data=torch.tensor(test_data).to(device)

                val_data=np.concatenate((np.array(val_T_R).reshape(-1,1),np.array(x_val).reshape(-1,1)),axis=1)
                val_data=torch.tensor(val_data).to(device)
                
                test_input=test_data
                test_label=torch.tensor(test_data_RG["Gmix"].values).to(device)
                train_label=gmix_select
                val_label=torch.tensor(val_data_RG["Gmix"].values).to(device)
                val_onest_derivative=torch.tensor(val_data_RG["derivative_x"].values).to(device)
                val_onest_derivative_2nd=torch.tensor(val_data_RG["2nd_derivative_x"].values).to(device)
                val_onest_derivative_T=torch.tensor(val_data_RG["derivative_T"].values).to(device)
                print(test_data.shape)
                x_select=selected_data[:,1]
                T_R_select=selected_data[:,0]

                g_mix_test=torch.tensor(np.array(test_label.cpu())).reshape(-1)
                g_mix=torch.tensor(np.array(train_label.cpu())).reshape(-1)

                T_R_test=test_input[:,0]
                x_test=standarize.standardize_tensor_test(test_input[:,1],x_mean,x_std)

                test_data_d=torch.cat((T_R_test.view(-1, 1), x_test.view(-1, 1)), dim=1)
                train_val_data_d=torch.cat((T_R_select.view(-1, 1), x_select.view(-1, 1)), dim=1)
                
                validation_split = 0  # Proportion of data for validation
               

                selected_data=selected_data.cpu()
                unique_temperatures = np.unique(selected_data[:, 0])
                train_indices = []
                val_indices = []
                for temp in unique_temperatures:
                    # 找出當前溫度下的所有數據點的索引
                    temp_indices = np.where(selected_data[:, 0] == temp)[0]
                    
                    # 使用 train_test_split 劃分當前溫度下的數據
                    temp_train_indices, temp_val_indices = train_test_split(
                        temp_indices, test_size=validation_split, random_state=42
                    )
                    
                    train_indices.extend(temp_train_indices)
                    val_indices.extend(temp_val_indices)
                # 將索引轉換為 NumPy 陣列並排序，確保順序一致
                train_indices = np.array(train_indices)
                val_indices = np.array(val_indices)
                train_indices.sort()
                val_indices.sort()
                selected_data=selected_data.to(device)

                T_R_val=torch.cat((T_R_select[val_indices].view(-1, 1), torch.tensor(val_T_R).to(device).view(-1, 1)), dim=0)
                x_val=torch.cat((x_select[val_indices].view(-1, 1), torch.tensor(x_val).to(device).view(-1, 1)), dim=0)
                g_mix_val=torch.cat((train_label[val_indices].view(-1, 1), val_label.view(-1, 1)), dim=0)
                onest_derivative_val=torch.cat((onest_derivative_select[val_indices].view(-1, 1), val_onest_derivative.view(-1, 1)), dim=0)
                onest_derivative_2nd_val=torch.cat((onest_derivative_2nd_select[val_indices].view(-1, 1), val_onest_derivative_2nd.view(-1, 1)), dim=0)
                onest_derivative_T_val=torch.cat((onest_derivative_T_select[val_indices].view(-1, 1), val_onest_derivative_T.view(-1, 1)), dim=0)
            
                dataset = TensorDataset(T_R_select[train_indices], x_select[train_indices], train_label[train_indices],onest_derivative_select[train_indices],onest_derivative_T_select[train_indices],onest_derivative_2nd_select[train_indices])
                dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
                val_dataset = TensorDataset(T_R_val, x_val, g_mix_val,onest_derivative_val,onest_derivative_T_val,onest_derivative_2nd_val)
                val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
                dataset_test = TensorDataset(T_R_test,x_test, test_label)
                dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

                model = DNN().to(device)
                criterion = nn.MSELoss()
                initial_guess = [ 0.0,2.3,11.5,13.8 ]
                log_vars = torch.nn.Parameter(torch.tensor(initial_guess, dtype=torch.float32, device=device), requires_grad=True)
                optimizer = optim.Adam(list(model.parameters()) + [log_vars], lr=0.0001)
                patience = 100  # Number of epochs to wait for improvement
                min_delta = 1e-10  # Minimum change in validation loss to be considered an improvement
                best_val_loss = float('inf')
                epochs_no_improve = 0
                loss=[]
                losstarget=[]
                lossd1=[]
                lossdT=[]
                lossd2=[]
                vallosstarget=[]
                vallossd1=[]
                vallossd2=[]
                vallossdT=[]
                valloss=[]
                num_epochs = 2000
                start_time = datetime.datetime.now()
                # '''
                for epoch in range(num_epochs):
                    
                    train_loss,train_loss_target,train_loss_d1,train_loss_dT, train_loss_d2, log_vars1 = train(model, dataloader, criterion, optimizer, device, scale_data,log_vars,ablation_mask)
                    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}')
                    epoch_n.append(epoch)
                    loss.append(train_loss)
                    losstarget.append(train_loss_target)
                    lossd1.append(train_loss_d1)
                    lossdT.append(train_loss_dT)
                    lossd2.append(train_loss_d2)
                    T_R_select[val_indices].requires_grad_(True)
                    val_P = predict(model, T_R_select[val_indices],x_select[val_indices], device)
                    val_P_d,val_P_dT,val_P_d2 = compute_derivative(model, T_R_select[val_indices], x_select[val_indices], device,x_mean,x_std,T_R_std)
                    
                    # val_P_dT = torch.autograd.grad(outputs=val_P, inputs=T_R_select[val_indices], grad_outputs=torch.ones_like(val_P), create_graph=True,)[0]
                    val_R_d = onest_derivative_select[val_indices].view(-1)
                    val_R_d2 = onest_derivative_2nd_select[val_indices].view(-1)
                    val_R_dT = onest_derivative_T_select[val_indices].view(-1)/T_R_std
                    val_loss = criterion(val_P,train_label[val_indices]) +  criterion(val_P_d,val_R_d)*10**-1 +  criterion(val_P_dT,val_R_dT)*10**-5 +  criterion(val_P_d2,val_R_d2)*10**-7 # Define an evaluate function similar to train

                    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.6f}')
                    valloss.append(val_loss.cpu().item())
                    vallosstarget.append(criterion(val_P,train_label[val_indices]).cpu().item())
                    vallossd1.append(criterion(val_P_d,val_R_d).cpu().item())
                    vallossdT.append(criterion(val_P_dT,val_R_dT).cpu().item())
                    vallossd2.append(criterion(val_P_d2,val_R_d2).cpu().item())

                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        torch.save(model, f'{path}/DNN_{dataname}_all.pkl')
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve == patience:
                            print('Early stopping!')
                            break#'''
                end_time = datetime.datetime.now()
                time_difference = end_time - start_time
                # Load the best model
                model = torch.load(f'{path}/DNN_{dataname}_all.pkl')
                
                x_select=torch.tensor(x_select).clone().detach().requires_grad_(True)
                x_train=x_select[train_indices]
                x_val=x_select[val_indices]
                T_val = T_R_select[val_indices]

                predictiontrain,acoe1_p_train, acoe1_r_train, acoe2_p_train, acoe2_r_train, GD_resudual_train, derivative_train, ac_train, Gex_train,Gex_train_r, gmix_dd_train,gex_d_r_train, gex_d_x_train, acoe1_P_d_train, acoe2_P_d_train,derivativeT_train, acT_train,derivative_d2_train=output_2_dataframe(train_val_data_d[train_indices],T_R_select[train_indices],x_train,model,device,scale_data)
                predictions, acoe1_p, acoe1_r, acoe2_p, acoe2_r, GD_resudual, derivative, ac, Gex_test,Gex_test_r, gmix_dd,gex_d_r, gex_d_x, acoe1_P_d, acoe2_P_d,derivativeT, acT, derivative_d2=output_2_dataframe(test_input,T_R_test,x_test,model,device,scale_data)
                prediction_val, acoe1_p_val, acoe1_r_val, acoe2_p_val, acoe2_r_val, GD_resudual_val, derivative_val, ac_val, Gex_val,Gex_val_r, gmix_dd_val,gex_d_r_val, gex_d_x_val, acoe1_P_d_val, acoe2_P_d_val,derivativeT_val, acT_val, derivative_d2_val=output_2_dataframe(train_val_data_d[val_indices] ,T_val,x_val,model,device,scale_data)
                
                
                R2.append((np.corrcoef(g_mix_test.cpu().detach().numpy(),predictions.cpu().detach().numpy())[0, 1])**2)
                R2_t.append((np.corrcoef(g_mix[train_indices],predictiontrain.cpu().detach().numpy())[0, 1])**2)
                R2_v.append((np.corrcoef(g_mix[val_indices],prediction_val.cpu().detach().numpy())[0, 1])**2)
                MSE.append(criterion(g_mix[train_indices].cpu(), predictiontrain.cpu()).cpu().detach().numpy())
                MSE_val.append(criterion(g_mix[val_indices].cpu(), prediction_val.cpu()).cpu().detach().numpy())
                MSE_test.append(criterion(g_mix_test.cpu(), predictions.cpu()).cpu().detach().numpy())
                ii.append(m)
                jj.append(n)
                time.append(time_difference)
                Ts.append(test_data[:,0])
                xs.append(test_data[:,1])

                df = pd.DataFrame({
                    'MSE':MSE,
                    'MSE_val':MSE_val,
                    'MSE_test':MSE_test,
                    'R2_t': R2_t,
                    'R2': R2,
                    'R2_v': R2_v,
                    'm': ii,
                    'n': jj,
                    'time_difference':time 
                })
                df.to_csv(f'{path}/DNN_R2_i_j1_{dataname}_{m}_{n}_test2T.csv', index=False)

                dfD = pd.DataFrame({
                    'T':test_data_d[:,0].flatten().cpu().detach().numpy()*T_R_std+T_R_mean,
                    'Gmix_test_r':g_mix_test.flatten().cpu().detach().numpy(),
                    'Gmix_test_p':predictions.flatten().cpu().detach().numpy(),
                    'Gmix_test_d_r':test_derivative.flatten().cpu().detach().numpy(),
                    'Gmix_test_d_p':derivative.flatten().cpu().detach().numpy(),
                    'Gex__test_d_r':gex_d_r.flatten().cpu().detach().numpy(),
                    'Gex__test_d_p':gex_d_x.flatten().cpu().detach().numpy(),
                    'Gex_test_r':Gex_test_r.flatten().cpu().detach().numpy(),
                    'Gex_test': Gex_test.flatten().cpu().detach().numpy(),
                    'acoe1_r': acoe1_r.flatten().cpu().detach().numpy(),
                    'acoe1_p': acoe1_p.flatten().cpu().detach().numpy(),
                    'acoe2_r': acoe2_r.flatten().cpu().detach().numpy(),
                    'acoe2_p': acoe2_p.flatten().cpu().detach().numpy(),
                    'GD_resudual': GD_resudual.flatten().cpu().detach().numpy(),
                    'x': standarize.unstandardize_tensor(x_test,x_mean,x_std).flatten().cpu().detach().numpy(),
                    'acoe1_p_d':acoe1_P_d.flatten().cpu().detach().numpy(),
                    'acoe2_p_d':acoe2_P_d.flatten().cpu().detach().numpy(),
                    'Gmix_test_dT_r':test_derivative_T.flatten().cpu().detach().numpy(),
                    'Gmix_test_dT_p':acT.flatten().cpu().detach().numpy(),
                    'Gmix_test_d2_p':derivative_d2.flatten().cpu().detach().numpy(),
                    'Gmix_test_d2_r':test_derivative_2nd.flatten().cpu().detach().numpy()
                })
                dfD.to_csv(f'{path}/DNN_D_{dataname}_{m}_{n}_test2T.csv', index=False)

                dfD_train = pd.DataFrame({
                    'T':train_val_data_d[train_indices][:,0].flatten().cpu().detach().numpy()*T_R_std+T_R_mean,
                    'Gmix_test_r':g_mix[train_indices].flatten().cpu().detach().numpy(),
                    'Gmix_test_p':predictiontrain.flatten().cpu().detach().numpy(),
                    'Gmix_test_d_r':onest_derivative_select[train_indices].flatten().cpu().detach().numpy(),
                    'Gmix_test_d_p':derivative_train.flatten().cpu().detach().numpy(),
                    'Gex__test_d_r':gex_d_r_train.flatten().cpu().detach().numpy(),
                    'Gex__test_d_p':gex_d_x_train.flatten().cpu().detach().numpy(),
                    'Gex_test_r':Gex_train_r.flatten().cpu().detach().numpy(),
                    'Gex_test': Gex_train.flatten().cpu().detach().numpy(),
                    'acoe1_r': acoe1_r_train.flatten().cpu().detach().numpy(),
                    'acoe1_p': acoe1_p_train.flatten().cpu().detach().numpy(),
                    'acoe2_r': acoe2_r_train.flatten().cpu().detach().numpy(),
                    'acoe2_p': acoe2_p_train.flatten().cpu().detach().numpy(),
                    'GD_resudual': GD_resudual_train.flatten().cpu().detach().numpy(),
                    'x': standarize.unstandardize_tensor(x_select[train_indices],x_mean,x_std).flatten().cpu().detach().numpy(),
                    'acoe1_p_d': acoe1_P_d_train.flatten().cpu().detach().numpy(),
                    'acoe2_p_d': acoe2_P_d_train.flatten().cpu().detach().numpy(),
                    'Gmix_test_dT_r':onest_derivative_T_select[train_indices].flatten().cpu().detach().numpy(),
                    'Gmix_test_dT_p':acT_train.flatten().cpu().detach().numpy(),
                    'Gmix_test_d2_p':derivative_d2_train.flatten().cpu().detach().numpy(),
                    'Gmix_test_d2_r':onest_derivative_2nd_select[train_indices].flatten().cpu().detach().numpy()
                })
                dfD_train.to_csv(f'{path}/DNN_D_{dataname}_{m}_{n}_train.csv', index=False)
                
                dfD_val = pd.DataFrame({
                    'T': train_val_data_d[val_indices][:, 0].flatten().cpu().detach().numpy()*T_R_std+T_R_mean,
                    'Gmix_val_r': g_mix[val_indices].flatten().cpu().detach().numpy(),  # Assuming g_mix is defined for all data
                    'Gmix_val_p': prediction_val.flatten().cpu().detach().numpy(),
                    'Gmix_val_d_r': onest_derivative_T_select[val_indices].flatten().cpu().detach().numpy(),
                    'Gmix_val_d_p': derivative_val.flatten().cpu().detach().numpy(),
                    'Gex__val_d_r': gex_d_r_val.flatten().cpu().detach().numpy(),
                    'Gex__val_d_p': gex_d_x_val.flatten().cpu().detach().numpy(),
                    'Gex_val_r': Gex_val_r.flatten().cpu().detach().numpy(),
                    'Gex_val': Gex_val.flatten().cpu().detach().numpy(),
                    'acoe1_r': acoe1_r_val.flatten().cpu().detach().numpy(),
                    'acoe1_p': acoe1_p_val.flatten().cpu().detach().numpy(),
                    'acoe2_r': acoe2_r_val.flatten().cpu().detach().numpy(),
                    'acoe2_p': acoe2_p_val.flatten().cpu().detach().numpy(),
                    'GD_resudual': GD_resudual_val.flatten().cpu().detach().numpy(),
                    'x': standarize.unstandardize_tensor(x_select[val_indices], x_mean, x_std).flatten().cpu().detach().numpy(),
                    'acoe1_p_d': acoe1_P_d_val.flatten().cpu().detach().numpy(),
                    'acoe2_p_d': acoe2_P_d_val.flatten().cpu().detach().numpy(),
                    'Gmix_val_dT_r': onest_derivative_T_select[val_indices].flatten().cpu().detach().numpy(),
                    'Gmix_val_dT_p': acT_val.flatten().cpu().detach().numpy(),
                    'Gmix_test_d2_p':derivative_d2_val.flatten().cpu().detach().numpy(),
                    'Gmix_test_d2_r': onest_derivative_2nd_select[val_indices].flatten().cpu().detach().numpy(),
                })
                dfD_val.to_csv(f'{path}/DNN_D_{dataname}_{m}_{n}_val.csv', index=False)
                with open(f'{path}/log_vars.txt', 'w') as f:
                    f.write(','.join(map(str, log_vars1.flatten().cpu().detach())))
                df_loss= pd.DataFrame({
                    'train loss':loss,
                    'train losstarget':losstarget,
                    'train lossd1':lossd1,
                    'train lossdT':lossdT,
                    'train lossd2':lossd2,
                    'val loss':valloss,
                    'val losstarget':vallosstarget,
                    'val lossd1':vallossd1,
                    'val lossdT':vallossdT,
                    'val lossd2':vallossd2,
                })
                df_loss.to_csv(f'{path}/DNN_loss_{dataname}_test2T.csv', index=False)
                plt.clf()
                plt.plot(loss, label='train loss')
                plt.plot(valloss, label='val loss')
                plt.yscale('log') 
                plt.savefig(f'{path}/DNN_D_{dataname}_loss_{m}_{n}.png')
            

