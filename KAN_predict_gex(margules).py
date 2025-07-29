import matplotlib.pyplot as plt
import torch.cuda
import numpy as np
from kan import KAN, utils
import sklearn
import pandas as pd
from toolboxandexpiredproject import standarize
from sklearn.model_selection import train_test_split
import datetime
import os
from tqdm import tqdm
from kan import *
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def real_mixing(x):
     return 2*x[:, [0]]*x[:, [1]] * (1- x[:, [1]])+x[:, [1]] * torch.log(x[:, [1]]) + (1-x[:, [1]]) * torch.log(1-x[:, [1]])
def real_mixing_mn(x,max,min):
    return 2*standarize.unmapping_T(x[:, [0]],max,min)*x[:, [1]] * (1- x[:, [1]])+x[:, [1]] * torch.log(x[:, [1]]) + (1-x[:, [1]]) * torch.log(1-x[:, [1]])
#-------------------------------------------------------------
def derivative1(x,max,min):
    '''
    計算free energy對x的一次微分(analytic function)
    Args:
        x: 包含溫度(x[:,0]) 和摩爾分率(x[:,1]) 的tensor
        max: x[:,0]的最大值 將T unmapping
        min: x[:,0]的最小值 將T unmapping
    '''
    ones=np.ones((len(x),1))
    derivative1=2*standarize.unmapping_T(x[:, [0]],max,min)*(1-2*(x[:, [1]]))#+np.log((x[:, [1]]+ones*10**-23)/(1-x[:, [1]]+ones*10**-23))
    # for i in range(len(x)):
    #     if x[i,1]==0 or x[i,1]==1:
    #         derivative1[i]=0
    return derivative1
def derivative2(x,max,min):
    '''
    計算free energy對x的一次微分(analytic function)
    Args:
        x: 包含溫度(x[:,0]) 和摩爾分率(x[:,1]) 的tensor
        max: x[:,0]的最大值 將T unmapping
        min: x[:,0]的最小值 將T unmapping
    '''
    derivative2=-2*x[:, [1]]*(1-x[:, [1]])*standarize.unmapping_T(x[:, [0]],max,min)**2
    return derivative2
# def derivative2(x,max,min):
#     derivative2 = 2*standarize.unmapping_T(x[:, [0]],max,min)*(x[:, [1]])+np.log(1-x[:, [1]])+ 1
    # return derivative2
def acoe1(x,max,min):
    acoe1=2*standarize.unmapping_T(x[:, [0]],max,min)*(1-x[:, [1]])**2
    return acoe1
def acoe2(x,max,min):
    acoe2=2*standarize.unmapping_T(x[:, [0]],max,min)*(x[:, [1]])**2
    return acoe2
def gex(x,pred):
    g_ex=pred #- (x[:, [1]]) * torch.log(x[:, [1]]) - (1 - x[:, [1]]) * torch.log(1 - x[:, [1]])
    g_ex=g_ex.requires_grad_(True)
    return g_ex
def acoe1_pred(x,gex,gex_d):
    return gex + (1-x[:, [1]]) * gex_d.view(-1, 1)
def acoe1_pred_dd(x,gex,gmix):
    '''
    Args:
        x: 包含溫度(x[:,0]) 和摩爾分率(x[:,1]) 的tensor
        gex: excess gibbs energy 隨便找一個值替代，已棄用
        gmix: gibbs energy 可能是mixing或是excess根據輸出值更改
    '''
    outputs_ones=torch.ones(gmix.size(), device=device)
    gmix_d=torch.autograd.grad(outputs=gmix, inputs=x, grad_outputs=outputs_ones,create_graph=True, retain_graph=True,only_inputs=True)[0][:,1]
    gmix_d2=torch.autograd.grad(outputs=gmix_d, inputs=x, grad_outputs=torch.ones(gmix_d.size(), device=device), retain_graph=True,only_inputs=True)[0][:,1]
    zz=gmix.view(-1, 1)  + (1-(x[:, [1]])) * (gmix_d.view(-1, 1))
    acoe1_d=torch.autograd.grad(outputs=zz.view(-1, 1), inputs=x, grad_outputs=torch.ones(zz.size(), device=device), retain_graph=True,only_inputs=True)[0][:,1]
    zz2=gmix.view(-1, 1)  - (x[:, [1]]) * (gmix_d.view(-1, 1) )
    acoe2_d=torch.autograd.grad(outputs=zz2.view(-1, 1), inputs=x, grad_outputs=torch.ones(zz.size(), device=device), retain_graph=True,only_inputs=True)[0][:,1]
    return zz,acoe1_d.view(-1,1),gmix_d.view(-1,1) ,gmix_d2.view(-1,1),zz2.view(-1,1),acoe2_d.view(-1,1)

def acoe2_pred_dd(x,gex,gmix):
    outputs_ones=torch.ones(gmix.size(), device=device)
    gmix_d=torch.autograd.grad(outputs=gmix, inputs=x, grad_outputs=outputs_ones, create_graph=True,only_inputs=True)[0][:,1]
    zz=gmix.view(-1, 1)  - (x[:, [1]]) * (gmix_d.view(-1, 1) )
    acoe2_d=torch.autograd.grad(outputs=zz.view(-1, 1), inputs=x, grad_outputs=torch.ones(zz.size(), device=device), retain_graph=True,only_inputs=True)[0][:,1]
    return zz,acoe2_d.view(-1,1)
def acoe2_pred(x,gex,gex_d):
    return gex - (1-(x[:, [1]])) * gex_d.view(-1, 1)
def GD_equation(x,gmix,acoe1_dd,acoe2_dd):
    x_x=x[:, [1]]
    # outputs_ones=torch.ones(gmix.size(), device=device)
    # gmix_d=torch.autograd.grad(outputs=gmix, inputs=x, grad_outputs=outputs_ones, create_graph=True)[0][:,1]
    # zz1=gmix.view(-1, 1) - (x[:, [1]]) * torch.log(x[:, [1]]) - (1 - x[:, [1]]) * torch.log(1 - x[:, [1]]) + (1-(x_x)) * (gmix_d.view(-1, 1) - torch.log((x_x)/(1-(x_x))))  
    # acoe1_d=torch.autograd.grad(outputs=zz1, inputs=x, grad_outputs=torch.ones(zz1.size(), device=device), create_graph=True)[0][:,1]
    # zz2=gmix.view(-1, 1) - (x[:, [1]]) * torch.log(x[:, [1]]) - (1 - x[:, [1]]) * torch.log(1 - x[:, [1]]) - (x_x) * (gmix_d.view(-1, 1) - torch.log((x_x)/(1-(x_x))))
    # acoe2_d=torch.autograd.grad(outputs=zz2, inputs=x, grad_outputs=torch.ones(zz2.size(), device=device), create_graph=True)[0][:,1]
    GD_R=(x_x)*acoe1_dd.view(-1, 1)+(1-x_x)*acoe2_dd.view(-1, 1)
    return GD_R
def gex_r(x,max,min):
    return 2*standarize.unmapping_T(x[:, [0]],max,min)*(x[:, [1]]) * (1- (x[:, [1]]))

def create_train_val_split(selected_data, g_mix, derivative, derivative_T,derivative_2nd , test_size=0.2, random_state=None,device=device):
    """
    Args:
        selected_data: NumPy 陣列，包含溫度 ([:,0]) 和摩爾分率 ([:,1])。
        g_mix: NumPy 陣列，包含自由能。
        test_size: 驗證集比例。
        random_state: 隨機種子，用於重現性。
        derivative: 對x的微分值(差分)
        derivative_T: 對y的微分值(差分)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if random_state is not None:
        np.random.seed(random_state)
    selected_data=selected_data.cpu()
    unique_temperatures = np.unique(selected_data[:, 0])
    train_indices = []
    val_indices = []

    for temp in unique_temperatures:
        # 找出當前溫度下的所有數據點的索引
        temp_indices = np.where(selected_data[:, 0] == temp)[0]
        
        # 使用 train_test_split 劃分當前溫度下的數據
        temp_train_indices, temp_val_indices = train_test_split(
            temp_indices, test_size=test_size, random_state=random_state
        )
        
        train_indices.extend(temp_train_indices)
        val_indices.extend(temp_val_indices)

    # 將索引轉換為 NumPy 陣列並排序，確保順序一致
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    train_indices.sort()
    val_indices.sort()
    
    # 使用索引提取數據
    train_input = selected_data[train_indices].to(device)
    val_input = selected_data[val_indices].to(device)
    train_label = g_mix[train_indices].to(device)
    val_label = g_mix[val_indices].to(device)
    train_derivative = derivative[train_indices].to(device).clone().detach().requires_grad_(True) #torch.tensor(derivative[train_indices].to(device))
    val_derivative = derivative[val_indices].to(device).clone().detach().requires_grad_(True) # torch.tensor(derivative[val_indices].to(device))
    train_derivative_T = derivative_T[train_indices].to(device).clone().detach().requires_grad_(True)# torch.tensor(derivative_T[train_indices].to(device))
    val_derivative_T = derivative_T[val_indices].to(device).clone().detach().requires_grad_(True) # torch.tensor(derivative_T[val_indices].to(device))
    train_derivative_2nd = derivative_2nd[train_indices].to(device).clone().detach().requires_grad_(True) #torch.tensor(derivative[train_indices].to(device))
    val_derivative_2nd = derivative_2nd[val_indices].to(device).clone().detach().requires_grad_(True) # torch.tensor(derivative[val_indices].to(device))
    return train_input, val_input, train_label, val_label, train_derivative, val_derivative, train_derivative_T, val_derivative_T,train_derivative_2nd,val_derivative_2nd
def equidistant_sampling_by_index(arr, step):
  return arr[::step]
time=[]
R2_v=[]
MSE_val=[]
MSE_test=[]
MSE=[]
R2T=[]
R2=[]
ii=[]
jj=[]
time=[]
num=[11,21,41,51,101,126,201,251,501,1001]

ablation_masks_list = list(itertools.product([0, 1], repeat=3))
for ablation_mask in range(1):
    ablation_mask= (0,0,0)
    torch.cuda.empty_cache()
    for g in range (1):
        print("Training with ablation_mask =", ablation_mask)
        mat=[8,0]
        path = f'C:/Users/Wonglab/Desktop/WONG/backup_result/PINN/KAN_gmix_RG_val_training/KAN_margules_{mat}_H{ablation_mask}_derivative_calculation_opwe_[0,1]spline_onlyTloss'
        dataname='Gex_data_RG_1overT_V8_1001_601T_difference3'
        data=pd.read_csv(f'C:/Users/Wonglab/Desktop/WONG/朱-專題研究/Project/tests/data/{dataname}.csv')
        data = data.sort_values(by=['T', 'x'],ascending=False).reset_index(drop=True)

        if not os.path.isdir(path):
            os.mkdir(path)
        filtered_data = data[(data['T'] != 1.25) & (data['T'] != 0.5813953) & (data['T'] != 1.7241379)].reset_index(drop=True)
        T_R=T_R_flat = filtered_data["T"].drop_duplicates().reset_index(drop=True)
        x = filtered_data['x'].drop_duplicates().reset_index(drop=True)

        T_R_grid, x_grid = np.meshgrid(T_R, x)
        T_R_flat_sort = np.sort(T_R_grid.flatten())[::-1]
        x_flat = x_grid.flatten()

        test_data_RG = data[(data['T'] == 1.25) | (data['T'] == 0.5813953) ].reset_index(drop=True)
        test_T_R= test_data_RG["T"].drop_duplicates().reset_index(drop=True)
        x_test = test_data_RG["x"].drop_duplicates().reset_index(drop=True)

        

        val_data_RG = data[(data['T'] == 1.7241379)].reset_index(drop=True)
        val_T_R= val_data_RG["T"].drop_duplicates().reset_index(drop=True)
        x_val = val_data_RG["x"].drop_duplicates().reset_index(drop=True)
        
        T_R_max = 2.
        T_R_min = 0.5

        T_R_flat= standarize.mapping_T(T_R_flat,T_R_max,T_R_min)
        test_T_R = standarize.mapping_T(test_T_R,T_R_max,T_R_min)
        val_T_R = standarize.mapping_T(val_T_R,T_R_max,T_R_min)

        T_R_mn =  standarize.mapping_T(T_R,T_R_max,T_R_min)

        TR_flat = filtered_data['T']
        TR_flat = standarize.mapping_T(TR_flat,T_R_max,T_R_min)
        xR_flat = filtered_data['x']

        data = np.vstack((TR_flat, xR_flat)).T
        T_num = [101,201,301]
        num=[21,51,101,126]#251,501,1001,201,
        step_loss_unprune=[]
        for i in T_num:
            for j in num:
                m = i
                n = j
                x_interval=np.linspace(0, 1, n)
                x_interval=np.around(x_interval,5)
                random_seed=14
                np.random.seed(random_seed)

                onest_derivative_select=[]
                onest_derivative_T_select=[]
                onest_derivative_2nd_select=[]
                gmix_select = []
                selected_data = []
                track_loss_train = []
                track_loss_d1_train = []
                track_loss_d2_train = []
                track_loss_val = []
                track_loss_d1_val = []
                track_loss_d2_val = []
                x_indice = []
                track_loss_d3_train = []
                track_loss_d3_val = []
                all_gmix=filtered_data['Gmix'].tolist()
                all_onest_derivative=filtered_data['derivative_x'].tolist()
                all_onest_derivative_T=filtered_data['derivative_T'].tolist()
                all_onest_derivative_2nd=filtered_data['2nd_derivative_x'].tolist()

                for x_R_level in x_interval:
                    selected_x = np.where(xR_flat == x_R_level)[0]
                    for idx in selected_x:
                        x_indice.append(idx)
                x_indice = np.asarray(x_indice)

                T_num = m
                index_step = len(T_R_flat) // (T_num-1)  
                if index_step == 0:
                    index_step = 1 
                selected_T_R_levels = equidistant_sampling_by_index(T_R_flat, index_step)
                

                for T_R_level in selected_T_R_levels:
                    T_R_level=round(T_R_level,6)
                    T_R_indices = np.where(TR_flat.round(6) == T_R_level)[0]
                    selected_x_indices = np.intersect1d(x_indice,T_R_indices)
                    selected_data.append(data[selected_x_indices])
                    gmix_select.extend([all_gmix[b] for b in selected_x_indices])
                    onest_derivative_select.extend([all_onest_derivative[b] for b in selected_x_indices])
                    onest_derivative_T_select.extend([all_onest_derivative_T[b] for b in selected_x_indices])
                    onest_derivative_2nd_select.extend([all_onest_derivative_2nd[b] for b in selected_x_indices])

                
                selected_data = np.vstack(selected_data)
                selected_data=torch.tensor(selected_data).to(device)
                print(selected_data.shape)

                T_R_test_grid, x_grid_test = np.meshgrid(test_T_R, x_test)
                T_R_test_grid_flat = T_R_test_grid.flatten()
                x_flat_test = x_grid_test.flatten()
                test_data = np.vstack((T_R_test_grid_flat, x_flat_test)).T
                test_data=torch.tensor(test_data).to(device)

                T_R_val_grid, x_grid_val = np.meshgrid(val_T_R, x_val)
                T_R_val_grid_flat = T_R_val_grid.flatten()
                x_flat_val = x_grid_val.flatten()
                val_data = np.vstack((T_R_val_grid_flat, x_flat_val)).T
                val_data=torch.tensor(val_data).to(device)

                sorted_indices_by_second_col_test = torch.argsort(test_data[:, 1], descending=True)
                sorted_data_by_second_col_test = test_data[sorted_indices_by_second_col_test]
                sorted_indices_by_first_col_test = torch.argsort(sorted_data_by_second_col_test[:, 0], descending=True)
                test_data = sorted_data_by_second_col_test[sorted_indices_by_first_col_test]

                g_mix = torch.tensor(gmix_select).clone().detach().requires_grad_(True).view(-1,1).to(device)
                g_mix_test = torch.tensor(test_data_RG['Gmix']).clone().detach().requires_grad_(True).view(-1,1).to(device)
                g_mix_val = torch.tensor(val_data_RG['Gmix']).view(-1,1).to(device)

                onest_derivative = torch.tensor(onest_derivative_select).clone().detach().requires_grad_(True).view(-1,1).to(device)
                onest_derivative_test = torch.tensor(test_data_RG['derivative_x']).clone().detach().requires_grad_(True).view(-1,1).to(device)
                onest_derivative_val = torch.tensor(val_data_RG['derivative_x']).view(-1,1).to(device)

                onest_derivative_T = torch.tensor(onest_derivative_T_select).clone().detach().requires_grad_(True).view(-1,1).to(device)
                onest_derivative_T_test = torch.tensor(test_data_RG['derivative_T']).clone().detach().requires_grad_(True).view(-1,1).to(device)
                onest_derivative_T_val = torch.tensor(val_data_RG['derivative_T']).view(-1,1).to(device)
                
                onest_derivative_2nd = torch.tensor(onest_derivative_2nd_select).clone().detach().requires_grad_(True).view(-1,1).to(device)
                onest_derivative_2nd_test = torch.tensor(test_data_RG['2nd_derivative_x']).clone().detach().requires_grad_(True).view(-1,1).to(device)
                onest_derivative_2nd_val = torch.tensor(val_data_RG['2nd_derivative_x']).view(-1,1).to(device)

                test_data=test_data.clone().detach().requires_grad_(True)

                train_input, val_input, train_label, val_label,train_derivative,val_derivative,train_derivative_T,val_derivative_T,train_derivative_2nd,val_derivative_2nd = create_train_val_split(selected_data, g_mix,onest_derivative,onest_derivative_T,onest_derivative_2nd, test_size=0.1, random_state=random_seed)

                val_one_derivative=val_derivative
                val_one_derivative_2nd=val_derivative_2nd
                val_one_derivative_T=val_derivative_T
                test_input = test_data
                test_label = g_mix_test
                dataset = {}
                dataset_test = {}
                dataset['train_input'] = train_input.clone().detach().requires_grad_(True).to(device).view(-1,2).to(torch.float)
                dataset['test_input'] = val_input.clone().detach().requires_grad_(True).to(device).view(-1,2).to(torch.float)
                dataset['train_label'] = train_label.clone().detach().requires_grad_(True).to(device).view(-1,1).to(torch.float)
                dataset['test_label'] = val_label.clone().detach().requires_grad_(True).to(device).view(-1,1).to(torch.float)
                dataset_test['test_input'] = test_input.to(torch.float)
                dataset_test['test_label'] = test_label.to(torch.float)
                
                
                start_time = datetime.datetime.now()
                # '''
                model = KAN(width=[2, mat, mat, 1], grid=25, k=4, seed=0, device=torch.device('cuda'),grid_range=[0,1])
                log = 1

                steps = 1500
                min_delta = 1e-10
                patience = 150
                epochs_no_improve = 0
                
                best_loss = np.inf
                initial_guess = [ 0.0,2.3,2.3,13.8 ]
                
                log_vars = torch.nn.Parameter(torch.tensor(initial_guess, dtype=torch.float32, device=device), requires_grad=True)
                optimizer = LBFGS( list(model.parameters()) + [log_vars], lr=0.01, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
                pbar = tqdm(range(steps), desc='description', ncols=100)
                for _ in pbar:
                    def closure():
                        global loss_d3, loss_T, loss_d1,loss_d2,loss
                        optimizer.zero_grad()
                        pred = model(dataset['train_input'])
                        true = dataset['train_label']
                        
                        loss_T = torch.mean((pred-true)**2)
                        
                        grad_pred = torch.ones(pred.size(), device=device)
                        pred_d1 = torch.autograd.grad(outputs=pred, inputs=dataset['train_input'], grad_outputs=grad_pred, create_graph=True)[0][:,1]
                        true_d1 = train_derivative
                        loss_d1 = torch.mean((pred_d1.view(-1,1)-true_d1.view(-1,1))**2)

                        true_d2 = train_derivative_T
                        pred_d2 = torch.autograd.grad(outputs=pred, inputs=dataset['train_input'], grad_outputs=grad_pred, create_graph=True)[0][:,0]*(-1/(1*T_R_max)*standarize.unmapping_T(dataset['train_input'][:,0],T_R_max,T_R_min)**2)
                        loss_d2 = torch.mean((pred_d2.view(-1,1)-true_d2.view(-1,1))**2)

                        pred_d3 = torch.autograd.grad(outputs=pred_d1, inputs=dataset['train_input'], grad_outputs=torch.ones(pred_d1.size(), device=device), create_graph=True)[0][:,1]
                        true_d3 = train_derivative_2nd
                        loss_d3 = torch.mean((pred_d3.view(-1,1)-true_d3.view(-1,1))**2)
 
                        loss = (
                            1  *    loss_T #+
                            # 1 * ablation_mask[0] * torch.exp(-log_vars[1]) * loss_d1 +
                            # 1 * ablation_mask[1] * torch.exp(-log_vars[2]) * loss_d2 +
                            # 1 * ablation_mask[2] * torch.exp(-log_vars[3]) * loss_d3
                            )
                        loss.backward()
                        return loss

                    
                    pred_val = model(dataset['test_input'])
                    
                    true_val = dataset['test_label']
                    loss_val_T = torch.mean((pred_val-true_val)**2)
                    grad_pred_val=torch.ones(pred_val.size(), device=device)

                    true_val_d1 = val_one_derivative
                    pred_val_d1 = torch.autograd.grad(outputs=pred_val, inputs=dataset['test_input'], grad_outputs=grad_pred_val, create_graph=True)[0][:,1]
                    loss_val_d1 = torch.mean((pred_val_d1.view(-1,1)-true_val_d1.view(-1,1))**2)

                    true_val_d2 = val_one_derivative_T
                    pred_val_d2 = torch.autograd.grad(outputs=pred_val, inputs=dataset['test_input'], grad_outputs=grad_pred_val, create_graph=True)[0][:,0]*(-1/(1*T_R_max)*standarize.unmapping_T(dataset['test_input'][:,0],T_R_max,T_R_min)**2)
                    loss_val_d2 = torch.mean((pred_val_d2.view(-1,1)-true_val_d2.view(-1,1))**2)

                    true_val_d3 = val_one_derivative_2nd
                    pred_val_d3 = torch.autograd.grad(outputs=pred_val_d1, inputs=dataset['test_input'], grad_outputs=torch.ones(pred_val_d1.size(), device=device), create_graph=True)[0][:,1]
                    loss_val_d3 = torch.mean((pred_val_d3.view(-1,1)-true_val_d3.view(-1,1))**2)
                    
                    
                    loss_val =  loss_val_T # +loss_val_d1 + loss_val_d2 + loss_val_d3 
                    

                    optimizer.step(closure)
                    
                    if _ % log == 0:
                        pbar.set_description(f"loss: {loss.item():.2e}, loss_val_d1: {loss_val_d1.item():.2e}")

                    if loss_val < best_loss - min_delta :
                        best_loss = loss_val
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            pbar.set_description("Early stopping triggered!")
                            break
                    
                    track_loss_train.append(loss_T.flatten().cpu().detach().numpy().item())
                    track_loss_d1_train.append(loss_d1.flatten().cpu().detach().numpy().item())
                    track_loss_d2_train.append(loss_d2.flatten().cpu().detach().numpy().item())
                    track_loss_d3_train.append(loss_d3.flatten().cpu().detach().numpy().item())
                    track_loss_val.append(loss_val_T.flatten().cpu().detach().numpy().item())
                    track_loss_d1_val.append(loss_val_d1.flatten().cpu().detach().numpy().item())
                    track_loss_d2_val.append(loss_val_d2.flatten().cpu().detach().numpy().item())
                    track_loss_d3_val.append(loss_val_d3.flatten().cpu().detach().numpy().item())
                    
                print(log_vars)
                path_model = f'{path}/model_{m}_{n}'
                model.saveckpt(path_model)#'''
                path_model = f'{path}/model_{m}_{n}'
                model = KAN.loadckpt(path_model)

                pred_se=model(dataset_test['test_input']).reshape(-1)
                R2.append((np.corrcoef(dataset_test['test_label'].cpu().detach().numpy().reshape(-1),pred_se.cpu().detach().numpy())[0, 1])**2)
                R2T.append((np.corrcoef(dataset['train_label'].cpu().detach().numpy().reshape(-1),model(dataset['train_input']).cpu().detach().numpy().reshape(-1))[0, 1])**2)
                R2_v.append((np.corrcoef(dataset['test_label'].cpu().detach().numpy().reshape(-1),model(dataset['test_input']).cpu().detach().numpy().reshape(-1))[0, 1])**2)
                ii.append(m)
                jj.append(n)
                MSE.append(sklearn.metrics.mean_squared_error(dataset['train_label'].cpu().detach().numpy(),model(dataset['train_input']).cpu().detach().numpy()))
                MSE_test.append(sklearn.metrics.mean_squared_error(dataset_test['test_label'].cpu().detach().numpy(),pred_se.cpu().detach().numpy()))
                MSE_val.append(sklearn.metrics.mean_squared_error(dataset['test_label'].cpu().detach().numpy(),model(dataset['test_input']).cpu().detach().numpy()))
                print(ii,jj)
                print(R2)
                
                dataset_test['test_input'].requires_grad_(True)
                output = model(dataset_test['test_input'])
                grad_outputs = torch.ones(output.size(), device=device)
                Gex_test = gex(test_input,output)
                Gex_test_r= gex_r(test_input,T_R_max,T_R_min)
                acoe1_p,acoe1_p_dd,gradients,gradients_2nd,acoe2_p,acoe2_p_dd=acoe1_pred_dd(dataset_test['test_input'],Gex_test,output)
                gradients2=torch.autograd.grad(outputs=output.view(-1, 1), inputs=dataset_test['test_input'], grad_outputs=torch.ones(output.size(), device=device), create_graph=True)[0][:,0]*(-1/(1*T_R_max)*standarize.unmapping_T(dataset_test['test_input'][:,0],T_R_max,T_R_min)**2)
                acoe1_r=acoe1(test_input,T_R_max,T_R_min)
                acoe2_r=acoe2(test_input,T_R_max,T_R_min)
                derivative_wrt_test_input_1 = gradients
                test_input = dataset_test['test_input'].requires_grad_(True).to(device)
                ac=onest_derivative_test.cpu().detach().numpy()
                gradientstoT_real=onest_derivative_T_test.cpu().detach().numpy()
                
                
                GD_residaul=GD_equation(test_input,output,acoe1_p_dd,acoe2_p_dd)

                dfD = pd.DataFrame({
                    'T':test_input[:,0].flatten().cpu().detach().numpy(),
                    'Gmix_test_r':dataset_test['test_label'].flatten().cpu().detach().numpy(),
                    'Gmix_test_p':pred_se.flatten().cpu().detach().numpy(),
                    'Gmix_test_d_r':ac.flatten(),
                    'Gmix_test_d_p':gradients.flatten().cpu().detach().numpy(),
                    'Gex_test_r':Gex_test_r.flatten().cpu().detach().numpy(),
                    'Gex_test': Gex_test.flatten().cpu().detach().numpy(),
                    'acoe1_r': acoe1_r.flatten().cpu().detach().numpy(),
                    'acoe1_p': acoe1_p.flatten().cpu().detach().numpy(),
                    'acoe2_r': acoe2_r.flatten().cpu().detach().numpy(),
                    'acoe2_p': acoe2_p.flatten().cpu().detach().numpy(),
                    'x':dataset_test['test_input'][:,1].flatten().cpu().detach().numpy(),
                    'GD_residaul': GD_residaul.flatten().cpu().detach().numpy(),
                    'acoe1_p_dd':acoe1_p_dd.flatten().cpu().detach().numpy(),
                    'acoe2_p_dd':acoe2_p_dd.flatten().cpu().detach().numpy(),
                    'gradientstoT_real':gradientstoT_real.flatten(),
                    'gradientstoT':gradients2.flatten().cpu().detach().numpy(),
                    'gradients_2nd':gradients_2nd.flatten().cpu().detach().numpy(),
                    'gradients_2nd_real':onest_derivative_2nd_test.flatten().cpu().detach().numpy()
                })
                dfD.to_csv(f'{path}/KAN_D_{dataname}_{m}_{n}_test2T.csv', index=False)
                dataset['train_input'].requires_grad_(True)
                output_train = model(dataset['train_input'])
                Gex_train = gex(train_input,output_train)
                Gex_train_r= gex_r(train_input,T_R_max,T_R_min)
                grad_outputs_train = torch.ones(output_train.size(), device=device)
                acoe1_p_train,acoe1_p_dd_train,gradients_train,gradients_2nd_train,acoe2_p_train,acoe2_p_dd_train=acoe1_pred_dd(dataset['train_input'],Gex_train,output_train)
                gradients2_train=torch.autograd.grad(outputs=output_train.view(-1, 1), inputs=dataset['train_input'], grad_outputs=torch.ones(output_train.size(), device=device), create_graph=True)[0][:,0]*(-1/(1*T_R_max)*standarize.unmapping_T(dataset['train_input'][:,0],T_R_max,T_R_min)**2)
                acoe1_r_train=acoe1(train_input,T_R_max,T_R_min)
                acoe2_r_train=acoe2(train_input,T_R_max,T_R_min)
                derivative_wrt_train_input_1 = gradients_train
                train_input = dataset['train_input'].requires_grad_(True).to(device)
                ac_train=train_derivative.cpu().detach().numpy()
                gradientstoT_real_train=train_derivative_T.cpu().detach().numpy()

                GD_residaul_train=GD_equation(train_input,output_train,acoe1_p_dd_train,acoe2_p_dd_train)
                
                dfD_train = pd.DataFrame({
                    'T':train_input[:,0].flatten().cpu().detach().numpy(),
                    'Gmix_test_r':dataset['train_label'].flatten().cpu().detach().numpy(),
                    'Gmix_test_p':output_train.flatten().cpu().detach().numpy(),
                    'Gmix_test_d_r':ac_train.flatten(),
                    'Gmix_test_d_p':gradients_train.flatten().cpu().detach().numpy(),
                    'Gex_test_r':Gex_train_r.flatten().cpu().detach().numpy(),
                    'Gex_test': Gex_train.flatten().cpu().detach().numpy(),
                    'acoe1_r': acoe1_r_train.flatten().cpu().detach().numpy(),
                    'acoe1_p': acoe1_p_train.flatten().cpu().detach().numpy(),
                    'acoe2_r': acoe2_r_train.flatten().cpu().detach().numpy(),
                    'acoe2_p': acoe2_p_train.flatten().cpu().detach().numpy(),
                    'x':dataset['train_input'][:,1].flatten().cpu().detach().numpy(),
                    # 'GD_residaul': GD_residaul_train.flatten().cpu().detach().numpy(),
                    'acoe1_p_dd':acoe1_p_dd_train.flatten().cpu().detach().numpy(),
                    'acoe2_p_dd':acoe2_p_dd_train.flatten().cpu().detach().numpy(),
                    'gradientstoT_real':gradientstoT_real_train.flatten(),
                    'gradientstoT':gradients2_train.flatten().cpu().detach().numpy(),
                    'gradients_2nd':gradients_2nd_train.flatten().cpu().detach().numpy(),
                    'gradients_2nd_real':train_derivative_2nd.flatten().cpu().detach().numpy()
                })
                
                dfD_train.to_csv(f'{path}/KAN_D_{dataname}_{m}_{n}_train2T.csv', index=False)
                pred_se_val=model(dataset['test_input']).reshape(-1)
                dataset['test_input'].requires_grad_(True)
                output_val = model(dataset['test_input'])
                val_input = dataset['test_input'].requires_grad_(True).to(device)
                Gex_val = gex(val_input,output_val)
                Gex_val_r= gex_r(val_input,T_R_max,T_R_min)
                grad_outputs_val = torch.ones(output_val.size(), device=device)
                acoe1_p_val,acoe1_p_dd_val,gradients_val,gradients_2nd_val,acoe2_p_val,acoe2_p_dd_val=acoe1_pred_dd(val_input,Gex_val,output_val)
                gradients2_val=torch.autograd.grad(outputs=output_val.view(-1, 1), inputs=dataset['test_input'], grad_outputs=torch.ones(output_val.size(), device=device), create_graph=True)[0][:,0]*(-1/(1*T_R_max)*standarize.unmapping_T(dataset['test_input'][:,0],T_R_max,T_R_min)**2)
                acoe1_r_val=acoe1(val_input,T_R_max,T_R_min)
                acoe2_r_val=acoe2(val_input,T_R_max,T_R_min)
                ac_val=val_one_derivative
                gradientstoT_real_val=val_one_derivative_T.cpu().detach().numpy()
                gex_d_val =val_one_derivative_T.cpu().detach().numpy()
                
                GD_residaul_val=GD_equation(val_input,pred_se_val,acoe1_p_dd_val,acoe2_p_dd_val)

                df = pd.DataFrame({
                    'MSE':MSE,
                    'MSE_val':MSE_val,
                    'MSE_test':MSE_test,
                    'R2T':R2T,
                    'R2': R2,
                    "R2_v": R2_v,
                    'ii': ii,
                    'jj': jj,
                    # 'time':time
                })
                df.to_csv(f'{path}/KAN_R2_i3_{dataname}_{m}_{n}.csv', index=False)

                
                

                dfD_val = pd.DataFrame({
                    'T':val_input[:,0].flatten().cpu().detach().numpy(),
                    'Gmix_test_r':dataset['test_label'].flatten().cpu().detach().numpy(),
                    'Gmix_test_p':pred_se_val.flatten().cpu().detach().numpy(),
                    'Gmix_test_d_r':ac_val.flatten().cpu().detach().numpy(),
                    'Gmix_test_d_p':gradients_val.flatten().cpu().detach().numpy(),
                    'Gex_test_r':Gex_val_r.flatten().cpu().detach().numpy(),
                    'Gex_test': Gex_val.flatten().cpu().detach().numpy(),
                    'acoe1_r': acoe1_r_val.flatten().cpu().detach().numpy(),
                    'acoe1_p': acoe1_p_val.flatten().cpu().detach().numpy(),
                    'acoe2_r': acoe2_r_val.flatten().cpu().detach().numpy(),
                    'acoe2_p': acoe2_p_val.flatten().cpu().detach().numpy(),
                    'x':dataset['test_input'][:,1].flatten().cpu().detach().numpy(),
                    'GD_residaul': GD_residaul_val.flatten().cpu().detach().numpy(),
                    'acoe1_p_dd':acoe1_p_dd_val.flatten().cpu().detach().numpy(),
                    'acoe2_p_dd':acoe2_p_dd_val.flatten().cpu().detach().numpy(),
                    'gradientstoT_real':gradientstoT_real_val.flatten(),
                    'gradientstoT':gradients2_val.cpu().flatten().detach().numpy(),
                    'gradients_2nd':gradients_2nd_val.cpu().detach().flatten().numpy(),
                    'gradients_2nd_real':val_one_derivative_2nd.cpu().detach().flatten().numpy()
                })
                dfD_val.to_csv(f'{path}/KAN_D_{dataname}_{m}_{n}_val2T.csv', index=False)
                
                df_loss_up= pd.DataFrame({
                    'train loss':track_loss_train,
                    'train d1 loss':track_loss_d1_train,
                    'train d2 loss':track_loss_d2_train,
                    'train d3 loss':track_loss_d3_train,
                })
                df_loss_up.to_csv(f'{path}/KAN_loss_{dataname}_{m}_{n}_test2T_up.csv', index=False)
                
                df_loss= pd.DataFrame({
                    'val loss':track_loss_val,
                    'val d1 loss':track_loss_d1_val,
                    'val d2 loss':track_loss_d2_val,
                    'val d3 loss':track_loss_d3_val,
                })
                df_loss.to_csv(f'{path}/KAN_loss_{dataname}_{m}_{n}_test2T.csv', index=False)
    #--------------------------------------------------------------------------------------------------------------
