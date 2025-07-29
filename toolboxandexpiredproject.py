import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
import tqdm
import time
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
import math
from torch.func import grad
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
class standarize:
    def standardize_tensor(input_tensor):
        input_tensor = input_tensor[~torch.isinf(input_tensor)]
        input_tensor = input_tensor[~torch.isnan(input_tensor)]
        mean = torch.mean(input_tensor)
        std = torch.std(input_tensor)
        standardized_tensor = (input_tensor - mean) / std
        return standardized_tensor,mean,std
    def standardize_tensor_test(standardized_tensor, mean, std):
        standardized_tensor = (standardized_tensor - mean ) / std
        return standardized_tensor
    def unstandardize_tensor(standardized_tensor, mean, std):
        unstandardized_tensor = (standardized_tensor * std) + mean
        return unstandardized_tensor
    def minmax_tensor(input_tensor):
        # input_tensor = input_tensor[~np.isinf(input_tensor)]
        # input_tensor = input_tensor[~np.isnan(input_tensor)]
        max = np.max(input_tensor)
        min = np.min(input_tensor)
        standardized_tensor = (input_tensor - min) / (max-min)
        return standardized_tensor,max,min
    def max_min_tensor_test(tensors, max, min):
        max_min_tensor = (tensors - min ) / (max-min)
        return max_min_tensor
    def unmax_min_tensor(max_min_tensors, max, min):
        unmaxmin_tensor = max_min_tensors *(max-min)+min
        return unmaxmin_tensor
    def mapping_T(T_tensor,max, min):
        mapping_T_tensor=(T_tensor)/(1*max)
        return mapping_T_tensor
    def unmapping_T(T_tensor,max, min):
        unmapping_T_tensor=(T_tensor)*(1*max)
        return unmapping_T_tensor
class peng_robinson(nn.Module):
    def __init__(self):
        super(peng_robinson, self).__init__()
        self.hidden_layer1 = nn.Linear(5,2000)
        self.hidden_layer2 = nn.Linear(2000,1000)
        self.hidden_layer3 = nn.Linear(1000,250)
        self.output_layer = nn.Linear(250,1)

    def forward(self, T,V,PC,TC,OMEGA):
        inputs = torch.cat((T.unsqueeze(1), V.unsqueeze(1),PC.unsqueeze(1),TC.unsqueeze(1), OMEGA.unsqueeze(1)),axis=1) # combined arrays 
        layer1_out = torch.relu(self.hidden_layer1(inputs))
        layer2_out = torch.relu(self.hidden_layer2(layer1_out))
        layer3_out = torch.relu(self.hidden_layer3(layer2_out))
        output = self.output_layer(layer3_out) 
        output = torch.reshape(output,(-1,))
        return output
class peng_robinson_one(nn.Module):
    def __init__(self):
        super(peng_robinson_one, self).__init__()
        self.hidden_layer1 = nn.Linear(3,1000)
        self.hidden_layer2 = nn.Linear(1000,500)
        self.hidden_layer3 = nn.Linear(500,350)
        self.output_layer = nn.Linear(350,1)
        self.hidden_layer2_3 = nn.Linear(4,350)
        self.output_layer2 = nn.Linear(350,1)
        self.hidden_layer3_1 = nn.Linear(1,100)
        self.hidden_layer3_2 = nn.Linear(100,50)
        self.hidden_layer3_3 = nn.Linear(50,25)
        self.output_layer3 = nn.Linear(25,1)


    def forward(self,T,V):
        inputs = T.unsqueeze(1) # combined arrays 
        layer3_1_out = torch.relu(self.hidden_layer3_1(inputs))
        layer3_2_out = torch.relu(self.hidden_layer3_2(layer3_1_out))
        layer3_3_out = torch.relu(self.hidden_layer3_3(layer3_2_out))
        output3 = self.output_layer3(layer3_3_out) 
        output3 = torch.reshape(output3,(-1,))
        inputs = torch.cat((T.unsqueeze(1), V.unsqueeze(1), output3.unsqueeze(1)),axis=1) # combined arrays 
        layer1_out = torch.relu(self.hidden_layer1(inputs))
        layer2_out = torch.relu(self.hidden_layer2(layer1_out))
        layer3_out = torch.relu(self.hidden_layer3(layer2_out))
        output = self.output_layer(layer3_out) 
        output = torch.reshape(output,(-1,))
        layer4_out = torch.cat((T.unsqueeze(1), V.unsqueeze(1), output3.unsqueeze(1), output.unsqueeze(1) ),axis=1)
        layer3_out_2 = torch.relu(self.hidden_layer2_3(layer4_out))
        output2 = self.output_layer2(layer3_out_2) 
        output2 = torch.reshape(output2,(-1,))

        return output,output2,output3

def pressure_cal (T,V,PC,TC,OMEGA,net_out,EOS):
    P_net = torch.reshape(net_out,(-1,))
    if EOS == "PR":
        a = 0.45723553 * ((8.314 * 1000) * (8.314 * 1000)) * TC * TC / PC
        b = 0.07779607 * 8.314 * 1000 * TC / PC
        kappa = 0.37464 + 1.54226 * OMEGA - 0.26993 * OMEGA * OMEGA
        Tr=T / TC
        alpha = (1 + kappa * (1 - Tr **(0.5)))** 2
        u = alpha * a / (V * V + 2 * b * V - b * b)
        v_minus_b= V - b 
        g = 8314 * T / v_minus_b
        P_cal = g-u
    if EOS == "VSTR":
        a = 0.45723553 * ((8.314 * 1000) ** 2) * TC * TC / PC
        c = 8.314 * TC / PC * (-0.0065 + 0.0198 * OMEGA)
        L = 0.1290 * OMEGA * OMEGA + 0.6039 * OMEGA + 0.0877
        M = 0.1760 * OMEGA * OMEGA - 0.2600 * OMEGA + 0.8884
        b = 0.07779607 * 8.314 * 1000 * TC / PC - c
        alpha = ((T/TC)**(2*(M-1))) * (2.718281828 ** (L* (1 - (T/TC)**(2 * M))))
        P_cal = (8314 * T / ( V - b ) - alpha * a / ((V + c) * (V + b + 2 * c) + (b + c) * (V - b)))
    P_cal=torch.reshape(P_cal,(-1,))
    f = P_net - P_cal
    return f
#-----------------------------------------------------------------------------------------------------
def val_training(net, loss_name, data, validation_split=0.1,EOS="PR",valout=False,):
    batch_size = 32  # size of each batch
    iterations =1000
    mse_cost_function = torch.nn.MSELoss()    
    loss_values = []
    f_loss_values=[]
    validation_loss_values = []  # to store validation loss
    partial_T_val_loss_value=[]
    partial_V_val_loss_value=[]
    partial_T_LOSS_value=[]
    partial_V_LOSS_value=[]
    losses_fu_value=[]
    losses_fu_val_value=[]
    loss_total=[]
    loss_total_value=[]
    losses_alpha_value=[]
    losses_alpha_val_value=[]
    min_loss = float("inf")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    keys = ['T', 'TC', 'PC', 'V', 'OMEGA', 'P', 'fu' , 'alpha']
    data_tensors = {}
    for key in keys:
        data_tensors[key] = torch.tensor(data[key], dtype=torch.float32).to(device)
    T, TC, PC, V, OMEGA, P, fu, a = data_tensors.values()
    T_S,T_mean,T_std = standarize.standardize_tensor(T)
    V_S,V_mean,V_std = standarize.standardize_tensor(V)
    PC_S,PC_mean,PC_std = standarize.standardize_tensor(PC)
    TC_S,TC_mean,TC_std = standarize.standardize_tensor(TC)
    OMEGA_S,OMEGA_mean,OMEGA_std = standarize.standardize_tensor(OMEGA)
    fu=torch.log(fu)
    fu_S, fu_mean, fu_std=standarize.standardize_tensor(fu)
    fu_S = (fu - fu_mean ) / fu_std
    alpha_S, alpha_mean, alpha_std = standarize.standardize_tensor(a)
    P = torch.reshape(P, (-1,))
    P,P_mean,P_std=standarize.standardize_tensor(P)
    num_samples = len(T)
    num_val_samples = int(validation_split * num_samples)
    indices = np.random.RandomState(seed=16).permutation(num_samples)
    train_indices, val_indices = indices[num_val_samples:], indices[:num_val_samples]
    early_stopper = EarlyStopper(patience=30, min_delta=0)
    for epoch in range(iterations):
        t0 = time.time()
        with tqdm.tqdm(range(0, len(train_indices), batch_size), unit="batch", mininterval=0) as bar:
            nan_train_count = 0
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # net.train()
                optimizer.zero_grad() # to make the gradients zero
                indices = train_indices[start:start+batch_size]
                T_batch = T_S[indices]
                V_batch = V_S[indices]
                PC_batch = PC_S[indices]
                TC_batch = TC_S[indices]
                OMEGA_batch = OMEGA_S[indices]
                P_batch = P[indices]
                fu_batch = fu_S[indices]
                alpha_batch= alpha_S[indices]
                T_batch=T_batch.requires_grad_(True)
                V_batch=V_batch.requires_grad_(True)
                net_out_P,net_out_fu,net_out_alpha = net(T_batch, V_batch)
                net_out_P=net_out_P.requires_grad_(True)
                partial_V_batch=torch.autograd.grad(
                outputs=net_out_P,
                inputs=V_batch,grad_outputs=torch.ones_like(V_batch,),retain_graph=True,allow_unused=True,create_graph=True)
                partial_T_batch=torch.autograd.grad(
                outputs=net_out_P,
                inputs=T_batch,grad_outputs=torch.ones_like(T_batch,),retain_graph=True,allow_unused=True,create_graph=True)
                all_zeros = np.zeros(len(T_batch),)
                all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
                if loss_name == 'PINN':
                    mse_P = mse_cost_function(net_out_P, P_batch) #有scale
                    nan_count = 0
                    for i in range(len(fu_batch)):
                        if torch.isnan(fu_batch[i]) or fu_batch[i] == 0 or torch.isinf(fu_batch[i]):
                            net_out_fu[i] =  fu_batch[i] = fu_S[train_indices[start+i]] = 0
                            nan_count = nan_count + 1
                            nan_train_count = nan_train_count + 1
                    mse_fu = mse_cost_function (2.7182**net_out_fu , 2.7182**fu_batch) * len(fu_batch) / (len(fu_batch) - nan_count)
                    mse_alpha = mse_cost_function (net_out_alpha, alpha_batch)
                    f_out = pressure_cal(T[indices], V[indices], PC[indices], TC[indices], OMEGA[indices], net_out_P * P_std + P_mean, EOS) # output of f(x,t)
                    f_out = (f_out ) / P_std
                    partial_T_LOSS = (mse_cost_function(partial_T_gen(T[indices], V[indices], TC[indices], PC[indices], OMEGA[indices])*T_std/P_std,torch.abs(partial_T_batch[0]) ))
                    partial_V_LOSS = (mse_cost_function(partial_V_gen(T[indices], V[indices], TC[indices], PC[indices], OMEGA[indices])*V_std/P_std,partial_V_batch[0] ))
                    mse_f = mse_cost_function(f_out, all_zeros)
                    weight_partialT = 0.9
                    weight_partialV = 0.001
                    loss = partial_T_LOSS * weight_partialT + partial_V_LOSS * weight_partialT + mse_P + mse_f + mse_fu 
                    mse_alpha.backward(retain_graph=True) 
                    loss.backward(retain_graph=True) 
                    optimizer.step() 
                    bar.update(1)
        # net.eval()
        T_S_train=T_S[train_indices].requires_grad_(True)
        V_S_train= V_S[train_indices].requires_grad_(True)
        y_pred_P,y_pred_fu,y_pred_alpha = net(T_S_train, V_S_train)
        losses_fu = mse_cost_function(y_pred_fu , fu_S[train_indices])
        f_losses = pressure_cal(T[train_indices], V[train_indices], PC[train_indices], TC[train_indices], OMEGA[train_indices], y_pred_P * P_std + P_mean, EOS)
        all_zero = np.zeros(len(T_S[train_indices]),)
        all_zero = Variable(torch.from_numpy(all_zero).float(), requires_grad=False).to(device)
        partial_V= torch.autograd.grad(
                outputs=y_pred_P,
                inputs=V_S_train,grad_outputs=torch.ones_like(V_S_train,),retain_graph=True,allow_unused=True,create_graph=True)
        partial_T= torch.autograd.grad(
                outputs=y_pred_P,
                inputs=T_S_train,grad_outputs=torch.ones_like(T_S_train,),retain_graph=True,allow_unused=True,create_graph=True)
        print(nan_train_count,"---------------")
        losses_fu = mse_cost_function(2.7182**y_pred_fu , 2.7182**fu_S[train_indices]) / len(y_pred_fu) * (len(y_pred_fu) - nan_train_count)
        losses_alpha = mse_cost_function (y_pred_alpha, alpha_S[train_indices])
        mse_f = mse_cost_function(f_losses/P_std, all_zero)
        losses = mse_cost_function(y_pred_P , P[train_indices])
        partial_T_LOSSes = weight_partialT *(mse_cost_function(partial_T_gen(T[train_indices], V[train_indices], TC[train_indices], PC[train_indices], OMEGA[train_indices])/P_std*T_std,partial_T[0]))
        partial_V_LOSSes = weight_partialV *(mse_cost_function(partial_V_gen(T[train_indices], V[train_indices], TC[train_indices], PC[train_indices], OMEGA[train_indices])/P_std*V_std,torch.abs(partial_V[0]) ))
        loss_total=losses_fu  +  partial_T_LOSSes + partial_V_LOSSes + losses + losses_alpha
        losses_alpha_value.append(losses_alpha.detach().cpu())
        losses_fu_value.append(losses_fu.detach().cpu())
        partial_T_LOSS_value.append(partial_T_LOSSes.detach().cpu())
        f_loss_values.append(mse_f.detach().cpu())
        partial_V_LOSS_value.append(partial_V_LOSSes.detach().cpu())
        loss_values.append(losses.detach().cpu())
        loss_total_value.append(loss_total.detach().cpu())
        print("partial_T_Loss:",format(partial_T_LOSSes.data.item(), ".4E"))
        print("partial_V_Loss:",format(partial_V_LOSSes.data.item(), ".4E"))
        print("P Loss:",  format(losses.data.item(), ".4E"))
        print("fu_Loss:",format(losses_fu.data.item(), ".4E"))
        print("alpha_Loss:",format(losses_alpha.data.item(), ".4E"))
        print("total_training_loss:",format(loss_total.data.item(), ".4E"))
        # Validation
        if validation_split > 0:
            val_tensors = {'T': T_S, 'V': V_S, 'PC': PC_S, 'TC': TC_S, 'OMEGA': OMEGA_S, 'P': P, 'fu' : fu_S, 'alpha': alpha_S}
            val_batches = {}
            for key, tensor in val_tensors.items():
                val_batches[key + '_batch'] = tensor[val_indices]
            val_T_batch, val_V_batch, val_PC_batch, val_TC_batch, val_OMEGA_batch, val_P_batch, val_fu_batch, val_alpha_batch = val_batches.values()
            val_T_batch=val_T_batch.requires_grad_(True)
            val_V_batch=val_V_batch.requires_grad_(True)
            val_net_out_P , val_net_out_fu, val_net_out_alpha = net(val_T_batch, val_V_batch)
            val_loss_P = mse_cost_function(val_net_out_P, val_P_batch)
            nan_val_count = 0
            for i in range(len(val_fu_batch)):
                if torch.isnan(val_fu_batch[i]) or torch.isinf(val_fu_batch[i]) or val_fu_batch[i]== 0:
                    val_net_out_fu[i] =  val_fu_batch[i]  = 0
                    nan_val_count = nan_val_count + 1
            val_loss_fu = mse_cost_function(2.7182**val_net_out_fu, 2.7182**val_fu_batch) / len(val_fu_batch) * (len(val_fu_batch) - nan_val_count)
            val_loss_alpha = mse_cost_function(val_alpha_batch, val_net_out_alpha)
            val_net_out_P=val_net_out_P.requires_grad_(True)
            partial_T_val=torch.autograd.grad(
                outputs=val_net_out_P,
                inputs=val_T_batch,grad_outputs=torch.ones_like(val_T_batch,),retain_graph=True,allow_unused=True,create_graph=True)
            partial_V_val=torch.autograd.grad(
                outputs=val_net_out_P,
                inputs=val_V_batch,grad_outputs=torch.ones_like(val_V_batch,),retain_graph=True,allow_unused=True,create_graph=True)
            partial_T_val_loss=mse_cost_function(torch.abs(partial_T_val[0]),partial_T_gen(T[val_indices], V[val_indices], TC[val_indices], PC[val_indices], OMEGA[val_indices])/P_std*T_std)
            partial_V_val_loss=mse_cost_function(partial_V_val[0],partial_V_gen(T[val_indices], V[val_indices], TC[val_indices], PC[val_indices], OMEGA[val_indices])/P_std*V_std)
            losses_alpha_val_value.append(val_loss_alpha.detach().cpu())
            losses_fu_val_value.append(val_loss_fu.detach().cpu())
            partial_T_val_loss_value.append(partial_T_val_loss.detach().cpu())
            partial_V_val_loss_value.append(partial_V_val_loss.detach().cpu())
            val_loss = val_loss_P + partial_V_val_loss *  weight_partialV + partial_T_val_loss * weight_partialT + val_loss_fu + val_loss_alpha
            validation_loss_values.append(val_loss.detach().cpu())
            print("Validation Loss:", format(val_loss.data.item(), ".4E"))
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(net.state_dict(), f"C:/Users/Wonglab/Desktop/WONG/backup_result/PINN/model_{loss_name}.pkl")
            # if early_stopper.early_stop(val_loss):             
            #         break
        t1 = time.time()
        time.sleep(.01)
        print(f"Times {(t1 - t0):.4f}")
    if validation_split > 0:
        plt.plot(np.array(validation_loss_values), 'b', label='Validation_Loss')
    plt.plot(np.array(loss_values), 'r', label='P_Loss')
    plt.plot(np.array(partial_T_LOSS_value), 'k', label='partial_T_val_loss_value')
    plt.plot(np.array(partial_V_LOSS_value), 'g', label='partial_V_val_loss_value')
    plt.plot(np.array(losses_fu_value), 'm', label='losses_fu_value')
    plt.plot(np.array(loss_total_value), 'y', label='total_training_loss')
    plt.plot(np.array(losses_alpha_value), 'c', label='losses_alpha_value')
    plt.legend()
    plt.savefig(fname=f"C:/Users/Wonglab/Desktop/WONG/backup_result/PINN/model_{loss_name}_EPOCH.jpg",dpi=1000)
    plt.show()
    dT_train=T_S[train_indices]
    dv_train=V_S[train_indices]
    dT_train=dT_train.requires_grad_(True)
    dv_train=dv_train.requires_grad_(True)
    net_out_P_train,net_out_fu_train,net_out_alpha_train=net(dT_train, dv_train,)
    net_out_P_train=net_out_P_train.requires_grad_(True)
    val_P_batch=(val_P_batch * P_std) + P_mean
    P[train_indices]=(P[train_indices] * P_std) + P_mean
    val_net_out_P=(val_net_out_P * P_std) + P_mean
    val_net_out_fu=(val_net_out_fu * fu_std) + fu_mean
    net_out_alpha_train = (net_out_alpha_train * alpha_std) + alpha_mean
    net_out_P_train=(net_out_P_train * P_std) + P_mean
    partial_T=torch.autograd.grad(
                outputs=net_out_P_train,inputs=dT_train,grad_outputs=torch.ones_like(T_S[train_indices]),retain_graph=True,allow_unused=True,create_graph=True)
    partial_V=torch.autograd.grad(
                outputs=net_out_P_train,inputs=dv_train,grad_outputs=torch.ones_like(V_S[train_indices]),retain_graph=True,allow_unused=True,create_graph=True)
    if valout==True:
        df_v = pd.DataFrame()
        df = pd.DataFrame()
        df_scaler= pd.DataFrame()
        tensor_data_val={'T': T[val_indices].detach().cpu(),
                    'TC': TC[val_indices].detach().cpu(),'OMEGA': OMEGA[val_indices].detach().cpu(),
                    'PC': PC[val_indices].detach().cpu(),'V': V[val_indices].detach().cpu(),'P': P[val_indices].detach().cpu(),
                    "P_PINN":val_net_out_P.detach().cpu(),
                    "partial_T_val_PINN":partial_T_val[0].detach().cpu(),"partial_V_val_PINN":partial_V_val[0].detach().cpu(),
                    "partial_T_val":partial_T_gen(T[val_indices], V[val_indices], TC[val_indices], PC[val_indices],OMEGA[val_indices]).detach().cpu(),
                    "partial_V_val":partial_V_gen(T[val_indices], V[val_indices], TC[val_indices], PC[val_indices],OMEGA[val_indices]).detach().cpu(),
                    'f':val_net_out_fu.detach().cpu()}
        tensor_data={'T': T[train_indices].detach().cpu(),
                    'TC': TC[train_indices].detach().cpu(),'OMEGA': OMEGA[train_indices].detach().cpu(),
                    'PC': PC[train_indices].detach().cpu(),'V': V[train_indices].detach().cpu(),'P': P[train_indices].detach().cpu(),
                    "P_PINN":net_out_P_train.detach().cpu(),"partial_T_PINN":partial_T[0].detach().cpu(),"partial_V_PINN":partial_V[0].detach().cpu(),
                    "partial_T_val":partial_T_gen(T[train_indices], V[train_indices], TC[train_indices], PC[train_indices],OMEGA[train_indices]).detach().cpu(),
                    "partial_V_val":partial_V_gen(T[train_indices], V[train_indices], TC[train_indices], PC[train_indices],OMEGA[train_indices]).detach().cpu(),
                    'net_out_fu_train':net_out_fu_train.detach().cpu()}
        scaler_data={'T_mean': T_mean.detach().cpu().item(),'T_std': T_std.detach().cpu().item(),
                     'V_mean': V_mean.detach().cpu().item(),'V_std': V_std.detach().cpu().item(),
                     'TC_mean': TC_mean.detach().cpu().item(),'TC_std': TC_std.detach().cpu().item(),
                     'PC_mean': PC_mean.detach().cpu().item(),'PC_std': PC_std.detach().cpu().item(),
                     'OMEGA_mean': OMEGA_mean.detach().cpu().item(),'OMEGA_std': OMEGA_std.detach().cpu().item(),
                     'P_mean': P_mean.detach().cpu().item(),'P_std': P_std.detach().cpu().item(),"fu_mean":fu_mean.detach().cpu().item(),
                     "fu_std":fu_std.detach().cpu().item(),"alpha_mean":alpha_mean.detach().cpu().item(),"alpha_std":alpha_std.detach().cpu().item()}
        df=pd.DataFrame(tensor_data)
        df_v = pd.DataFrame(tensor_data_val)
        df_scaler=pd.DataFrame(scaler_data,index=[0])
    return  df_v, df, df_scaler


def predict(model, data, data_scaler, EOS="PR"):
    keys = ['T', 'TC', 'PC', 'V', 'OMEGA']
    data_tensors = {}
    for key in keys:
        data_tensors[key] = torch.tensor(data[key], dtype=torch.float32).to(device)
    T, TC, PC, V, OMEGA = data_tensors.values()
    data_tensors = {}
    keys = ['T_mean', 'TC_mean', 'PC_mean', 'V_mean', 'OMEGA_mean','P_mean', "fu_mean", "alpha_mean"]
    for key in keys:
        data_tensors[key] = torch.tensor(data_scaler[key], dtype=torch.float32).to(device)
    T_mean, TC_mean, PC_mean, V_mean, OMEGA_mean, P_mean, fu_mean, alpha_mean = data_tensors.values()
    data_tensors = {}
    keys = ['T_std', 'TC_std', 'PC_std', 'V_std', 'OMEGA_std','P_std','fu_std',"alpha_std"]
    for key in keys:
        data_tensors[key] = torch.tensor(data_scaler[key], dtype=torch.float32).to(device)
    T_std, TC_std, PC_std, V_std, OMEGA_std, P_std, fu_std ,alpha_std = data_tensors.values()
    data_tensors = {'T': T, 'TC': TC, 'V': V, 'PC': PC, 'OMEGA': OMEGA}
    normalized_tensors = {}
    for key, tensor in data_tensors.items():
        mean = locals()[f'{key}_mean']
        std = locals()[f'{key}_std']
        normalized_tensors[key + '_S'] = (tensor - mean) / std
    T_S, TC_S, V_S, PC_S, OMEGA_S = normalized_tensors.values()
    T_S=T_S.requires_grad_(True)
    V_S=V_S.requires_grad_(True)
    P_pred,fu_pred,alpha_pred=model(T_S, V_S)
    P_pred=P_pred.requires_grad_(True)
    partial_T=torch.autograd.grad(
                outputs=P_pred,inputs=T_S,grad_outputs=torch.ones_like(T_S),retain_graph=True,allow_unused=True)
    partial_V=torch.autograd.grad(
                outputs=P_pred,inputs=V_S,grad_outputs=torch.ones_like(V_S),retain_graph=True,allow_unused=True)
    tensor_data={"partial_T_PINN":partial_T[0].detach().cpu(),"partial_V_PINN":partial_V[0].detach().cpu()}
    tensor_data["partial_T_PINN"]=torch.abs(tensor_data["partial_T_PINN"]*P_std.detach().cpu()/T_std.detach().cpu())
    tensor_data["partial_V_PINN"]=tensor_data["partial_V_PINN"]*P_std.detach().cpu()/V_std.detach().cpu()
    df=pd.DataFrame(tensor_data)
    model.eval()
    u=(P_pred * P_std) + P_mean
    fu=(fu_pred * fu_std) + fu_mean
    fu=2.7182**fu
    alpha=(alpha_pred * alpha_std) + alpha_mean
    f=pressure_cal(T,V,PC,TC,OMEGA,u,EOS)
    return u, f, df, fu, alpha

def model_PARA(model_name,model_path,data): #測試用function 用於load model
    model = model_name
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    total_params = 0
    for name, param in state_dict.items():
        params_count = param.numel()
        print(f"Layer: {name}, Parameters: {params_count}")
        total_params += params_count
    print(f"Total parameters: {total_params}")
    layer_vars = {}
    for name, param in model.named_parameters():
            param_var = torch.autograd.Variable(param, requires_grad=False)
            layer_vars[name] = param_var
    T_S=standarize.standardize_tensor(torch.tensor(data['T'],dtype=torch.float32).to(device))
    TC_S=standarize.standardize_tensor(torch.tensor(data['TC'],dtype=torch.float32).to(device))
    PC_S=standarize.standardize_tensor(torch.tensor(data['PC'],dtype=torch.float32).to(device))
    V_S=standarize.standardize_tensor(torch.tensor(data['V'],dtype=torch.float32).to(device))
    OMEGA_S=standarize.standardize_tensor(torch.tensor(data['OMEGA'],dtype=torch.float32).to(device))
    X=torch.cat((T_S.unsqueeze(1),V_S.unsqueeze(1),PC_S.unsqueeze(1),TC_S.unsqueeze(1), OMEGA_S.unsqueeze(1)),axis=1)
    output1=torch.relu(torch.nn.functional.linear(X, weight=layer_vars["hidden_layer1.weight"], bias=layer_vars["hidden_layer1.bias"]))
    output2=torch.relu(torch.nn.functional.linear(output1, weight=layer_vars["hidden_layer2.weight"], bias=layer_vars["hidden_layer2.bias"]))
    output3=torch.relu(torch.nn.functional.linear(output2, weight=layer_vars["hidden_layer3.weight"], bias=layer_vars["hidden_layer3.bias"]))
    output=torch.nn.functional.linear(output3,layer_vars["output_layer.weight"],bias=None)
    print(output)

class DATALOAD():
    def __init__(self, path:str):
        self.path=path
        dataset=pd.read_csv(self.path)
        T=torch.tensor(dataset['T'],dtype=torch.float32).to(device)
        TC=torch.tensor(dataset['TC'],dtype=torch.float32).to(device)
        OMEGA=torch.tensor(dataset['OMEGA'],dtype=torch.float32).to(device)
        PC=torch.tensor(dataset['PC'],dtype=torch.float32).to(device)
        V=torch.tensor(dataset['V'],dtype=torch.float32).to(device)
        T_S=standarize.standardize_tensor(T)
        V_S=standarize.standardize_tensor(V)
        PC_S=standarize.standardize_tensor(PC)
        TC_S=standarize.standardize_tensor(TC)
        OMEGA_S=standarize.standardize_tensor(OMEGA)
        self.X=torch.cat((T_S.unsqueeze(1),V_S.unsqueeze(1),PC_S.unsqueeze(1),TC_S.unsqueeze(1), OMEGA_S.unsqueeze(1)),axis=1)
        P=torch.tensor(dataset['P'],dtype=torch.float32).to(device)
        self.y=P
        self.size=P.shape[0]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def item(self):
        return self.X,self.y
    def data(path):
        path=path
        dataset=pd.read_csv(path)
        T=torch.tensor(dataset['T'],dtype=torch.float32).to(device)
        TC=torch.tensor(dataset['TC'],dtype=torch.float32).to(device)
        OMEGA=torch.tensor(dataset['OMEGA'],dtype=torch.float32).to(device)
        PC=torch.tensor(dataset['PC'],dtype=torch.float32).to(device)
        V=torch.tensor(dataset['V'],dtype=torch.float32).to(device)
        P=torch.tensor(dataset['P'],dtype=torch.float32).to(device)
        b=torch.tensor(dataset['b'],dtype=torch.float32).to(device)
        Pr=torch.tensor(dataset['Pr'],dtype=torch.float32).to(device)
        return dataset,T,TC,OMEGA,PC,V,P,b,Pr
#----------------------------------------------------------------------------------------------------------------------------
def generate_random_temperature(TB,TC):
    T=round(np.random.normal((TB+1.5*TC)/2,scale = (1.5*TC-TB)/2), 4)
    while T<TB or T>2*TC or T == TC:
        T=round(np.random.normal((TB+1.5*TC)/2,scale = (1.5*TC-TB)/2), 4)                                                            
    return T                                                                                                                                                                                           
def generate_random_volume(VC,TC,PC,log):
    if log==True:
        l=round(math.log(VC/2),5)
        h=round(math.log(4*8314*TC/PC),5)
        V=round(np.random.uniform(low=l, high=h),5)
    else:
        V=round(np.random.normal((VC/2+4*8314*TC/PC)/2,scale = (4*8314*TC/PC-VC/2)/2), 5)
        while V<VC/2 or V>4*8314*TC/PC or V==VC:
            V=round(np.random.normal((VC/2+4*8314*TC/PC)/2,scale = (4*8314*TC/PC-VC/2)/2), 5)
    return V
def pressure_gen(T,V,TC,PC,OMEGA):
    a = 0.45723553 * ((8.314 * 1000) * (8.314 * 1000)) * TC * TC / PC
    b = 0.07779607 * 8.314 * 1000 * TC / PC
    kappa = 0.37464 + 1.54226 * OMEGA - 0.26993 * OMEGA * OMEGA
    Tr=float(T / TC)
    alpha = float(1 + kappa * (1 - Tr **(0.5)))** 2
    u = alpha * a / (V * V + 2 * b * V - b * b)
    v_minus_b= V - b 
    g = 8314 * T / v_minus_b
    P_ran = g-u
    return P_ran,a,b,kappa,alpha,Tr,u,g,v_minus_b
def vanderwaal_pressure_gen(T,V,TC,PC):
    b = 8.314* 1000 * TC / 8  / PC
    a = 27 * (8.314 * 1000) ** 2 * TC * TC / 64 / PC
    P_ran = 8.314 * 1000 * T / (V - b) - a / V **2
    return P_ran,b
def VSTR_pressure_gen(T,V,TC,PC,OMEGA):
    a = 0.45723553 * ((8.314 * 1000) ** 2) * TC * TC / PC
    c = 8.314 * TC / PC * (-0.0065 + 0.0198 * OMEGA)
    L = 0.1290 * OMEGA * OMEGA + 0.6039 * OMEGA + 0.0877
    M = 0.1760 * OMEGA * OMEGA - 0.2600 * OMEGA + 0.8884
    b = 0.07779607 * 8.314 * 1000 * TC / PC - c
    alpha = (T/TC)**(2*(M-1)) * math.exp(L* (1 - (T/TC)**(2*M)))
    P_ran = (8314 * T / ( V - b ) - alpha * a / ((V + c) * (V + b + 2 * c) + (b + c) * (V - b)))
    return P_ran
def partial_T_gen(T,V,TC,PC,OMEGA):
    a = 0.45723553 * ((8.314 * 1000) ** 2) * TC * TC / PC
    b = 0.07779607 * 8.314 * 1000 * TC / PC
    kappa = 0.37464 + 1.54226 * OMEGA - 0.26993 * OMEGA * OMEGA
    Tr=T/TC
    partial_T=(8314 / (V - b)) - 2 * a * (1 + kappa * (1 - Tr**0.5)) * (-kappa / (2 * (T * TC)**0.5)) / (V * V + 2 * b * V * V - b * b)
    return partial_T
def partial_V_gen(T,V,TC,PC,OMEGA):
    a = 0.45723553 * ((8.314 * 1000) ** 2) * TC * TC / PC
    b = 0.07779607 * 8.314 * 1000 * TC / PC
    kappa = 0.37464 + 1.54226 * OMEGA - 0.26993 * OMEGA * OMEGA
    Tr=(T / TC)
    alpha = (1 + kappa * (1 - Tr **(0.5)))** 2
    partial_V=(-8314 * T / (V - b)**2) + 2 * a* alpha * (V + b) / (V*(V+b) + b*(V-b))**(2)
    return partial_V
def fugacity_gen(T,V,TC,PC,OMEGA):
    a = 0.45723553 * ((8.314 * 1000) * (8.314 * 1000)) * TC * TC / PC
    b = 0.07779607 * 8.314 * 1000 * TC / PC
    kappa = 0.37464 + 1.54226 * OMEGA - 0.26993 * OMEGA * OMEGA
    Tr=float(T / TC)
    alpha = float(1 + kappa * (1 - Tr **(0.5)))** 2
    u = alpha * a / (V * V + 2 * b * V - b * b)
    v_minus_b= V - b 
    g = 8314 * T / v_minus_b
    P_ran = g-u
    Z=(P_ran*V)/(8314*T)
    B=(P_ran*b)/(8314*T)
    A=(P_ran*a*alpha)/(8314*8314*T*T)
    # ln_f_devide_p=(Z-1)-math.log(Z-B)-a*alpha/2.828*math.log((V+2.414*b)/(V-0.414*b))/b/8314/T
    ln_f_devide_p=(Z-1)-math.log(Z-B)-A/(2.828*B)*math.log((Z+2.414*B)/(Z-0.414*B))
    f=2.718281828**(ln_f_devide_p)*P_ran
    return f

def duplicate_rows_PV_van(input_file, output_file, num_duplicates,EOS):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        header = next(reader)
        greater_than_TC_count = 2
        less_than_TC_count = 3
        equal_to_TC_count = 1
        total_T_count = greater_than_TC_count + less_than_TC_count + equal_to_TC_count
        duplicated_rows_list = []
        for row in reader:
            duplicated_rows_list.append([row] * total_T_count)
        duplicated_rows_list=np.array(duplicated_rows_list).reshape(-1,6)
        duplicated_rows_list=pd.DataFrame(duplicated_rows_list,columns=header)
        for j in range (int(len(duplicated_rows_list)/total_T_count)):
            for _ in range(greater_than_TC_count):
                T = np.random.uniform(float(duplicated_rows_list.loc[total_T_count*j+_,"TC"]),2*float(duplicated_rows_list.loc[total_T_count*j+_,"TC"]))
                duplicated_rows_list.loc[total_T_count*j+_,"T"]=T
            for _ in range(less_than_TC_count):
                T = np.random.uniform(float(duplicated_rows_list.loc[total_T_count*j+_+greater_than_TC_count,"TB"]),float(duplicated_rows_list.loc[total_T_count*j+_+greater_than_TC_count,"TC"])) 
                duplicated_rows_list.loc[total_T_count*j+_+greater_than_TC_count,"T"]=T
            for _ in range(equal_to_TC_count):
                T=(float(duplicated_rows_list.loc[total_T_count*j+_+less_than_TC_count+greater_than_TC_count,"TC"]))
                duplicated_rows_list.loc[total_T_count*j+_+less_than_TC_count+greater_than_TC_count,"T"]=T
        header = duplicated_rows_list.columns
        duplicated_rows_list_2 = []
        for p in range (int(len(duplicated_rows_list)/total_T_count)):
            for j in range (total_T_count):
                for i in range (num_duplicates):
                    duplicated_rows_list_2.append(duplicated_rows_list.iloc[int(total_T_count*p+j),])
        duplicated_rows_list_2=np.array(duplicated_rows_list_2).reshape(-1,7)
        duplicated_rows_list_2=pd.DataFrame(duplicated_rows_list_2,columns=header)
        # duplicated_rows_list.set_index(header,inplace=True)
        for i in range(len(duplicated_rows_list_2)):     
            V=2.718281828459045**generate_random_volume(float(duplicated_rows_list_2.loc[i,"VC"]),float(duplicated_rows_list_2.loc[i,"TC"]),
                                    float(duplicated_rows_list_2.loc[i,"PC"]),log=True)
            TC=float(duplicated_rows_list_2.loc[i,"TC"])
            PC=float(duplicated_rows_list_2.loc[i,"PC"])
            OMEGA=float(duplicated_rows_list_2.loc[i,"OMEGA"])
            T=float(duplicated_rows_list_2.loc[i,"T"])
            if EOS == "PR":
                P_ran,a,b,kappa,alpha,Tr,u,g,v_minus_b = pressure_gen(T=T,V=V,TC=TC,PC=PC,OMEGA=OMEGA)
                if (P_ran*V-P_ran*b)<0 :
                    f="nan"
                else:
                    f=fugacity_gen(T,V,TC,PC,OMEGA)
                partial_T_ran = partial_T_gen(T,V,TC,PC,OMEGA)
                partial_V_ran = partial_V_gen(T,V,TC,PC,OMEGA)
                duplicated_rows_list_2.loc[i,"V"]=V
                duplicated_rows_list_2.loc[i,"P"]= P_ran
                duplicated_rows_list_2.loc[i,"a"]= a
                duplicated_rows_list_2.loc[i,"b"]= b
                duplicated_rows_list_2.loc[i,"kappa"]= kappa
                duplicated_rows_list_2.loc[i,"alpha"]= alpha
                duplicated_rows_list_2.loc[i,"Tr"]= Tr
                duplicated_rows_list_2.loc[i,"u"]= u
                duplicated_rows_list_2.loc[i,"g"]= g
                duplicated_rows_list_2.loc[i,"v_minus_b"]=v_minus_b
                duplicated_rows_list_2.loc[i,"partial_T"]= partial_T_ran
                duplicated_rows_list_2.loc[i,"partial_V"]= partial_V_ran 
                duplicated_rows_list_2.loc[i,"fu"]= f                   
            if EOS == "VSTR":
                while (0.01290 * OMEGA * OMEGA + 0.6039 * OMEGA + 0.0877)* (1 - (T/TC)**(2*(0.1760 * OMEGA * OMEGA - 0.2600 * OMEGA + 0.8884)))>2:
                    T=generate_random_temperature(float(duplicated_rows_list_2.loc[i,"TB"]),float(duplicated_rows_list_2.loc[i,"TC"]))
                    V=generate_random_volume(float(duplicated_rows_list_2.loc[i,"VC"]),float(duplicated_rows_list_2.loc[i,"TC"]),
                                           float(duplicated_rows_list_2.loc[i,"PC"]),log=True)
                P_ran = VSTR_pressure_gen(T,V,TC,PC,OMEGA)
                duplicated_rows_list_2.loc[i,"V"]=V
                duplicated_rows_list_2.loc[i,"P"]= P_ran
            if EOS == "VANDER":
                P_ran,b= vanderwaal_pressure_gen(T=T,V=V,TC=TC,PC=PC)
                duplicated_rows_list_2.loc[i,"V"]=V
                duplicated_rows_list_2.loc[i,"P"]= P_ran
                duplicated_rows_list_2.loc[i,"P/PC"]= P_ran/PC
                duplicated_rows_list_2.loc[i,"b"]= b
                                                                                                                                 
        duplicated_rows_list_2.to_csv(outfile, index=False)
def train_and_split(file_name):
    base_name, extension = os.path.splitext(file_name)
    if 'csv' in extension.lower():
        data = pd.read_csv(file_name)
        train_df, test_df = train_test_split(data, test_size=0.1, random_state=10)
        train_df.to_csv(f'{base_name}_train.csv', index=None)
        test_df.to_csv(f'{base_name}_test.csv', index=None)
    else:
        print("The file does not have a CSV extension.") 

def T_V_sequence_P(input_file, output_file, num_duplicates,EOS):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:      
        reader = csv.reader(infile)
        header = next(reader)         
        greater_than_TC_count = 50
        less_than_TC_count = 50
        equal_to_TC_count = 0
        total_T_count = greater_than_TC_count + less_than_TC_count + equal_to_TC_count
        duplicated_rows_list = []
        for row in reader:
            duplicated_rows_list.append([row] * total_T_count)
        duplicated_rows_list=np.array(duplicated_rows_list).reshape(-1,6)
        duplicated_rows_list=pd.DataFrame(duplicated_rows_list,columns=header)
        for j in range (int(len(duplicated_rows_list)/total_T_count)):
            for _ in range(greater_than_TC_count):
                local_TC= float(duplicated_rows_list.loc[total_T_count*j+_,"TC"])
                TC_interval =  local_TC / greater_than_TC_count
                T = local_TC + TC_interval * (_+1)
                duplicated_rows_list.loc[total_T_count*j+_,"T"]=T
            for _ in range(less_than_TC_count):
                local_TC= float(duplicated_rows_list.loc[total_T_count*j+_,"TC"])
                local_TB = float(duplicated_rows_list.loc[total_T_count*j+_+greater_than_TC_count,"TB"])
                TC_TB_interval = ( local_TC - local_TB ) / less_than_TC_count
                T = local_TB + TC_TB_interval * (_+1)
                duplicated_rows_list.loc[total_T_count*j+_+greater_than_TC_count,"T"]=T
            for _ in range(equal_to_TC_count):
                local_TC= float(duplicated_rows_list.loc[total_T_count*j+_,"TC"])
                T=(float(duplicated_rows_list.loc[total_T_count*j+_+less_than_TC_count+greater_than_TC_count,"TC"]))
                duplicated_rows_list.loc[total_T_count*j+_+less_than_TC_count+greater_than_TC_count,"T"]=T   
        header = duplicated_rows_list.columns
        duplicated_rows_list_2 = []
        for p in range (int(len(duplicated_rows_list)/total_T_count)):
            for j in range (total_T_count):
                for i in range (num_duplicates):
                    duplicated_rows_list_2.append(duplicated_rows_list.iloc[int(total_T_count*p+j),])
        duplicated_rows_list_2=np.array(duplicated_rows_list_2).reshape(-1,7)
        duplicated_rows_list_2=pd.DataFrame(duplicated_rows_list_2,columns=header)    
        g = 0
        for i in range(len(duplicated_rows_list_2)):
            TC=float(duplicated_rows_list_2.loc[i,"TC"])
            PC=float(duplicated_rows_list_2.loc[i,"PC"])
            OMEGA=float(duplicated_rows_list_2.loc[i,"OMEGA"])
            T=float(duplicated_rows_list_2.loc[i,"T"])
            VC=float(duplicated_rows_list_2.loc[i,"VC"])
            local_V_upper = 4*8314*TC/PC
            local_V_lower = VC/2
            V_interval = ( local_V_upper - local_V_lower ) / len(duplicated_rows_list_2) * total_T_count
            if g == num_duplicates :
                g = 0
            V = VC/2 + V_interval * g
            g = g + 1
            if EOS == "VANDER":
                P_ran,b= vanderwaal_pressure_gen(T=T,V=V,TC=TC,PC=PC)
                duplicated_rows_list_2.loc[i,"V"]=V
                duplicated_rows_list_2.loc[i,"P"]= P_ran
                duplicated_rows_list_2.loc[i,"Pr"]= P_ran/PC
                duplicated_rows_list_2.loc[i,"b"]= b

        duplicated_rows_list_2.to_csv(outfile, index=False)
