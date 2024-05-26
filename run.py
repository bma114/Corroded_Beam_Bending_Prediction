from utils import *

import pandas as pd
import numpy as np

import os
import time
import joblib as jb

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor



"""
                                    NEW PREDICTION - MECHANICAL RESPONSE OF CORRODED RC BEAM


Enter the input features of the new beam below, to generate predictions for the mechanical performance.

Please follow the guidelines below for formatting the input parameters to ensure optimal model performance. 
If new categorical inputs are present which are not included below (e.g. I-Beam cross-section), 
format the input as an abbreviated acronym following the convention adopted for the respective variable (e.g. I). 

See the acompanying Database Key document for more details on data formatting.


Input Guide:
1. Cross section = S (square) or R (rectangular).
2. Reinforcement design = T_C (Tension-compression layers), ST (single tension layer), 
                            UT_C (U-shaped tenison-compression layers), or T_M_C (tension-mid-compression layers).
3. Longitudinal type = D (deformed), P (plain), or N/A (no stirrups present).
4. End anchorage = S (straight), H_90 (hooked with 90 deg bend), or H_180 (hooked with a 180 degree bend). 
                    If an additional system is present, use an abreviated acronym as above.
5. Stirrup type = D (deformed) or P (plain).
6. Corrosion method = N (natural), IC (impressed-current), EI (environmentally-induced), C (control - uncorroded).
7. Width = Cross-sectional width, in mm.
8. Depth = Cross-sectional depth, in mm.
9. Length = Full member length, in mm.
10. Bottom cover = Concrete cover depth to the centre of the bottom tension reinforcement layer, in mm.
11. Tension ratio = Reinforcement ratio for the tension reinforcement, as a percentage.
12. fy = Yield stress of the tension reinforcement, in MPa.
13. fsu = Ultimate tensile strength of tension reinforcement, in MPa.
14. Volumetric ratio = Volumetric ratio of transverse reinforcement, as a percentage.
15. f_c = Characteristic compressive strength or real compressive strength at the current age, in MPa.
16. i_corr = Corrosion current density, in μA/cm2.
17. Duration = Corrosion incubation duration (artificial) or age of member (natural), in days.
18. Mass Loss = Average mass loss of tension reinforcement, as a percentage. 
                    If not experimentally measured, average mass loss can be estimated through Faraday's Law.
19. Shear span = Length of critical shear span during loading.


"""
# Enter data for new beam here
cross_section = ""                              # Enter a string
reinforcement_design = ""                       # Enter a string
longitudinal_type = ""                          # Enter a string
end_anchorage = ""                              # Enter a string
stirrup_type = ""                               # Enter a string
corrosion_method = ""                           # Enter a string
width =                                         # Enter an integer
depth =                                         # Enter an integer
length =                                        # Enter an integer
bottom_cover =                                  # Enter a floating point number or integer
tension_ratio =                                 # Enter a floating point
fy =                                            # Enter a floating point number or integer
fsu =                                           # Enter a floating point number or integer
volumetric_ratio =                              # Enter a floating point
f_c =                                           # Enter a floating point number or integer
i_corr =                                        # Enter a floating point number or integer
duration =                                      # Enter a floating point number or integer
mass_loss =                                     # Enter a floating point number or integer
shear_span =                                    # Enter a floating point number or integer


beam_data = [
    cross_section, reinforcement_design, longitudinal_type, end_anchorage,
    stirrup_type, corrosion_method, width, depth, length, bottom_cover,
    tension_ratio, fy, fsu, volumetric_ratio, f_c, i_corr, duration,
    mass_loss, shear_span]



### ========================= ###  LOAD DATASETS AND PREPROCESS MODEL TRAIN/TEST DATA  ### ========================= ###


# Load individual datasets
df_m, df_r, df_p, df_dy, df_du = load_data()

# Define column types for each dataset
cat_cols_m, num_cols_m, resp_col_m = col_types(df_m)
cat_cols_r, num_cols_r, resp_col_r = col_types(df_r)
cat_cols_p, num_cols_p, resp_col_p = col_types(df_p)
cat_cols_dy, num_cols_dy, resp_col_dy = col_types(df_dy)
cat_cols_du, num_cols_du, resp_col_du = col_types(df_du)


# Pre-process datasets
# M_max dataset
cat_cols_m_enc = encoder(cat_cols_m, 'encoder_m.pkl').astype(object)                      # Encode categorical variables
X_m = pd.concat([num_cols_m, cat_cols_m_enc], axis='columns')                             # Combine features
X_sm = feature_scaling(X_m, 'scaler_m.pkl')                                               # Normalize all features                         
Y_m = resp_col_m.to_numpy()                                                               # Convert response variable

# R_exp dataset
cat_cols_r_enc = encoder(cat_cols_r, 'encoder_r.pkl').astype(object)                      # Encode categorical variables
X_r = pd.concat([num_cols_r, cat_cols_r_enc], axis=1)                                     # Combine features
X_sr = feature_scaling(X_r, 'scaler_r.pkl')                                               # Normalize all features 
Y_r = resp_col_r.to_numpy()                                                               # Convert response variable
scaler_ry = MinMaxScaler()
Y_norm_r = scaler_ry.fit_transform(Y_r)                                                   # Scale Y to [0,1]
Y_transf_r = np.arcsin(Y_norm_r)                                                          # Transform Y using arcsine
jb.dump(scaler_ry, 'scaler_ry.pkl')                                                       # Save Y scaler (different from X)

# P_y dataset
cat_cols_p_enc = encoder(cat_cols_p, 'encoder_p.pkl').astype(object)                      # Encode categorical variables
X_p = pd.concat([num_cols_p, cat_cols_p_enc], axis=1)                                     # Combine features
X_sp = feature_scaling(X_p, 'scaler_p.pkl')                                               # Normalize all features                           
Y_p = resp_col_p.to_numpy()                                                               # Convert response variable

# Disp_y dataset
cat_cols_dy_enc = encoder(cat_cols_dy, 'encoder_dy.pkl').astype(object)                   # Encode categorical variables
X_dy = pd.concat([num_cols_dy, cat_cols_dy_enc], axis=1)                                  # Combine features
X_sdy = feature_scaling(X_dy, 'scaler_dy.pkl')                                            # Normalize all features
Y_dy = resp_col_dy.to_numpy()                                                             # Convert response variable

# Disp_u dataset
cat_cols_du_enc = encoder(cat_cols_du, 'encoder_du.pkl').astype(object)                   # Encode categorical variables
X_du = pd.concat([num_cols_du, cat_cols_du_enc], axis=1)                                  # Combine features
X_sdu = feature_scaling(X_du, 'scaler_du.pkl')                                            # Normalize all features          
Y_du = resp_col_du.to_numpy()                                                             # Convert response variable



### ========================= ###  PREPROCESS DATA FOR NEW PREDICTIONS  ### ========================= ###


# Load new data and preprocessing functions
new_beam_df = new_record(beam_data)

# Load processors for each response
enc_m, scaler_m, enc_r, scaler_r, scaler_ry, enc_p, scaler_p, enc_dy, scaler_dy, enc_du, scaler_du = load_processors()

# Preprocess new data using saved processors
X_new_sm = preprocess_input(new_beam_df, enc_m, scaler_m)
X_new_sr = preprocess_input(new_beam_df, enc_r, scaler_r)
X_new_sp = preprocess_input(new_beam_df, enc_p, scaler_p)
X_new_sdy = preprocess_input(new_beam_df, enc_dy, scaler_dy)
X_new_sdu = preprocess_input(new_beam_df, enc_du, scaler_du)



### ========================= ###  TRAIN/TEST/EVALUATE/PREDICT ALL MODELS  ### ========================= ###


"""

Train, test and evaluate each model based on the full dataset and using a Repeated K-Fold cross-validation approach. Repeat the 10-fold cross-validation 10 times (100-fold total) reshuffling the dataset split each 10 folds.

For every trained model and for each response variable, make new predictions based on the user-input data. 

The final output is the mean prediction of all 100 models for each repsonse.


"""

# Initialize the k-fold split
kf_init = 10
fold_shuffle = np.random.randint(10,100,10)

# Empty lists for storing error metrics
r2_m, mse_m, rmse_m, mae_m = [], [], [], []
r2_r, mse_r, rmse_r, mae_r = [], [], [], []
r2_p, mse_p, rmse_p, mae_p = [], [], [], []
r2_dy, mse_dy, rmse_dy, mae_dy = [], [], [], []
r2_du, mse_du, rmse_du, mae_du = [], [], [], []

# Empty lists for combining test sets and predictions (Y = test set, y = prediction).
Y_m_all, Y_r_all, Y_p_all, Y_dy_all, Y_du_all = [], [], [], [], []
y_m_all, y_r_all, y_p_all, y_dy_all, y_du_all = [], [], [], [], []
y_m_new_all, y_r_new_all, y_p_new_all, y_dy_new_all, y_du_new_all = [], [], [], [], []



time_start = time.time()
for j in range(kf_init):
    
    
    kf = KFold(n_splits=10, random_state=fold_shuffle[j], shuffle=True) # Define fold split and reshuffle each loop.
    
    
    ###   M_Max GBRT   ###
    for train_index_m, test_index_m in kf.split(X_sm, Y_m):
        
        X_train_m, X_test_m = X_sm[train_index_m], X_sm[test_index_m]
        Y_train_m, Y_test_m = Y_m[train_index_m], Y_m[test_index_m]
        Y_inv_m = np.exp(Y_test_m) # Convert test set back into original scale
        Y_m_all.append(Y_inv_m)
        
        # Train the optimised model
        gbrt_model_m = gbrt_m()
        gbrt_model_m.fit(X_train_m, Y_train_m.ravel())

        # Test subset - predict the response and convert back to original magnitude
        y_pred_m = gbrt_model_m.predict(X_test_m)
        y_pred_m = np.exp(y_pred_m).reshape(-1,1)
        y_m_all.append(y_pred_m)
        
        # New user-supplied data - predict the response and convert back to original magnitude
        y_pred_m_new = gbrt_model_m.predict(X_new_sm)
        y_pred_m_new = np.exp(y_pred_m_new)
        y_m_new_all.append(y_pred_m_new)

        # Record error metrics from each fold into existing arrays - from test subsets
        r2_m.append(r_squared(Y_inv_m, y_pred_m))
        mse_m.append(mean_squared_err(Y_inv_m, y_pred_m))
        rmse_m.append(root_mean_squared_err(Y_inv_m, y_pred_m))
        mae_m.append(mean_abs_err(Y_inv_m, y_pred_m))             
        
        
        
    ###  R_exp GBRT   ###
    for train_index_r, test_index_r in kf.split(X_sr, Y_transf_r):

        X_train_r, X_test_r = X_sr[train_index_r], X_sr[test_index_r]
        Y_train_r, Y_test_r = Y_transf_r[train_index_r], Y_transf_r[test_index_r]
        
        Y_norm_r = np.sin(Y_test_r) # Convert test set back into original scale
        Y_inv_r = scaler_ry.inverse_transform(Y_norm_r) # Convert normalized Y back to original magnitude
        Y_r_all.append(Y_inv_r)
        
        # Train the optimised model
        gbrt_model_r = gbrt_r()  
        gbrt_model_r.fit(X_train_r, Y_train_r.ravel())

        # Predict the response and convert back to original magnitude
        y_pred_r = gbrt_model_r.predict(X_test_r)
        y_pred_r = np.sin(y_pred_r).reshape(-1,1)
        y_pred_r = scaler_ry.inverse_transform(y_pred_r)
        y_r_all.append(y_pred_r)
        
        # New user-supplied data - predict the response and convert back to original magnitude
        y_pred_r_new = np.sin(gbrt_model_r.predict(X_new_sr)).reshape(-1,1)
        y_pred_r_new = scaler_ry.inverse_transform(y_pred_r_new)
        y_r_new_all.append(y_pred_r_new)

        # Record error metrics from each fold into existing arrays - from test subsets
        r2_r.append(r_squared(Y_inv_r, y_pred_r))
        mse_r.append(mean_squared_err(Y_inv_r, y_pred_r))
        rmse_r.append(root_mean_squared_err(Y_inv_r, y_pred_r))
        mae_r.append(mean_abs_err(Y_inv_r, y_pred_r))      
        
        
        
    ###  P_y GBRT   ###
    for train_index_p, test_index_p in kf.split(X_sp, Y_p):
        
        X_train_p, X_test_p = X_sp[train_index_p], X_sp[test_index_p]
        Y_train_p, Y_test_p = Y_p[train_index_p], Y_p[test_index_p]
        Y_inv_p = np.exp(Y_test_p) # Convert test set back into original scale
        Y_p_all.append(Y_inv_p)
        
        # Train the optimised model
        gbrt_model_p = gbrt_p()
        gbrt_model_p.fit(X_train_p, Y_train_p.ravel())

        # Test subset - predict the response and convert back to original magnitude
        y_pred_p = gbrt_model_p.predict(X_test_p)
        y_pred_p = np.exp(y_pred_p).reshape(-1,1)
        y_p_all.append(y_pred_p)
        
        # New user-supplied data - predict the response and convert back to original magnitude
        y_pred_p_new = gbrt_model_p.predict(X_new_sp)
        y_pred_p_new = np.exp(y_pred_p_new).reshape(-1,1)
        y_p_new_all.append(y_pred_p_new)

        # Record error metrics from each fold into existing arrays
        r2_p.append(r_squared(Y_inv_p, y_pred_p))
        mse_p.append(mean_squared_err(Y_inv_p, y_pred_p))
        rmse_p.append(root_mean_squared_err(Y_inv_p, y_pred_p))
        mae_p.append(mean_abs_err(Y_inv_p, y_pred_p))
        
        
        
    ###   Disp_y RF   ###
    for train_index_dy, test_index_dy in kf.split(X_sdy, Y_dy):
        
        X_train_dy, X_test_dy = X_sdy[train_index_dy], X_sdy[test_index_dy]
        Y_train_dy, Y_test_dy = Y_dy[train_index_dy], Y_dy[test_index_dy]
        Y_inv_dy = np.exp(Y_test_dy) # Convert test set back into original scale
        Y_dy_all.append(Y_inv_dy)

        # Train the optimised model
        rf_model_dy = rf_dy()
        rf_model_dy.fit(X_train_dy, Y_train_dy.ravel())
        
        # Test subset - predict the response and convert back to original magnitude
        y_pred_dy = rf_model_dy.predict(X_test_dy)
        y_pred_dy = np.exp(y_pred_dy).reshape(-1,1)
        y_dy_all.append(y_pred_dy)
        
        # New user-supplied data - predict the response and convert back to original magnitude
        y_pred_dy_new = rf_model_dy.predict(X_new_sdy)
        y_pred_dy_new = np.exp(y_pred_dy_new).reshape(-1,1)
        y_dy_new_all.append(y_pred_dy_new)

        # Record error metrics from each fold into existing arrays
        r2_dy.append(r_squared(Y_inv_dy, y_pred_dy))
        mse_dy.append(mean_squared_err(Y_inv_dy, y_pred_dy))
        rmse_dy.append(root_mean_squared_err(Y_inv_dy, y_pred_dy))
        mae_dy.append(mean_abs_err(Y_inv_dy, y_pred_dy))
        
        
        
    ###   Disp_u RF   ###
    for train_index_du, test_index_du in kf.split(X_sdu, Y_du):
        
        X_train_du, X_test_du = X_sdu[train_index_du], X_sdu[test_index_du]
        Y_train_du, Y_test_du = Y_du[train_index_du], Y_du[test_index_du]
        Y_inv_du = np.exp(Y_test_du) # Convert test set back into original scale
        Y_du_all.append(Y_inv_du)

        # Train the optimised model
        rf_model_du = rf_du()
        rf_model_du.fit(X_train_du, Y_train_du.ravel())
        
        # Test subset - predict the response and convert back to original magnitude
        y_pred_du = rf_model_du.predict(X_test_du)
        y_pred_du = np.exp(y_pred_du).reshape(-1,1)
        y_du_all.append(y_pred_du)
        
        # New user-supplied data - predict the response and convert back to original magnitude
        y_pred_du_new = rf_model_du.predict(X_new_sdu)
        y_pred_du_new = np.exp(y_pred_du_new).reshape(-1,1)
        y_du_new_all.append(y_pred_du_new)

        # Record error metrics from each fold into existing arrays
        r2_du.append(r_squared(Y_inv_du, y_pred_du))
        mse_du.append(mean_squared_err(Y_inv_du, y_pred_du))
        rmse_du.append(root_mean_squared_err(Y_inv_du, y_pred_du))
        mae_du.append(mean_abs_err(Y_inv_du, y_pred_du))

        
    j += 1
    

    
# Calculate mean error metrics - for tested models
mean_test_metrics = {
    "$M_{max}$": [f"{np.mean(r2_m):.4f}", f"{np.mean(mse_m):.4f}", f"{np.mean(rmse_m):.4f}", f"{np.mean(mae_m):.4f}"],
    "$R_{res}$": [f"{np.mean(r2_r):.4f}", f"{np.mean(mse_r):.4f}", f"{np.mean(rmse_r):.4f}", f"{np.mean(mae_r):.4f}"],
    "$P_{y}$": [f"{np.mean(r2_p):.4f}", f"{np.mean(mse_p):.4f}", f"{np.mean(rmse_p):.4f}", f"{np.mean(mae_p):.4f}"],
    r"$\Delta_{y}$": [f"{np.mean(r2_dy):.4f}", f"{np.mean(mse_dy):.4f}", f"{np.mean(rmse_dy):.4f}", f"{np.mean(mae_dy):.4f}"],
    r"$\Delta_{ult}$": [f"{np.mean(r2_du):.4f}", f"{np.mean(mse_du):.4f}", f"{np.mean(rmse_du):.4f}", f"{np.mean(mae_du):.4f}"]}


# Calculate mean error metrics - for new predictions
mean_new_metrics = {
    "$M_{max}$": [f"{np.mean(y_m_new_all):.4f}"],
    "$R_{res}$": [f"{np.mean(y_r_new_all):.4f}"],
    "$P_{y}$": [f"{np.mean(y_p_new_all):.4f}"],
    r"$\Delta_{y}$": [f"{np.mean(y_dy_new_all):.4f}"],
    r"$\Delta_{ult}$": [f"{np.mean(y_du_new_all):.4f}"]}

# Convert to dataframe
error_metrics = ["$R^{2}$", "MSE", "RMSE", "MAE"]
df_test_results = pd.DataFrame(mean_test_metrics, index=error_metrics)
df_new_results = pd.DataFrame(mean_new_metrics)


time_end = time.time()
print("Elapsed time: {} minutes and {:.0f} seconds".format
      (int((time_end - time_start) // 60), (time_end - time_start) % 60)) 



# Final results
df_test_results.head()        # This dataframe contains the mean error metrics of the tested models. 
df_new_results.head()         # This dataframe contains the mean predictions for the new user-supplied input data. 


