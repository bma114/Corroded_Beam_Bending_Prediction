import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn import preprocessing


##############################################################################################################################


# load data worksheets
def load_data():
    
    df_m = pd.read_excel("Refined Databasev2 -- All Responses.xlsx", sheet_name="M_max_exp", keep_default_na=False)
    df_r = pd.read_excel("Refined Databasev2 -- All Responses.xlsx", sheet_name="R_exp", keep_default_na=False)
    df_p = pd.read_excel("Refined Databasev2 -- All Responses.xlsx", sheet_name="P_y", keep_default_na=False)
    df_dy = pd.read_excel("Refined Databasev2 -- All Responses.xlsx", sheet_name="Disp_y", keep_default_na=False)
    df_du = pd.read_excel("Refined Databasev2 -- All Responses.xlsx", sheet_name="Disp_u", keep_default_na=False)

    return (df_m, df_r, df_p, df_dy, df_du)
    # keep_default_na=False ensures data with N/A are interpretted as strings not missing numbers.


    
# Seperate data into types
def col_types(df):
    
    # Categorical Features
    cat_cols = df.iloc[:, [0, 5, 7, 8, 11, 14]]
    cat_cols.columns = ["Cross_section", "Reinforcement_Design", "Longitudinal_Type", 
                "End_Anchorage", "Stirrup_Type", "Corrosion_Method"] 
    
    # Numerical Features
    num_cols = df.iloc[:, [1, 2, 3, 4, 6, 9, 10, 12, 13, 15, 16, 17, 18]]
    num_cols.columns = ["W (mm)", "D (mm)", "L (mm)", "Bottom Cover (mm)", "Tension Ratio (%)", 
                "fy (MPa)", "fsu (MPa)", "Volumetric Ratio", "fc (MPa)", "Icorr", 
                "Duration (days)", "Mass Loss (%)", "Shear Span (mm)"]
    
    # Response variables
    resp_cols = df.iloc[:, [20]]
    
    return cat_cols, num_cols, resp_cols



# Encode categorical features  
def encoder(df, save_path='encoder.pkl'):
    
    enc = OrdinalEncoder()
    enc_array = enc.fit_transform(df)
    df_enc = pd.DataFrame(enc_array)
    
    # Save encoder
    jb.dump(enc, save_path)    
    
    return df_enc



# Normalize numerical features
def feature_scaling(df, save_path='scaler.pkl'):
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Save scaler
    jb.dump(scaler, save_path)  
    
    return df_scaled



# M_max GBRT Model
def gbrt_m():

    gbrt_model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.2, max_depth=2, 
                                               max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=4,
                                               random_state=0, loss='squared_error')            
    return gbrt_model



# R_exp GBRT Model
def gbrt_r(): # Add optimized hyperparameters into GBRT model

    gbrt_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.25, max_depth=2, 
                                           max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=2,
                                           random_state=0, loss='squared_error')            
    return gbrt_model



# P_y GBRT Model
def gbrt_p(): # Add optimized hyperparameters into GBRT model

    gbrt_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.15, max_depth=2, 
                                           max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=6,
                                           random_state=0, loss='squared_error')            
    return gbrt_model



# Disp_y RF Model
def rf_dy():

    rf_model = RandomForestRegressor(n_estimators=500, max_depth=7, max_features='sqrt', 
                                     min_samples_leaf=1, min_samples_split=2, bootstrap=True,
                                     random_state=25, n_jobs=-1, criterion='squared_error')
    return rf_model



# Disp_u RF Model
def rf_du():

    rf_model = RandomForestRegressor(n_estimators=500, max_depth=7, max_features='sqrt', 
                                     min_samples_leaf=1, min_samples_split=2, bootstrap=True, 
                                     random_state=25, n_jobs=-1, criterion='squared_error')
    return rf_model



# Organise data for new prediction into a dataframe
def new_record(beam_data):

    new_beam = pd.DataFrame([beam_data], columns=[
        "Cross_section", "Reinforcement_Design", "Longitudinal_Type", "End_Anchorage", 
        "Stirrup_Type", "Corrosion_Method", "W (mm)", "D (mm)", "L (mm)", 
        "Bottom Cover (mm)", "Tension Ratio (%)", "fy (MPa)", "fsu (MPa)", 
        "Volumetric Ratio", "fc (MPa)", "Icorr", 
        "Duration (days)", "Mass Loss (%)", "Shear Span (mm)"
    ])
    
    return new_beam



# Preprocess the new data inputs
def preprocess_input(new_beam_df, encoder, scaler):
    
    # Split the new input dataframe into categorical and numerical parts
    cat_cols = ["Cross_section", "Reinforcement_Design", "Longitudinal_Type", 
                "End_Anchorage", "Stirrup_Type", "Corrosion_Method"]
    num_cols = ["W (mm)", "D (mm)", "L (mm)", "Bottom Cover (mm)", "Tension Ratio (%)", 
                "fy (MPa)", "fsu (MPa)", "Volumetric Ratio", "fc (MPa)", "Icorr", 
                "Duration (days)", "Mass Loss (%)", "Shear Span (mm)"]
    
    # Encode categorical columns
    cat_df = new_beam_df[cat_cols]
    cat_df_encoded = encoder.transform(cat_df).astype(object)
    cat_df_encoded = pd.DataFrame(cat_df_encoded, columns=cat_cols)

    # Combine numerical and encoded categorical features
    num_df = new_beam_df[num_cols]
    X_new = pd.concat([num_df, cat_df_encoded], axis='columns')
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled = pd.DataFrame(X_new_scaled, columns=X_new.columns)

    return X_new_scaled



def load_processors():
    
    # Load encoder and scaler for each model 
    enc_m = jb.load('encoder_m.pkl')
    scaler_m = jb.load('scaler_m.pkl')
    
    enc_r = jb.load('encoder_r.pkl')
    scaler_r = jb.load('scaler_r.pkl') # MinMax scaler for features
    scaler_ry = jb.load('scaler_ry.pkl') # MinMax scaler for R_res response
    
    enc_p = jb.load('encoder_p.pkl')
    scaler_p = jb.load('scaler_p.pkl')
    
    enc_dy = jb.load('encoder_dy.pkl')
    scaler_dy = jb.load('scaler_dy.pkl')
    
    enc_du = jb.load('encoder_du.pkl')
    scaler_du = jb.load('scaler_du.pkl')
    
    return enc_m, scaler_m, enc_r, scaler_r, scaler_ry, enc_p, scaler_p, enc_dy, scaler_dy, enc_du, scaler_du



# Define R^2 error function
def r_squared(Y, y_hat):
    y_bar = Y.mean()
    ss_res = ((Y - y_hat)**2).sum()
    ss_tot = ((Y - y_bar)**2).sum()
    return 1 - (ss_res/ss_tot)



# Define MSE error function
def mean_squared_err(Y, y_hat):
    var = ((Y - y_hat)**2).sum()
    n = len(Y)
    return var/n



# Define RMSE error function
def root_mean_squared_err(Y, y_hat):
    MSE = mean_squared_err(Y, y_hat)
    return np.sqrt(MSE)



# Define MAE error function
def mean_abs_err(Y, y_hat):
    abs_var = (np.abs(Y - y_hat)).sum()
    n = len(Y)
    return abs_var/n







