# Predict the Mechanical Response of Corroded Beams Loaded in Flexure
A database aggregating the results of 50 experimental programs worldwide is used to predict the mechanical performance of corroded reinforced concrete beams loaded under flexural bending. The output predictions include the maximum bending moment, residual capacity percentage, yield load, yield displacement, and ultimate displacement.

Based on a comprehensive study comparing the effectiveness of different machine learning methodologies to predict each response variable, the top-performing models were selected for this application. A gradient-boosting regression tree (GBRT) model was trained and optimized for the maximum bending moment (725 beams), residual capacity percentage (717 beams), and yield load (636 beams). A random forest model was trained and optimized for the yield displacement (604 beams) and ultimate displacement (612 beams). 

See the attached Database Key document for all assumptions, analytical estimations, equations, and abbreviations used throughout the database.
The complete open-source database can be accessed at: https://zenodo.org/records/8062007 

# Model Training
Each model is trained and tested using a Repeated k-fold cross-validation approach. A 10-fold split is implemented and repeated ten times, representing a 100-fold  approach, with each dataset randomly reshuffled between repetitions. 

The new predictions are likewise generated for each trained model and stored over 100 folds. The final output is the mean value of the 100 trained models for each response variable. 

# Application
To run the application, open the run.py file in this repository. Insert the required feature information, following the supplied instructions at the top of the script, and run the file. 
The mean prediction of the five mechanical properties should be printed after successfully running the script. Note that the models may take several minutes to loop through 100 training iterations.

# Related Work
An accompanying database and study investigating the mechanical degradation of corroded reinforcing steel can be found at: https://github.com/bma114/corroded-steel-machine-learning 

With a published journal article which can be accessed at: https://doi.org/10.1016/j.conbuildmat.2024.137023

The complete open-source database is available at: https://zenodo.org/records/8035720

# Dependencies
The application includes the following dependencies to run:

*	Python == 3.11.0
*	pandas == 1.4.4
*	numPy == 1.26.4
*	joblib == 1.1.0
*	scikit-learn == 1.0.2

