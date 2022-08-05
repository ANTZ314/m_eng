
import pickle

#=================================================================#
# SAVE THE TRAINED MODEL TO DISK - [UNTESTED]
#=================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [1] Save pickle file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
filename = 'finalized_model.sav'
pickle.dump(hist, open(filename, 'wb'))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [2] Save pickle file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
Pkl_Filename = "Pickle_RL_Model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(LR_Model, file)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [3] Save the trained model as a pickle string.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
saved_model = pickle.dumps(hist)



#=================================================================#
# LOAD THE TRAINED MODEL FROM DISK - [UNTESTED]
#=================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [1] Load the Pickle file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
loaded_model = pickle.load(open(filename, 'rb'))
score = loaded_model.score(X_test, Y_test)			# calculate loaded model score
print("Test score: {0:.2f} %".format(100 * score))	# view score
print(result)									


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [2] Load the Pickle file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)
Pickled_LR_Model									# loaded model
Ypredict = Pickled_LR_Model.predict(Xtest) 			# make prediction with loaded model


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [3] Load the Pickle String
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
model_from_pickle = pickle.loads(saved_model)
