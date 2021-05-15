
"""
Created on Wed Apr  7 19:04:21 2021
@author: alizeekastler

Create a scatterplot showing the position of fish in Non Social relative to Social trials 
Compare between conditions 
Change between Mean and Mode 
"""

# -----------------------------------------------------------------------------
# Set Library Path - Social Pain Repo
lib_path = r'C:\Repos\Social_Pain\libs'
import sys
sys.path.append(lib_path)
# Set Base Path
base_path = r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient'
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics as st

# Import local modules
import SP_Utilities as SPU


# Read folder list files
Folderlist_Heat = base_path + '/Folderlist_Heat.txt' 
Folderlist_Control = base_path + '/Folderlist_Control.txt' 
Folderlist_Lidocaine= base_path + '/Folderlist_Lidocaine.txt' 
Folderlist_Isolated = base_path + '/Folderlist_Isolated.txt'
Folderlist_L368_899 = base_path + '/Folderlist_L368,899.txt'

#Heat
groups, ages, folderNames, fishStatus = SPU.read_folder_list(Folderlist_Heat)
# XMs
XMs_Heat = []

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    # ---------------------
    # Analyze Tracking for each fish 
    for i in range(0,6):
        fish_number = i + 1
     
        # Extract tracking data (NS)     
        tracking_file = NS_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking']
        fx_NS = tracking[:,0] 
   
        # Extract tracking data (S)
        tracking_file = S_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking'] 
        fx_S = tracking[:,0][0:60000] 
       
        # Store XMs
        #XMs_Heat.append([np.mean(fx_NS), np.mean(fx_S)])
        XMs_Heat.append([st.mode(fx_NS), st.mode(fx_S)]) #use if you want the mode 
        Heat = np.array(XMs_Heat)
    # Report
    print("Next File: {0}".format(idx))

#Control
groups, ages, folderNames, fishStatus = SPU.read_folder_list(Folderlist_Control)

# XMs
XMs_Control = []

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    # ---------------------
    # Analyze Tracking for each fish 
    for i in range(0,6):
        fish_number = i + 1
     
        # Extract tracking data (NS)     
        tracking_file = NS_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking']
        fx_NS = tracking[:,0] 
   
        # Extract tracking data (S)
        tracking_file = S_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking'] 
        fx_S = tracking[:,0][0:60000]
       
        # Store XMs
        #XMs_Control.append([np.mean(fx_NS), np.mean(fx_S)])
        XMs_Control.append([st.mode(fx_NS), st.mode(fx_S)]) #use if you want the mode 
        Control = np.array(XMs_Control)
    # Report
    print("Next File: {0}".format(idx))

#Lidocaine
groups, ages, folderNames, fishStatus = SPU.read_folder_list(Folderlist_Lidocaine)

# XMs
XMs_Lidocaine = []

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    # ---------------------
    # Analyze Tracking for each fish 
    for i in range(0,6):
        fish_number = i + 1
     
        # Extract tracking data (NS)     
        tracking_file = NS_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking']
        fx_NS = tracking[:,0] 
   
        # Extract tracking data (S)
        tracking_file = S_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking'] 
        fx_S = tracking[:,0][0:60000] 
       
        # Store XMs
        #XMs_Lidocaine.append([np.mean(fx_NS), np.mean(fx_S)])
        XMs_Lidocaine.append([st.mode(fx_NS), st.mode(fx_S)]) #use if you want the mode 
        Lidocaine = np.array(XMs_Lidocaine)
    # Report
    print("Next File: {0}".format(idx))

#Isolated
groups, ages, folderNames, fishStatus = SPU.read_folder_list(Folderlist_Isolated)
# XMs
XMs_Isolated = []

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    # ---------------------
    # Analyze Tracking for each fish 
    for i in range(0,6):
        fish_number = i + 1
     
        # Extract tracking data (NS)     
        tracking_file = NS_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking']
        fx_NS = tracking[:,0] 
   
        # Extract tracking data (S)
        tracking_file = S_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking'] 
        fx_S = tracking[:,0][0:60000]
       
        # Store XMs
        #XMs_Isolated.append([np.mean(fx_NS), np.mean(fx_S)])
        XMs_Isolated.append([st.mode(fx_NS), st.mode(fx_S)]) #use if you want the mode 
        Isolated = np.array(XMs_Isolated)
    # Report
    print("Next File: {0}".format(idx))

#L368,899
groups, ages, folderNames, fishStatus = SPU.read_folder_list(Folderlist_L368_899)
# XMs
XMs_L368_899 = []

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    # ---------------------
    # Analyze Tracking for each fish 
    for i in range(0,6):
        fish_number = i + 1
     
        # Extract tracking data (NS)     
        tracking_file = NS_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking']
        fx_NS = tracking[:,0] 
   
        # Extract tracking data (S)
        tracking_file = S_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking'] 
        fx_S = tracking[:,0]
       
        # Store XMs
        #XMs_Isolated.append([np.mean(fx_NS), np.mean(fx_S)])
        XMs_L368_899.append([st.mode(fx_NS), st.mode(fx_S)]) #use if you want the mode 
        L368_899 = np.array(XMs_L368_899)
    # Report
    print("Next File: {0}".format(idx))
Position_Heat = pd.DataFrame(data = Heat, columns = ["Non_Social","Social"])
Position_Heat['condition']='Heat'

Position_Control = pd.DataFrame(data = Control, columns = ["Non_Social","Social"])
Position_Control['condition']='control'

Position_Lidocaine = pd.DataFrame(data = Lidocaine, columns = ["Non_Social","Social"])
Position_Lidocaine['condition']='Lidocaine'

Position_Isolated = pd.DataFrame(data = Isolated, columns = ["Non_Social","Social"])
Position_Isolated['condition']='Isolated'

Position_L368_899 = pd.DataFrame(data = L368_899, columns = ["Non_Social","Social"])
Position_L368_899['condition']='L368_899'

Position = Position_Heat.append([Position_Control, Position_Lidocaine, Position_Isolated, Position_L368_899 ])



plt.figure(figsize=(10,10), dpi=300)
plt.axis([250,900,250,900])
ax=sns.scatterplot(data=Position, x='Non_Social', y='Social', hue='condition', palette=['steelblue', 'springgreen','coral', 'gold', 'hotpink'])
ax.set(xlabel='Non_Social XM(px)', ylabel='Social XM(px)')
plt.title('Most Current Position Non_Social vs Social', size=16)
#plt.title('Mean Position Non_Social vs Social', size=16)
plt.show()





S_Heat = np.count_nonzero(Position_Heat['Social'][Position_Heat['Social']>700])
S_Control = np.count_nonzero(Position_Control['Social'][Position_Control['Social']>700])
S_Lidocaine = np.count_nonzero(Position_Lidocaine['Social'][Position_Lidocaine['Social']>700])
S_Isolated = np.count_nonzero(Position_Isolated['Social'][Position_Isolated['Social']>700])

P_Heat_High = (S_Heat/len(Position_Heat['Social'])) 
P_Control_High = (S_Control/len(Position_Control['Social']))          
P_Lidocaine_High = (S_Lidocaine/len(Position_Lidocaine['Social'])) 
P_Isolated_High = (S_Isolated/len(Position_Isolated['Social'])) 


PHS1 = pd.Series(P_Heat_High, name='Heat')
PHS2 = pd.Series(P_Control_High, name='Control')
PHS3 = pd.Series(P_Lidocaine_High, name='Lidocaine')
PHS4 = pd.Series(P_Isolated_High, name= 'Isolated')
Proportion_High = pd.concat([PHS1,PHS2,PHS3, PHS4], axis=1)

plt.figure(figsize=(4,8), dpi=300)
plt.ylim(0,1)
sns.barplot(data=Proportion_High, ci ='sd', palette= ['steelblue','springgreen','coral','gold'], dodge= False)
plt.title('Proportion of fish > 700px')
ax.set(ylabel= 'Proportion Fish')
sns.despine() 
plt.show()



NS_Heat = np.count_nonzero(Position_Heat['Social'][Position_Heat['Social']<500])
NS_Control = np.count_nonzero(Position_Control['Social'][Position_Control['Social']<500])
NS_Lidocaine = np.count_nonzero(Position_Lidocaine['Social'][Position_Lidocaine['Social']<500])
NS_Isolated = np.count_nonzero(Position_Isolated['Social'][Position_Isolated['Social']<500])

P_Heat_Low = (NS_Heat/len(Position_Heat['Social'])) 
P_Control_Low = (NS_Control/len(Position_Control['Social']))          
P_Lidocaine_Low = (NS_Lidocaine/len(Position_Lidocaine['Social'])) 
P_Isolated_Low = (NS_Isolated/len(Position_Isolated['Social'])) 


PLS1 = pd.Series(P_Heat_Low, name='Heat')
PLS2 = pd.Series(P_Control_Low, name='Control')
PLS3 = pd.Series(P_Lidocaine_Low, name='Lidocaine')
PLS4 = pd.Series(P_Isolated_Low, name= 'Isolated')
Proportion_Low = pd.concat([PLS1,PLS2,PLS3, PLS4], axis=1)

plt.figure(figsize=(4,8), dpi=300)
plt.ylim(0,1)
sns.barplot(data=Proportion_Low, ci ='sd', palette= ['steelblue','springgreen','coral','gold'], dodge= False)
plt.title('Proportion of fish < 500px')
ax.set(ylabel= 'Proportion Fish')
sns.despine() 
plt.show()


        


#FIN

