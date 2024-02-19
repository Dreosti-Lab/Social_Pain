# Social_Pain
Lonely and hurting (fish)

## Behaviour 

Gradient : Analysis of 2 different movies of the same fish comparing conditions to a baseline 

0. Pre-process the videos to have a first idea of average position and make sure tracking can be done correctly 
1. Track 6 fish at a time in 6 individual wells and extract, body, eye and orientation parameters for each frame
2. Extract information from the tracking data and store into summary npz files for each fish 
3. Analyse and plot the summary data and save the plots in a figure folder 
4. Compare different conditions directly 
5. Use kmeans clustering to define clusters of fish based on their average position in the chamber. Generate an excel file with the saved localisation of each fish and their group belonging. 
6. Analyse and plot parameters for different groups
7. Create a summary fingerprint comparing the change (incerease or decrease) in analysed parameters

Noxious: Analysis of 1 long movie 
