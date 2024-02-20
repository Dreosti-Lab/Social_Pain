# Social_Pain
Lonely and hurting (fish)

## Behaviour 

### Gradient: Analysis of 2 different movies of the same fish comparing conditions to a baseline 

**0. Pre-process** the videos to have a first idea of average position and make sure tracking can be done correctly 

**1. Track** 6 fish at a time in 6 individual wells and extract x and y coordinates of the body, eyes, and heading orientation. 

**2.Measure** swim parameters and store into summary npz files for each fish 

**3. Plot** the summary data and save the plots in a figure folder 

**4. Compare** different conditions

**5. Cluster** fish based on their average position in the chamber using k-means. 

**6. Compare** swim parameters for different groups

**7. Summary** fingerprint comparing the change (increase or decrease) in analysed parameters

### Noxious: Analysis of 1 long movie 

**1. Track** 6 fish at a time in 6 individual wells and extract x and y coordinates of the body, eyes, and heading orientation.

**2. Measure** swim parameters and store into summary npz files for each fish 

**3. Plot** the summary data and compare different conditions. Save plots in a figure folder 

**4. Summary** fingerprint comparing the change (increase or decrease) in analysed parameters for each condition compared to baseline

## Imaging
