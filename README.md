# Social_Pain


## Behaviour 
Buddy-boosted Tolerance (in fish)

### Gradient: Analysis of 2 different movies of the same fish comparing conditions to a baseline 

**0. Pre-process** the videos to have a first idea of average position and make sure tracking can be done correctly 

**1. Track** 6 fish at a time in 6 individual wells and extract x and y coordinates of the body, eyes, and heading orientation. 

**2. Measure** swim parameters and store into summary npz files for each fish 

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

**1. Normalize** each voxel to the background value (mode of all voxels histogram)

**2. Compare 2 different groups** and generate mean and diff stacks. Sscale diff stack for display

**3. Overlay** RGB cfos stack on a light DAPI background for better contrast in final figures

**4. Calculate cfos values** within specific ROI and compare values between groups

**5. Cross-Correlation** analysis to evaluate similarities and divergences between groups 

**6. Process individual fish** based on cfos values in ROIs and behavioural metrics


