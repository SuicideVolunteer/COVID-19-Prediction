# COVID-19-Prediction

Using a dendritic neural regression(DNR) to forecast the transmiision trend of the COVID-19 pandemic. The DNR was trained by a novel scale-free state-of-matter search (SFSMS) algorithm. The program entry is main.m. parameters denote the hyperparameters of DNR-SFSMS. L denotes the embedding dimensions, r represents the time delay. The specific parameters of the experiment are as follows:

India: parameters=[5 0.5 6 100 1000], L=2, r=1, normalized range=[0.3,0.65];
Angola: parameters=[6 0.5 3 100 1000], L=6, r=1, normalized range=[0.2,0.5];
Indonesia: parameters=[5 0.5 5 100 1000], L=4, r=1, normalized range=[0.4,0.469]; 
Ethiopia: parameters=[6 0.5 7 100 1000], L=3, r=1, normalized range=[0.1,0.19];
Azerbaijan: parameters=[6 0.5 4 100 1000], L=4, r=5, normalized range=[0.15,0.255]; 
Israel: parameters=[5 0.5 4 100 1000], L=2, r=3, normalized range=[0.3,0.313];
