# Expectation-Maximization
The program will take as input a set of (n-dimensional) data points that come from 1 or more clusters, and the number of clusters to find.  For example:
"em data.txt 3" would read in the points from data.txt, and return the best 3 cluster centers it can find.  

The goal is to use expectation maximization with random restarts to return the best-fitting cluster centers. The program should output the best-fitting cluster centers (mean and variance), as well as the log-likelihood of the model.  

It may be assumed that the data points are generated from an n-dimensional Gaussian where each dimension has a differing mean and variance and the dimensions are independent of each other.

Final_main file initializes and runs the program

em_final file has functions defined for the program


## Execute File 
Run the program using python3 final_main.py sample.txt #clusters
