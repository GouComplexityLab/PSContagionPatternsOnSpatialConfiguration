This repository provides the Python and Matlab codes that simulate the phase-separated contagion patterns on complex spatial configurations from our manuscript titled "Phase-separated contagion patterns in complex spatial configurations jeopardize efforts towards elimination" by Jianmeng Cui, Wei Gou*, and Zhen Jin*, submitted in 2025.
    
![](https://github.com/GouComplexityLab/PSContagionPatternsOnSpatialConfiguration/blob/main/Fig3.png)
Figure: Phase-separated contagion patterns on complex spatial configurations (lattice embedded networks).


# Guide for Code
This folder includes several subfolders for different Figures in main text and Supplementary Information.
## Figure0 (generate underlying networks):
* generate_lattice_embedded_networks.py: Generate the lattice embedded networks with the delta, Poisson and power-law degree distribution, which will be used as the underlying networks for our conserving reaction-diffusion model later.

* eigenvalues_eigenvectors.py: Calculate the smallest few Laplacian eigenvalues of the previously generated lattice embedded networks.
## Figure1 FigureS1 (model framework):
### in subfolder Figure1ab (generate and plot small underlying networks):
* generate_small_lattice_embedded_networks.py Generate the illustrative small-size lattice embedded networks with the delta, Poisson and power-law degree distribution plotted in Figure1 and FigureS1 through taking $L=100$.
* edge_plot2dLis100powerlaw12.m Plot the illustrative small-size lattice embedded networks with power-law degree distribution in Figure1a (FigureS1e) and its part in Figure1b (FigureS1f).
* edge_plot2dLis100delta12.m and edge_plot2dLis100poisson12.m Plot the illustrative small-size lattice embedded networks with delta and Poisson degree distribution and their parts in FigureS1.

### in subfolder Figure1ef (bifurcation diagram):
* figure1e_backward_bifurcation.m Plot the backward bifurcation diagram of model (1) in Figure1e.

* BinodalSpinodalRegion_calc_largenet.m Calculate the spinodal and binodal curves of model (2) plotted in Figure1f.

* figure1f_BinodalSpinodalRegion_plot_largenet.m Plot the spinodal and binodal curves of model (2) in Figure1f.

## Figure2 FigureS8 FigureS9 (contagion outbreak):
* Simulate the contagion outbreak behaviors in random and localized seeding on the underlying lattice embedded networks with power-law degree distribution (Figure2), delta degree distribution (FigureS8) and Poisson degree distribution (FigureS9).
* Seel more details in readmein.txt.

## Figure3 FigureS10 (contagion prevalence):
* Simulate the contagion prevalence behaviors, i.e., phase-separated contagion patterns on the underlying lattice embedded networks with delta, Poisson and power-law degree distribution in Figure3 and FigureS10.
* Seel more details in readmein.txt.

## Figure4 FigureS11 FigureS12 (contagion elimination):
### in subfolder Figure4abc：
* Simulate the effect of the control measures, namely, isolating infectious individuals and reducing transmission rates, for contagion elimination in the model (1).

### in subfolder Figure4def：
* Calculate and plot the spinodal curves (regions) of the model (2) defined on the lattice embedded networks with power-law degree distribution under the control measures, namely, isolating infectious individuals and reducing transmission rates.

### in subfolder Figure4ghi (contagion elimination)：
* Simulate the influence of phase-separated patterns for contagion elimination under the control measures, namely, isolating infectious individuals and reducing transmission rates, on the underlying lattice embedded networks with power-law degree distribution (Figure2), delta degree distribution (FigureS8) and Poisson degree distribution (FigureS9).
* Seel more details in readmein.txt.

## Figure5 (contagion patterns on gridded urban landscapes):
* Simulate the phase-separated contagion patterns in gridded urban landscapes, taking Beijing as an example.
* step1_beijingshpfileread.py Generate the gridded urban landscapes for Beijing.
* step2_BeijingNetGenerate_EuclideanPowerlaw12.py Generate the associated networks in the gridded urban landscapes for Beijing.

* Seel more details in readmein.txt.
### in subfolder dataBeijing:
* The subsubfolder beijingshp provides the shp data with population densities of Beijing’s 338 township and street administrative regions.

## FigureS2 (bifurcation diagram):
* FigureS2a_backward.m Plot the backward bifurcation diagram of model (1) in FigureS2a.
* FigureS2b_forward.m Plot the forward bifurcation diagram of model (1) in FigureS2b.

## FigureS3 (spinodal regions on small networks):
* Calculate and plot the spinodal curves (regions) of the model (2) defined on different lattice embedded networks with tuned network size and alternated degree distribution.

## FigureS4 (intersection cases):
* Plot the intersection points in $I−S$ plane of the line $d_{S}S + d_{I}I = P^{\*}$ and the curve $f(S, I) = 0$ in two different cases.

## FigureS5 FigureS6 (variational analysis):
* Plot the equation $g(P^{\*},Q)$ and potential function $V(P^{\*},Q)$ for different $P^{\*}$ in the variational analysis.

## FigureS7 (binodal and spinodal regions):
* Plot the binodal and spinodal curves (regions) of model (2) in two different cases.

# Questions:
Please contact Wei Gou, wgou@sxu.edu.cn (or goucomplexitylab@163.com) if you have any questions on the code taste (execution).
