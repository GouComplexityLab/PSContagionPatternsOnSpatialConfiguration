First, prepare the underlying following networks and put them inside:
Generated delta Lattice Embedded Networks L=500 dg=4
Generated delta Lattice Embedded Networks L=500 dg=12
Generated mypoisson Lattice Embedded Networks L=500 lam=12
Generated powerlaw Lattice Embedded Networks L=500 mu=3.6231

Then, execute the inside codes in order：
1. Step0_GenerateSameICs.py

2. Step1_gpucupy_delta4_SimSISModel_many.py
3. Step1_gpucupy_delta12_SimSISModel_many.py
4. Step1_gpucupy_poisson12_SimSISModel_many.py
5. Step1_gpucupy_powerlaw12_SimSISModel_many.py
6. Step2_prepare_2d_Data.py

7. Step3_MultiPatternsDelta4_hOmegaX.m
8. Step3_MultiPatternsDelta12_hOmegaX.m
9. Step3_MultiPatternsPoisson12_hOmegaX.m
10. Step3_MultiPatternsPowerlaw12_hOmegaX.m

11. Act1_SeriesData_AveIandS_FromDataPath_Step1.m
12. Act1_SeriesData_AveIandS_FromDataPath_Step2.m
13. Act2_SeriesPlot_AveIandS_FromDataPath_Step1.m
14. Act2_SeriesPlot_AveIandS_FromDataPath_Step2_p.m

15. Step4_prepare_3d_data.py

16. Step5_Surfaceplot_Delta4Omega13dot36.m
17. Step5_Surfaceplot_Delta12Omega13dot36.m

18. Step3_MultiPatternsDelta4_hOmega13dot36_with_CircularityCalc.m
19. Step3_MultiPatternsDelta12_hOmega13dot36_with_CircularityCalc.m
20. Step3_MultiPatternsPoisson12_hOmega13dot36_with_CircularityCalc.m
21. Step3_MultiPatternsPowerlaw12_hOmega13dot36_withCircularityCalc.m
