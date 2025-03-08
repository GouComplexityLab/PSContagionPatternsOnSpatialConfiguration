First, prepare the underlying following networks and put them inside:
Generated delta Lattice Embedded Networks L=500 dg=12
Generated mypoisson Lattice Embedded Networks L=500 lam=12
Generated powerlaw Lattice Embedded Networks L=500 mu=3.6231

Then, execute the inside codes in order：
1. GenerateManySameIRandomSeedingICs.py
2. GenerateManyCorrespondingDiskLocalizedSeedingICs.py

3. SimSISModel_Step1_delta12_RandomSeeding.py
4. SimSISModel_Step1_poisson12_RandomSeeding.py
5. SimSISModel_Step1_powerlaw12_RandomSeeding.py
6. SimSISModel_Step2_delta12_DiskLocalizedSeeding.py
7. SimSISModel_Step2_poisson12_DiskLocalizedSeeding.py
8. SimSISModel_Step2_powerlaw12_DiskLocalizedSeeding.py

9. AveSeriesStep1_delta12_RandomSeeding.py
10. AveSeriesStep1_poisson12_RandomSeeding.py
11. AveSeriesStep1_powerlaw12_RandomSeeding.py
12. AveSeriesStep2_delta12_DiskLocalizedSeeding.py
13. AveSeriesStep2_poisson12_LocalizedSeeding.py
14. AveSeriesStep2_powerlaw12_DiskLocalizedSeeding.py

15. Act1_SeriesData_AveIandS_FromDataPath_Step1.m
16. Act1_SeriesData_AveIandS_FromDataPath_Step2.m
17. Act2_SeriesPlot_AveIandS_FromDataPath_Step1.m
18. Act2_SeriesPlot_AveIandS_FromDataPath_Step2_p.m


19. prepare_2dData_Step1_delta12_RandomSeeding.py
20. prepare_2dData_Step1_poisson12_RandomSeeding.py
21. prepare_2dData_Step1_powerlaw12_RandomSeeding.py
22. prepare_2dData_Step2_delta12_DiskLocalizedSeeding.py
23. prepare_2dData_Step2_poisson12_DiskLocalizedSeeding.py
24. prepare_2dData_Step2_powerlaw12_DiskLocalizedSeeding.py

25. MultiPatterns_delta12_Step1_RandomSeeding.m
26. MultiPatterns_delta12_Step2_DiskLocalizedSeeding.m
27. MultiPatterns_poisson12_Step1_RandomSeeding.m
28. MultiPatterns_poisson12_Step2_DiskLocalizedSeeding.m
29. MultiPatterns_powerlaw12_Step1_RandomSeeding.m
30. MultiPatterns_powerlaw12_Step2_DiskLocalizedSeeding.m







