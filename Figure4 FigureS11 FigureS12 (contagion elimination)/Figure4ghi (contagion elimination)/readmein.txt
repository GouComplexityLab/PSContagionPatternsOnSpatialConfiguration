First, prepare the underlying following networks and put them inside:
Generated delta Lattice Embedded Networks L=500 dg=4
Generated delta Lattice Embedded Networks L=500 dg=12
Generated mypoisson Lattice Embedded Networks L=500 lam=12
Generated powerlaw Lattice Embedded Networks L=500 mu=3.6231

Second, prepare the control states and put them inside:
Step1 Cupy Delta12 Data N=250000 Omega=13.36 beta=0.01 nu=0.2
Step1 Cupy Poisson12 Data N=250000 Omega=13.36 beta=0.01 nu=0.2
Step1 Cupy Powerlaw12 Data N=250000 Omega=13.36 beta=0.01 nu=0.2


Then, execute the inside codes in order：
1. Step2_gpucupy_delta12_SimSISModel_IsolateIQ1_KeepDs.py
2. Step2_gpucupy_delta12_SimSISModel_IsolateIQ1_DecreaseDs.py
3. Step3_gpucupy_delta12_SimSISModel_DecreaseBeta_KeepDs.py
4. Step3_gpucupy_delta12_SimSISModel_DecreaseBeta_DecreaseDs.py
5. Step4_gpucupy_delta12_SimSISModel_DecreaseNu_KeepDs.py
6. Step4_gpucupy_delta12_SimSISModel_DecreaseNu_DecreaseDs.py

7. ChoosePlotStep2_delta12.py
8. ChoosePlotStep3_delta12.py
9. ChoosePlotStep4_delta12.py


10. Step2_gpucupy_poisson12_SimSISModel_IsolateIQ1_KeepDs.py
11. Step2_gpucupy_poisson12_SimSISModel_IsolateIQ1_DecreaseDs.py
12. Step3_gpucupy_poisson12_SimSISModel_DecreaseBeta_KeepDs.py
13. Step3_gpucupy_poisson12_SimSISModel_DecreaseBeta_DecreaseDs.py
14. Step4_gpucupy_poisson12_SimSISModel_DecreaseNu_KeepDs.py
15. Step4_gpucupy_poisson12_SimSISModel_DecreaseNu_DecreaseDs.py

16. ChoosePlotStep2_poisson12.py
17. ChoosePlotStep3_poisson12.py
18. ChoosePlotStep4_poisson12.py

19. Step2_gpucupy_powerlaw12_SimSISModel_IsolateIQ1_KeepDs.py
20. Step2_gpucupy_powerlaw12_SimSISModel_IsolateIQ1_DecreaseDs.py
21. Step3_gpucupy_powerlaw12_SimSISModel_DecreaseBeta_KeepDs.py
22. Step3_gpucupy_powerlaw12_SimSISModel_DecreaseBeta_DecreaseDs.py
23. Step4_gpucupy_powerlaw12_SimSISModel_DecreaseNu_KeepDs.py
24. Step4_gpucupy_powerlaw12_SimSISModel_DecreaseNu_DecreaseDs.py

25. ChoosePlotStep2_powerlaw12.py
26. ChoosePlotStep3_powerlaw12.py
27. ChoosePlotStep4_powerlaw12.py

