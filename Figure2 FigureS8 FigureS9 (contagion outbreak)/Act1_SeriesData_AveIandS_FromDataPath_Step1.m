clear all
close all
clc

gwLineWidth = 1.0;
gwcolors = lines(6);

ICns = [1];

main_path = ['./Series dataStep1'];
pvalues = [0.1:0.01:0.25,0.3];
Netname = 'Delta12';
% Netname = 'Poisson12';
% Netname = 'Powerlaw12';

ending_averagedI = zeros(length(ICns),length(pvalues));
gwcolors_p = lines(length(pvalues));

for ICn = ICns

    p_n = 1;
    for p = pvalues
    
        SeriesData_Path = [main_path '/Series Step1 Cupy ' Netname ' Figure ICn=' num2str(ICn, '%.f') ' N=250000 Omega=13.36 beta=0.01 nu=0.2 p=' num2str(p, '%.3f')];
        disp(['SeriesData Path:' SeriesData_Path])
        SeriesData_Name = ['/AveSeries.mat'];

        SeriesData_PathName = strcat(SeriesData_Path,SeriesData_Name);
        load(SeriesData_PathName)
        
        ending_averagedI(ICn,p_n) = array_series_averagedI(end);
        
        p_n = p_n + 1;
    
    end

end



mat_name = ['./SeriesDataStep1RandomSeeding' Netname '.mat'];
save(mat_name, 'pvalues','ending_averagedI','ICns')


%% 2
% Netname = 'Delta12';
Netname = 'Poisson12';
% Netname = 'Powerlaw12';

ending_averagedI = zeros(length(ICns),length(pvalues));
gwcolors_p = lines(length(pvalues));

for ICn = ICns


    p_n = 1;
    for p = pvalues
    
        SeriesData_Path = [main_path '/Series Step1 Cupy ' Netname ' Figure ICn=' num2str(ICn, '%.f') ' N=250000 Omega=13.36 beta=0.01 nu=0.2 p=' num2str(p, '%.3f')];
        disp(['SeriesData Path:' SeriesData_Path])
        SeriesData_Name = ['/AveSeries.mat'];

        SeriesData_PathName = strcat(SeriesData_Path,SeriesData_Name);
        load(SeriesData_PathName)
        
        ending_averagedI(ICn,p_n) = array_series_averagedI(end);
        
        p_n = p_n + 1;
    
    end

end



mat_name = ['./SeriesDataStep1RandomSeeding' Netname '.mat'];
save(mat_name, 'pvalues','ending_averagedI','ICns')


%% 3

% Netname = 'Delta12';
% Netname = 'Poisson12';
Netname = 'Powerlaw12';

ending_averagedI = zeros(length(ICns),length(pvalues));
gwcolors_p = lines(length(pvalues));

for ICn = ICns


    p_n = 1;
    for p = pvalues
    
        SeriesData_Path = [main_path '/Series Step1 Cupy ' Netname ' Figure ICn=' num2str(ICn, '%.f') ' N=250000 Omega=13.36 beta=0.01 nu=0.2 p=' num2str(p, '%.3f')];
        disp(['SeriesData Path:' SeriesData_Path])
        SeriesData_Name = ['/AveSeries.mat'];

        SeriesData_PathName = strcat(SeriesData_Path,SeriesData_Name);
        load(SeriesData_PathName)
        
        ending_averagedI(ICn,p_n) = array_series_averagedI(end);
        
        p_n = p_n + 1;
    
    end

end



mat_name = ['./SeriesDataStep1RandomSeeding' Netname '.mat'];
save(mat_name, 'pvalues','ending_averagedI','ICns')
