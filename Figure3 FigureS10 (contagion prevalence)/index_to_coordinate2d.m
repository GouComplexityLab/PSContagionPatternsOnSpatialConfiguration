function [ cov_x, cov_y ] = index_to_coordinate2d(index, Ny )
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

        cov_y = mod(index,Ny);
        if cov_y == 0; cov_y = Ny; end
        cov_x = floor((index-1)/Ny) + 1;

end