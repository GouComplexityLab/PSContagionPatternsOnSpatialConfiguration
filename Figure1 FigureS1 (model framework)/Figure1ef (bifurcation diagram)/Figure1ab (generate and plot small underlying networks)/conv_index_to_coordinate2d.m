function [ cov_x, cov_y ] = conv_index_to_coordinate2d(index, Ny )
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��


        cov_x = floor((index-1)/Ny) + 1;
        
        cov_y = mod(index,Ny);
        p = find(cov_y == 0);
        if length(p)>0; cov_y(p) = Ny; end
%         if cov_y == 0; cov_y = Ny; end
        
end