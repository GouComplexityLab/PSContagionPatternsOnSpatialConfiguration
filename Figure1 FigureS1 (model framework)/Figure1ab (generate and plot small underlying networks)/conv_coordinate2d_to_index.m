function [ index ] = conv_coordinate2d_to_index(x, y, Ny )
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

    index = y + (x-1) * Ny;

end

