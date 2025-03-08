function [ index ] = conv_coordinate2d_to_index(x, y, Ny )
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明

    index = y + (x-1) * Ny;

end

