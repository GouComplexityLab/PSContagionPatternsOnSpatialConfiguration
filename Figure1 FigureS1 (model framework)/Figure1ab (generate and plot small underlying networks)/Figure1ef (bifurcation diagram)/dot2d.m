function str = dot2d(num)
str_pre = num2str(num,'%.6f');
% str(str == '.') = 'dot';
str = strrep(str_pre,'.','dot');