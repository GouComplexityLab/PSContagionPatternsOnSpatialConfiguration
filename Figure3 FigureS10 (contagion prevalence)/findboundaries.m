function [ InBoundaryPoints_index, InnerPoints_index, OutBoundaryPoints_index ] = findboundaries( underlying_grid,  underlyingpoints_index, Ny)

    underlying_grid(underlyingpoints_index) = 1;
    [px, py] = gradient(underlying_grid);
    px_m = find(px>0); px_p = find(px<0); px_mp = union(px_m, px_p);
    py_m = find(py>0); py_p = find(py<0); py_mp = union(py_m, py_p);

    pxy_mp = union(px_mp, py_mp);

    InBoundaryPoints_index = intersect(underlyingpoints_index, pxy_mp);
    InnerPoints_index = setdiff(underlyingpoints_index, InBoundaryPoints_index);
    OutBoundaryPoints_index = setdiff(pxy_mp, underlyingpoints_index);
end

