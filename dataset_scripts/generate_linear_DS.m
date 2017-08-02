function [ X , Y ] = generate_linear_DS(n, m, q, noise_sd, xMin, xMax)
%GENERATE_LINEAR_DS generates  a 1-d linear regression dataset with gaussian
%noise
%   INPUT
%   n: number of points
%   m: slope
%   q: intercept
%   noise_sd: standard deviation of the gaussian noise
%   xMin: minimum of the x range
%   xMax: maximum of the x range
%
%   OUTPUT
%   X: Input matrix (N x n)
%   Y: Output matrix ()

    X = rand(1,n) * (xMax - xMin) + xMin;
    yNoiseVec = randn(1,n) * noise_sd;
    Y = m * X + yNoiseVec + q;
    Y = Y';

end

