function y = imnorm(x)

mx = max(x(:));
mn = min(x(:));
y = (x - mn) ./ (mx - mn);