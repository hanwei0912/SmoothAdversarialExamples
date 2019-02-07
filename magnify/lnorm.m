function out = lnorm(in, sigma1, sigma2, type)

if nargin < 2, sigma1 = 5; end
if nargin < 3, sigma2 = 5; end
if nargin < 4, type = 'b'; end

width = @(s) 2 * ceil(-norminv(1e-2,0,s)) + 1;

switch type
case 'g', f = @(x,s) imfilter(first(x), fspecial('gaussian',width(s),s), 'symmetric');
case 'b', f = @(x,s) bilateral(x,s);
otherwise error('unexpected filter type')
end

[n,m,~] = size(in);
num = in - repmat(mean(mean(in)),[n,m]);
glob = sqrt(mean(mean(num.^2)));
alpha = 0.2;

num = in - f(in, sigma1);
den = sqrt(f({num.^2,in}, sigma2));
out = num ./ ((1-alpha) * den + repmat(alpha * glob,[n,m]));
