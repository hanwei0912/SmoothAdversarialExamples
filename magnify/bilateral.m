function out = bilateral(in, sigma_s, sigma_r, w)

% parameters:
% sigma_s: domain parameter for spatial kernel
% sigma_r: range parameter for intensity kernel
% w: window size

if nargin < 2, sigma_s = 10; end
if nargin < 3, sigma_r = .2; end
if nargin < 4, w = ceil(-norminv(1e-2, 0, sigma_s)); end

% in1: image to be filtered
% in2: image to guide filtering
if iscell(in), [in1, in2] = deal(in{:});
else           [in1, in2] = deal(in);
end

[x y] = meshgrid(-w:w,-w:w);
domain = exp(-(x.^2+y.^2)/(2*sigma_s^2));

[r c d] = size(in1);
out = zeros(size(in1));

for k=1:d
	display(sprintf('bilateral filtering, channel %d...', k));
	for i=1:r
		for j=1:c
			ir = max(i-w,1):min(i+w,r);
			jr = max(j-w,1):min(j+w,c);

			win1 = in1(ir,jr,k);
			win2 = in2(ir,jr,:);
            
            [n,m,~] = size(win2);
			range = exp(-sum((win2 - repmat(in2(i,j,:),[n,m])).^2, 3) / (2*sigma_r^2));
			bilateral = range .* domain(ir-i+w+1, jr-j+w+1);
			out(i,j,k) = sum(sum(bilateral .* win1)) / sum(bilateral(:));
		end
	end
end