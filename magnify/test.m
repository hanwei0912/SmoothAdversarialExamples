% data = {
% 	'ori-flag.png',
% 	'sgd-z.png',
% 	'sgd-f.png',
% 	'sgd-e.png',
% 	'gd-z.png',
% 	'gd-f.png',
% 	'gd-e.png'
% };

data = {
    'oriWsgd.png',
    'oriWgd.png',
    'oriWcw.png',
    'oriWclip.png',
    'BmnistWfgsmMfgsm.png',
	'BmnistWfgsmMcw.png',
	'BmnistWfgsmMclip.png',
	'BmnistWfgsmMbim.png',
	'BmnistWbimMfgsm.png',
	'BmnistWbimMcw.png',
	'BmnistWbimMclip.png',
	'BmnistWbimMbim.png',
    'BmnistWcwMfgsm.png',
	'BmnistWcwMcw.png',
	'BmnistWcwMclip.png',
	'BmnistWcwMbim.png',
    'BmnistWclipMfgsm.png',
	'BmnistWclipMcw.png',
	'BmnistWclipMclip.png',
	'BmnistWclipMbim.png'
};

sel = 1:length(data);
% sel = [1 2 3 4 5 6 7 8];
%  sel = [1 2];
%  sel = 2;

%  type = 'g'; % Gaussian
type = 'b'; % bilateral

len = length(sel);
im = cell(len,1);
proc = cell(len,1);
ori = im2double(imread(['deal_images/' data{1}]));

for i = 1:len
	name = data{sel(i)};
	display(sprintf('image %d: %s...', sel(i), name))
	im{i} = im2double(imread(['deal_images/' name]));

	s = size(im{i});
	ori_r = imresize(ori, s([1 2]));
	display(sprintf('l2 distortion: %.2f', norm(im{i}(:)-ori_r(:))))

	proc{i} = lnorm(im{i},3,3,type);
end

mx = zeros(len,1);
mn = zeros(len,1);
for i = 1:len
	mx(i) = max(proc{i}(:));
	mn(i) = min(proc{i}(:));
end
mx = max(mx);
mn = min(mn);

close all
gnorm = @(x) (x - mn) ./ (mx - mn);
for i = 1:len
	name = data{sel(i)};
	figure(i), imshow(gnorm(proc{i}))
	imwrite(gnorm(proc{i}), ['output/' type '3-' name]);
end
