function imagenet_test()

x_path=sprintf('/nfs/pyrex/raid6/hzhang/2017-nips/images/7f045354f4577c16.png');
ori=imread(x_path);
[m1,m2,m3]=size(ori);
m=m1*m2;
adv_x_path=sprintf('/nfs/pyrex/raid6/hzhang/2017-nips/fgsm/images/7f045354f4577c16.png');
adv=imread(adv_x_path);
adv = im2double(adv);
ori = im2double(ori);
adv_c= reshape(adv-ori,[m,m3]);

namuda=20;
alpha=0.996;
iter=50;

% Aa=zeros(4,m,3);

% for dim_i=1:3
%     [image_feature,image_index]=similarity(ori(:,:,dim_i),m,namuda);
% %%
% %     [n,~,~]=size(image_feature);
%     A_ = knngraph(image_index', image_feature');
%     S=transition_matrix(A_);
%     A=speye(size(S))-alpha*S;
%     HA= graph_matrix(A,m,image_index,299,4);
%     Aa(:,:,dim_i) =HA';
% end
A=load('../dataset/A/7f045354f4577c16.png_300_0.997.mat');
Aa = A.A;
Aa=reshape(Aa,[4,m,3]);

X=cg_li(Aa,adv_c,1e-6,iter);
div_z = cg_li(Aa,ones(m,m3),1e-6,iter);
n_x=X./div_z;

% x = X + abs(min(X(:)));
% s_x=max(x(:));
% n_x=x./s_x;
im_x=reshape(n_x,[299,299,3]);
figure,
subplot(1,3,1)
imshow(ori,[])
title('ori image');
subplot(1,3,2)
imshow(adv,[])
title('fgsm')
subplot(1,3,3)
% imshow(im_x,[])
imshow(im_x+ori,[]);
% saveas('image.png',im_x)
title(sprintf('lamb=%f,alpha=%f',namuda,alpha));
% imwrite(im_x+ori,sprintf('smooth/%f_%f_%s_S%d.png',namuda,alpha,'641d03527848417f.png',iter));

end

function x = cg_li(f, b, tol, iter, term)
% conjugate gradient, book version

if nargin < 3, tol = 1e-6; end
if nargin < 4, iter = 20; end
if nargin < 5, term = @(varargin) false; end

if ~isa(f, 'function_handle'), f = @(x) smooth(f,x); end
tol = tol ^ 2;


x = ones(size(b));
r = b - f(x);
p = r;
s = r' * r;
d = b' * b;
[n_s,~] = size(b);

for i = 1:iter

	q = f(p);
	alpha = diag(s ./ (p' * q));
	x = x + repmat(alpha',[n_s,1]) .* p;
	r = r - repmat(alpha',[n_s,1]) .* q;
	t = r' * r;
	beta = diag(t ./ s);
	p = r + repmat(beta',[n_s,1]) .* p;
	s = t;

% 	if term(i, s, x, r, p) || s < tol * d, break, end
end
end

function ss=smooth(HA,adv_c)

C=padarray(reshape(adv_c,[299,299,3]),[1,1]);
c1=C(1:299,2:300,:);
c2=C(3:301,2:300,:);
c3=C(2:300,1:299,:);
c4=C(2:300,3:301,:);
img_4=zeros(4,299*299,3);
img_4(1,:,:)=reshape(c1,[1,299*299,3]);
img_4(2,:,:)=reshape(c2,[1,299*299,3]);
img_4(3,:,:)=reshape(c3,[1,299*299,3]);
img_4(4,:,:)=reshape(c4,[1,299*299,3]);


smo=HA.*img_4;
ss=shiftdim(sum(smo,1))+adv_c;

end
