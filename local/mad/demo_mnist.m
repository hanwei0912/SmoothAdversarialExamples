x=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/X_test.mat');
x=x.X_test;
adv = load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_BasicCnn_cw_1.0_15.0_adv_x.mat');
adv = adv.adv_x;
[n,m]=size(x);
mad = zeros(n,1);
hi = zeros(n,1);
lo = zeros(n,1);
for i= 1:n
    x_img=int64(vec2mat(x(i,:),28)*255);
    a_img=int64(vec2mat(adv(i,:),28)*255);
    [I Map] = MAD_index_april_2010( x_img , a_img,2 );
    mad(i)=I.MAD;
    lo(i)=I.LO;
    hi(i)=I.HI;
end
save('./madcw_mnist.mat','mad','hi','lo')

x=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/X_test.mat');
x=x.X_test;
adv = load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_BasicCnn_clipl2_1.0_15.0_adv_x.mat');
adv = adv.adv_x;
[n,m]=size(x);
mad = zeros(n,1);
hi = zeros(n,1);
lo = zeros(n,1);
for i= 1:n
    x_img=int64(vec2mat(x(i,:),28)*255);
    a_img=int64(vec2mat(adv(i,:),28)*255);
    [I Map] = MAD_index_april_2010( x_img , a_img,2 );
    mad(i)=I.MAD;
    lo(i)=I.LO;
    hi(i)=I.HI;
end
save('./madscw_mnist.mat','mad','hi','lo')