x=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/X_test.mat');
x=x.X_test;
para_list = {'1.0','1.5','1.75','2.25','3.0','5.0'};
for k=1:length(para_list)
    adv = load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/mnist_BasicCnn_bim_%s_3.0_adv_x.mat',para_list{k}));
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
    save(sprintf('./madpgd_mnist_%s.mat',para_list{k}),'mad','hi','lo')
end