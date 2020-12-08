based_path = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/whole_image/mnist_BasicCnn_bim_';
name = {'1.0_3.0','1.5_3.0','1.75_3.0','2_3','2.25_3.0','2.5_3.0','3.0_3.0','4_3','5.0_3.0','6_3'};
% name = {'1_3','1.5_3','1.75_3','2_3','3.25_3','2.5_3','3_3','4_3','5_3','6_3'};

num = length(name);
L2 = zeros(num,10000);
P  = zeros(num,10000); 

for i=1:length(name)
    path_name = sprintf('%s%s.mat',based_path,name{i});
    data=load(path_name);
    l2 = data.l2;
    p=data.p;
    L2(i,:)=l2;
    P(i,:)=p;
end

ind = find(P);
L2(ind)=99999;

ll2 = min(L2);
ind =find(ll2==99999);
ll2(ind)=0;
[Y,U] = max(ll2);



