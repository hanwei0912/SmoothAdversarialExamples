%%

% 
at =load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/whole_data/fgsm_inc_%d_p_l2.mat',1));
d_num = sum(at.ori_a);

data = zeros(5,d_num);

% data(1,:)=at.l2(at.ori_a==1);
% suc = at.p(at.ori_a==1);
% data(1,suc==1)=99999;

% at =   load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/whole_image/mnist_AdvTrain_sbim_2_3.mat');
% data(2,:)=at.l2(at.ori_a==1);
% suc = at.p(at.ori_a==1);
% data(2,suc==1)=99999;
% at = load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/whole_image/mnist_AdvTrain_sbim_4_3.mat');
% data(3,:)=at.l2(at.ori_a==1);
% suc = at.p(at.ori_a==1);
% data(3,suc==1)=99999;
% at =  load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/whole_image/mnist_AdvTrain_sbim_6_3.mat');
% data(4,:)=at.l2(at.ori_a==1);
% suc = at.p(at.ori_a==1);
% data(4,suc==1)=99999;
% at =  load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/whole_image/mnist_AdvTrain_bim_0.1_0.08.mat');
% data(5,:)=at.l2(at.ori_a==1);
% suc = at.p(at.ori_a==1);
% data(5,suc==1)=99999;
% at =  load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/whole_image/mnist_AdvTrain_bim_1.0_3.0.mat');
% data(6,:)=at.l2(at.ori_a==1);
% suc = at.p(at.ori_a==1);
% data(6,suc==1)=99999;


for i=1:5
at = load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/whole_data/fgsm_inc_%d_p_l2.mat',i));
data(i,:)=at.l2(at.ori_a==1);
suc = at.p(at.ori_a==1);
data(i,suc==1)=99999;
end
%%

mindis=min(data);

vm=ones(d_num,2);
%%
vm(:,1)=mindis;
vm(mindis==99999,1)=0;
vm(mindis==99999,2)=0;
% vm(at.ori_a==0,1)=0;
% vm(at.ori_a==0,2)=0;

%%

vm_c =sortrows(vm);
vm_c(:,2)=cumsum(vm_c(:,2))/d_num;

p_suc = sum(vm(:,2))/d_num
c_l2  = mean(vm(vm(:,2)==1,1))

plot(vm_c(:,1),vm_c(:,2))
dlmwrite('./data/fgsmImageNet.txt',vm_c,'delimiter',' ')