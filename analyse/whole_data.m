% cw=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/1g_cw_0.1_p_l2.mat');
% clip=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/1g_clipl2_0.1_p_l2.mat');
%  cw=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/res_cw_0.1_p_l2.mat');
 % clip=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/res_clip_0.1_p_l2.mat');
%  fgsm=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/res_fgsm_0.1_p_l2.mat');
%  bim=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/res_bim_0.1_p_l2.mat');
cw=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/cw_0.1_p_l2.mat');
%% % clip=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/clip_0.1_p_l2.mat');
clip = load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/improve/inc_0.1_p_l2.mat');
fgsm=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/fgsm_0.01_p_l2.mat');
bim=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/bim_0.01_p_l2.mat');
% bim=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/T_bim_0.1_p_l2.mat'); 
% cw=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/transf_cw_inc_p_l2.mat');
% clip=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/transf_clip_inc_p_l2.mat');
% fgsm=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/transf_fgsm_inc_p_l2.mat');
% bim=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/transf_bim_inc_p_l2.mat');
% cw=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/adv_cw_inc_p_l2.mat');
% clip=load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/improve/adv_inc_0.1_p_l2.mat');
% fgsm=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/adv_fgsm_inc_p_l2.mat');
% bim=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/adv_bim_inc_p_l2.mat');
% cw=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/10-0.2b_cw_0.1_p_l2.mat');
% clip=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/10-0.2b_clipl2_0.1_p_l2.mat');
% fgsm=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/10-0.2b_fgsm_0.1_p_l2.mat');
% bim=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/10-0.2b_bim_0.1_p_l2.mat');
% clip = load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/improve/res_inc_0.1_p_l2.mat');
% bim = load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/improve/smoothBiml2_inc_0.1_p_l2.mat');
% fgsm=load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/whole_data/0.50.2b_fgsm_inc_p_l2.mat');
% bim=load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/whole_data/0.50.2b_bim_inc_p_l2.mat');
% cw=load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/whole_data/0.50.2b_cw_inc_p_l2.mat');
% clip=load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/whole_data/0.50.2b_sbim_inc_p_l2.mat');
% %
% cw_m = zeros(1000,2);
% cw_m(:,1)=cw.l2;
% cw_m(:,2)=1-cw.p;
% cw_m(cw.ori_a==0,1) = 0;
% cw_m(cw.ori_a==0,2) = 1;
% % cw_m(cw_m(:,2)==0,1) = 0;
% % cw_m(cw_m(:,2)==0,1) = cw.l2_worst(cw_m(:,2)==0);
d_total_num = sum(cw.ori_a);
cw_m = zeros(d_total_num,2);
cw_m(:,1)=cw.l2(cw.ori_a==1);
cw_m(:,2)=1-cw.p(cw.ori_a==1);
%%
cw_m=sortrows(cw_m);
cw_c=zeros(d_total_num,2);
cw_c(:,1)=cw_m(:,1);
cw_c(:,2)=cumsum(cw_m(:,2))/d_total_num;
% save('./data/cw_raw.mat','cw_m');
% csvwrite('./data/cw-imagenet.csv',cw_c)
% cw_s = zeros(101,2);
% cw_s(1:100,:) = cw_c(1:10:1000,:);
% cw_s(101,:)= cw_c(1000,:);
% dlmwrite('./data/cwImageNetAdv.txt',cw_s,'delimiter',' ')
% dlmwrite('./data/cwImageNetResV2.txt',cw_s,'delimiter',' ')
% dlmwrite('./data/cwImageNet.txt',cw_c,'delimiter',' ')

clip_m = zeros(d_total_num,2);
clip_m(:,1)=clip.l2(clip.ori_a==1);
clip_m(:,2)=1-clip.p(clip.ori_a==1);
%clip_m(clip.ori_a==0,1) = 0;
%clip_m(clip.ori_a==0,2) = 1;
% clip_m(clip_m(:,2)==0,1) = 0;
% clip_m(clip_m(:,2)==0,1) = clip.l2_worst(clip_m(:,2)==0);
clip_m=sortrows(clip_m);
clip_c=zeros(d_total_num,2);
clip_c(:,1)=clip_m(:,1);
clip_c(:,2)=cumsum(clip_m(:,2))/(d_total_num*1.0);
% save('./data/clip_raw.mat','clip_m');
% csvwrite('./data/clip-imagenet.csv',clip_c)
%clip_s = zeros(101,2);
%clip_s(1:100,:) = clip_c(1:10:num,:);
%clip_s(101,:)= clip_c(num-1,:);
% dlmwrite('./data/clipImageNetO.txt',clip_s,'delimiter',' ')
% dlmwrite('./data/clipImageNet.txt',clip_c,'delimiter',' ')
% dlmwrite('./data/clipImageNetAdv.txt',clip_s,'delimiter',' ')
acc=sum(clip.ori_a)/1000.0;

%%
fgsm_m = zeros(1000,2);
fgsm_m(:,1)=fgsm.l2;
fgsm_m(:,2)=1-fgsm.p;
fgsm_m(fgsm.ori_a==0,1) = 0;
fgsm_m(fgsm.ori_a==0,2) = 1;
% fgsm_m(fgsm_m(:,2)==0,1) = 0;
% fgsm_m(fgsm_m(:,2)==0,1) = cw.l2_worst(fgsm_m(:,2)==0);
fgsm_m=sortrows(fgsm_m);
fgsm_c=zeros(1000,2);
fgsm_c(:,1)=fgsm_m(:,1);
fgsm_c(:,2)=cumsum(fgsm_m(:,2))/1000.0;
% csvwrite('./data/fgsm-imagenet.csv',fgsm_c)
fgsm_s = zeros(101,2);
fgsm_s(1:100,:) = fgsm_c(1:10:1000,:);
fgsm_s(101,:)= fgsm_c(1000,:);
% dlmwrite('./data/fgsmImageNetResV2.txt',fgsm_s,'delimiter',' ')
dlmwrite('./data/fgsmImageNet.txt',fgsm_s,'delimiter',' ')

% save('./data/fgsm_raw.mat','fgsm_m');

bim_m = zeros(1000,2);
bim_m(:,1)=bim.l2;
bim_m(:,2)=1-bim.p;
bim_m(bim.ori_a==0,1) = 0;
bim_m(bim.ori_a==0,2) = 1;
% bim_m(bim_m(:,2)==0,1) = 0;
% bim_m(bim_m(:,2)==0,1) = cw.l2_worst(bim_m(:,2)==0);
bim_m=sortrows(bim_m);
bim_c=zeros(1000,2);
bim_c(:,1)=bim_m(:,1);
bim_c(:,2)=cumsum(bim_m(:,2))/1000.0;
% csvwrite('./data/bim-imagenet.csv',bim_c)
bim_s = zeros(101,2);
bim_s(1:100,:) = bim_c(1:10:1000,:);
bim_s(101,:)= bim_c(1000,:);
% dlmwrite('./data/bimImageNet.txt',bim_s,'delimiter',' ')
% dlmwrite('./data/bimImageNetAdv.txt',bim_s,'delimiter',' ')
dlmwrite('./data/bimImageNet.txt',bim_s,'delimiter',' ')

% save('./data/ifgsm_raw.mat','bim_m');

%%
plot(cw_c(:,1),cw_c(:,2),'r',clip_c(:,1),clip_c(:,2),'b',fgsm_s(:,1),fgsm_s(:,2),'m',bim_s(:,1),bim_s(:,2),'k')
% plot(cw_s(:,1),cw_s(:,2),'r',fgsm_s(:,1),fgsm_s(:,2),'m',bim_s(:,1),bim_s(:,2),'k')
xlabel('l2 distortion');
ylabel('success probability');
cw_le=sprintf('CarliniWagner(suc=%f)',cw_c(d_total_num,2));
clip_le=sprintf('Our method(suc=%f)',clip_c(d_total_num,2));
fgsm_le=sprintf('FGSM(suc=%f)',fgsm_c(1000,2));
bim_le=sprintf('I-FGSM(suc=%f)',bim_c(1000,2));
legend({cw_le,clip_le,fgsm_le,bim_le});
title(sprintf('1000 imagenet(acc=%f)',acc));

% %%
% cw_m = zeros(1000,2);
% cw_m(:,1)=cw.l2;
% cw_m(:,2)=1-cw.c;
% cw_m(cw_m(:,2)==0,1) = 0;
% cw_m=sortrows(cw_m);
% cw_c=zeros(1000,2);
% cw_c(:,1)=cw_m(:,1);
% cw_c(:,2)=cumsum(cw_m(:,2))/1000.0;
% 
% 
% clip_m = zeros(1000,2);
% clip_m(:,1)=clip.l2;
% clip_m(:,2)=1-clip.c;
% clip_m(clip_m(:,2)==0,1) = 0;
% clip_m=sortrows(clip_m);
% clip_c=zeros(1000,2);
% clip_c(:,1)=clip_m(:,1);
% clip_c(:,2)=cumsum(clip_m(:,2))/1000.0;
% 
% acc=sum(clip.ori_a)/1000.0;
% 
% %%
% fgsm_m = zeros(1000,2);
% fgsm_m(:,1)=fgsm.l2;
% fgsm_m(:,2)=1-fgsm.c;
% fgsm_m(fgsm_m(:,2)==0,1) = 0;
% fgsm_m=sortrows(fgsm_m);
% fgsm_c=zeros(1000,2);
% fgsm_c(:,1)=fgsm_m(:,1);
% fgsm_c(:,2)=cumsum(fgsm_m(:,2))/1000.0;
% 
% bim_m = zeros(1000,2);
% bim_m(:,1)=bim.l2;
% bim_m(:,2)=1-bim.c;
% bim_m(bim_m(:,2)==0,1) = 0;
% bim_m=sortrows(bim_m);
% bim_c=zeros(1000,2);
% bim_c(:,1)=bim_m(:,1);
% bim_c(:,2)=cumsum(bim_m(:,2))/1000.0;
% 
% figure
% plot(cw_c(:,1),cw_c(:,2),'r',clip_c(:,1),clip_c(:,2),'b',fgsm_c(:,1),fgsm_c(:,2),'m',bim_c(:,1),bim_c(:,2),'k')
% xlabel('l2 distortion');
% ylabel('change rate');
% cw_le=sprintf('CarliniWagner(cha=%f)',cw_c(1000,2));
% clip_le=sprintf('Our method(cha=%f)',clip_c(1000,2));
% fgsm_le=sprintf('FGSM(cha=%f)',fgsm_c(1000,2));
% bim_le=sprintf('I-FGSM(cha=%f)',bim_c(1000,2));
% legend({cw_le,clip_le,fgsm_le,bim_le});
% title(sprintf('1000 imagenet(acc=%f)',acc));


% subplot(1,2,1),
% plot(cw_c(:,1),cw_c(:,2))
% xlabel('l2 distortion');
% ylabel('success probability');
% title(sprintf('CarliniWagner(suc=%f)',cw_c(1000,2)));
% subplot(1,2,2)
% plot(clip_c(:,1),clip_c(:,2))
% xlabel('l2 distortion');
% ylabel('success probability');
% title(sprintf('Our method(suc=%f)',clip_c(1000,2)));
% suptitle(sprintf('1000 imagenet(acc=%f)',acc));