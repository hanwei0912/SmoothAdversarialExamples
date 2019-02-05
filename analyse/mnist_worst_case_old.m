path_name='/nfs/pyrex/raid6/hzhang/SmoothPerturbation/X_test.mat';
cw_path = '/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_BasicCnn_cw_1.0_15.0_adv_x.mat';
clip_path = '/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_BasicCnn_clipl2_1.0_15.0_adv_x.mat';
% fgsm_path = '/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_BasicCnn_fgsm_0.1_adv_x.mat';
% bim_path = '/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_BasicCnn_bim_0.1_0.08_adv_x.mat';
fgsm_path = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/mnist_BasicCnn_bim_2.25_3.0_adv_x.mat';
bim_path = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/mnist_BasicCnn_sbim_6_3_adv_x.mat';

% cw_path = '/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_advTrain_cw_adv_x.mat';
% clip_path = '/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_advTrain_clipl2_adv_x.mat';
% % fgsm_path = '/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_advTrain_fgsm_adv_x.mat';
% % bim_path = '/nfs/pyrex/raid6/hzhang/SmoothPerturbation/mnist_advTrain_bim_adv_x.mat';
% fgsm_path = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/mnist_AdvTrain_fgsm_0.3_adv_x.mat';
% bim_path = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/mnist_AdvTrain_bim_0.3_0.08_adv_x.mat';


ori = load(path_name);ori=ori.X_test;
cw = load(cw_path);cw=cw.adv_x;
clip = load(clip_path);clip=clip.adv_x;
fgsm = load(fgsm_path);fgsm=fgsm.adv_x;
bim = load(bim_path);bim=bim.adv_x;

%%
% cw_p = cw-ori;
% clip_p=clip-ori;
% fgsm_p=fgsm-ori;
% bim_p=bim-ori;
% cw_ind=0;cw_max=0;
% clip_ind=0;clip_max=0;
% fgsm_ind=0;fgsm_max=0;
% bim_ind=0;bim_max=0;
% for i=1:10000
%     if norm(squeeze(cw_p(i,:,:)),2)>cw_max
%         cw_max=norm(squeeze(cw_p(i,:,:)),2);
%         cw_ind = i;
%     end
%     if norm(squeeze(clip_p(i,:,:)),2)>clip_max
%         clip_max=norm(squeeze(clip_p(i,:,:)),2);
%         clip_ind = i;
%     end
%     if norm(squeeze(fgsm_p(i,:,:)),2)>fgsm_max
%         fgsm_max=norm(squeeze(fgsm_p(i,:,:)),2);
%         fgsm_ind = i;
%     end
%     if norm(squeeze(bim_p(i,:,:)),2)>bim_max
%         bim_max=norm(squeeze(bim_p(i,:,:)),2);
%         bim_ind = i;
%     end
% end
% %%
cw_d=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/mnist_BasicCnn_cw_1.0_15.0.mat');
clip_d=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/mnist_BasicCnn_clipl2_1.0_15.0.mat');
% fgsm_d=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/mnist_BasicCnn_fgsm_0.1.mat');
% bim_d=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/mnist_BasicCnn_bim_0.1_0.08.mat');

fgsm_d = load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/whole_image/mnist_BasicCnn_bim_6_3.mat');
bim_d = load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/mnist/whole_image/mnist_BasicCnn_sbim_6_3.mat');


% cw_d=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/mnist_advTrain_cw.mat');
% clip_d=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/mnist_advTrain_clipl2.mat');
% % fgsm_d=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/mnist_advTrain_fgsm.mat');
% % bim_d=load('/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/mnist_advTrain_bim.mat');
% 
% fgsm_d = load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/test/mnist/adv_mnist_BasicCnn_fgsm_0.3.mat');
% bim_d = load('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/test/mnist/adv_mnist_BasicCnn_bim_0.3_0.08.mat');


cw_l2=cw_d.l2;
cw_l2(cw_d.p==1)=0;

clip_l2=clip_d.l2;
clip_l2(clip_d.p==1)=0;

fgsm_l2=fgsm_d.l2;
fgsm_l2(fgsm_d.p==1)=0;

bim_l2=bim_d.l2;
bim_l2(bim_d.p==1)=0;
[~,cw_ind]=max(cw_l2);
cw_l2(cw_ind)=0;
[~,cw_ind]=max(cw_l2);
[~,clip_ind]=max(clip_l2);
[~,fgsm_ind]=max(fgsm_l2);
[~,bim_ind]=max(bim_l2);


figure,
subplot(4,4,1),
imshow(squeeze(cw(cw_ind,:,:)));
imwrite(squeeze(cw(cw_ind,:,:)),sprintf('data/fig/mnistWcwMcw.png'));
if cw_d.p(cw_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('CW-worst case\n(l2=%f,%s)',cw_d.l2(cw_ind),str_t));
subplot(4,4,2),
imshow(squeeze(clip(cw_ind,:,:)));
imwrite(squeeze(clip(cw_ind,:,:)),sprintf('data/fig/mnistWcwMclip.png'));
if clip_d.p(cw_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('our\n(l2=%f,%s)',clip_d.l2(cw_ind),str_t));
subplot(4,4,3),
imshow(squeeze(fgsm(cw_ind,:,:)));
imwrite(squeeze(fgsm(cw_ind,:,:)),sprintf('data/fig/mnistWcwMfgsm.png'));
if fgsm_d.p(cw_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('FGSM\n(l2=%f,%s)',fgsm_d.l2(cw_ind),str_t));
subplot(4,4,4),
imshow(squeeze(bim(cw_ind,:,:)));
imwrite(squeeze(bim(cw_ind,:,:)),sprintf('data/fig/mnistWcwMbim.png'));
if bim_d.p(cw_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('I-FGSM\n(l2=%f,%s)',bim_d.l2(cw_ind),str_t));

subplot(4,4,5),
imshow(squeeze(cw(clip_ind,:,:)));
imwrite(squeeze(cw(clip_ind,:,:)),sprintf('data/fig/mnistWclipMcw.png'));
if cw_d.p(clip_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('CW\n(l2=%f,%s)',cw_d.l2(clip_ind),str_t));
subplot(4,4,6),
imshow(squeeze(clip(clip_ind,:,:)));
imwrite(squeeze(clip(clip_ind,:,:)),sprintf('data/fig/mnistWclipMclip.png'));
if clip_d.p(clip_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('our-worst case\n(l2=%f,%s)',clip_d.l2(clip_ind),str_t));
subplot(4,4,7),
imshow(squeeze(fgsm(clip_ind,:,:)));
imwrite(squeeze(fgsm(clip_ind,:,:)),sprintf('data/fig/mnistWclipMfgsm.png'));
if fgsm_d.p(clip_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('FGSM\n(l2=%f,%s)',fgsm_d.l2(clip_ind),str_t));
subplot(4,4,8),
imshow(squeeze(bim(clip_ind,:,:)));
imwrite(squeeze(bim(clip_ind,:,:)),sprintf('data/fig/mnistWclipMbim.png'));
if bim_d.p(clip_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('I-FGSM\n(l2=%f,%s)',bim_d.l2(clip_ind),str_t));

subplot(4,4,9),
imshow(squeeze(cw(fgsm_ind,:,:)));
imwrite(squeeze(cw(fgsm_ind,:,:)),sprintf('data/fig/mnistWfgsmMcw.png'));
if cw_d.p(fgsm_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('CW\n(l2=%f,%s)',cw_d.l2(fgsm_ind),str_t));
subplot(4,4,10),
imshow(squeeze(clip(fgsm_ind,:,:)));
imwrite(squeeze(clip(fgsm_ind,:,:)),sprintf('data/fig/mnistWfgsmMclip.png'));
if clip_d.p(fgsm_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('our\n(l2=%f,%s)',clip_d.l2(fgsm_ind),str_t));
subplot(4,4,11),
imshow(squeeze(fgsm(fgsm_ind,:,:)));
imwrite(squeeze(fgsm(fgsm_ind,:,:)),sprintf('data/fig/mnistWfgsmMfgsm.png'));
if fgsm_d.p(fgsm_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('FGSM-worst case\n(l2=%f,%s)',fgsm_d.l2(fgsm_ind),str_t));
subplot(4,4,12),
imshow(squeeze(bim(fgsm_ind,:,:)));
imwrite(squeeze(bim(fgsm_ind,:,:)),sprintf('data/fig/mnistWfgsmMbim.png'));
if bim_d.p(fgsm_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('I-FGSM\n(l2=%f,%s)',bim_d.l2(fgsm_ind),str_t));

subplot(4,4,13),
imshow(squeeze(cw(bim_ind,:,:)));
imwrite(squeeze(cw(bim_ind,:,:)),sprintf('data/fig/mnistWbimMcw.png'));
if cw_d.p(bim_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('CW\n(l2=%f,%s)',cw_d.l2(bim_ind),str_t));
subplot(4,4,14),
imshow(squeeze(clip(bim_ind,:,:)));
imwrite(squeeze(clip(bim_ind,:,:)),sprintf('data/fig/mnistWbimMclip.png'));
if clip_d.p(bim_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('our\n(l2=%f,%s)',clip_d.l2(bim_ind),str_t));
subplot(4,4,15),
imshow(squeeze(fgsm(bim_ind,:,:)));
imwrite(squeeze(fgsm(bim_ind,:,:)),sprintf('data/fig/mnistWbimMfgsm.png'));
if fgsm_d.p(bim_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('FGSM\n(l2=%f,%s)',fgsm_d.l2(bim_ind),str_t));
subplot(4,4,16),
imshow(squeeze(bim(bim_ind,:,:)));
imwrite(squeeze(bim(bim_ind,:,:)),sprintf('data/fig/mnistWbimMbim.png'));
if bim_d.p(bim_ind)==0
    str_t='suc';
else
    str_t='fail';
end
title(sprintf('I-FGSM-worst case\n(l2=%f,%s)',bim_d.l2(bim_ind),str_t));

suptitle('MNIST: Worst case');