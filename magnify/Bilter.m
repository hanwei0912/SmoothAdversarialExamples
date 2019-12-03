path_name='/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/new/inception/scw/0.01';
% path_name='/nfs/pyrex/raid6/hzhang/2017-nips/test/';
pic_list=dir(path_name);
p_n=size(pic_list);

save_path = sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/new/inception/scw/b0.01');
for i=3:p_n
    name=pic_list(i).name;
    ori_path=sprintf('%s/%s',path_name,name);
    ori=double(imread(ori_path));

    image=ori/255;
    Rchannel=image(:,:,1);
    Gchannel=image(:,:,2);
    Bchannel=image(:,:,3);
    w=5; % window size
    sigma_d=0.5;
    sigma_r=.2;
    tic;
    [Rout]=bilateral_each_channel(w,sigma_r,sigma_d,Rchannel);
    [Gout]=bilateral_each_channel(w,sigma_r,sigma_d,Gchannel);
    [Bout]=bilateral_each_channel(w,sigma_r,sigma_d,Bchannel);
    image(:,:,1)=(Rout);
    image(:,:,2)=(Gout);
    image(:,:,3)=(Bout);
%         subplot(1,2,1)
%         imshow(ori);
%         subplot(1,2,2)
%         imshow(simg);
%         pause(1);
    smo_path = sprintf('%s/%s',save_path,name);
    imwrite(image,smo_path);

end