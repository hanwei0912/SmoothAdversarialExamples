ori_path = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/images/';
%path_name = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/inceptionV3/0.1/';
path_name = '/nfs/pyrex/raid6/hzhang/2017-nips/cw/learning_rate/0.1/';

pic_list=dir(path_name);
p_n=size(pic_list);
mad = zeros(1000,1);
for i=3:p_n
    name=pic_list(i).name;
    
    ori_path_name=sprintf('%s%s',ori_path,name);
    adv_path_name=sprintf('%s%s',path_name,name);
    ori = imread(ori_path_name);
    adv = imread(adv_path_name);
    [I Map] = MAD_index_april_2010( ori , adv );
    mad(i-2)=I.MAD;
end

save('./madcw.mat','mad')