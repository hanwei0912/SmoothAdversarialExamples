path_name = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/inceptionV3/BIM/l2_1/';
ori_path = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/images/';

pic_list=dir(path_name);
p_n=size(pic_list);
mad = zeros(1000,1);
hi = zeros(1000,1);
lo = zeros(1000,1);
for i=3:p_n
    name=pic_list(i).name;
    
    ori_path_name=sprintf('%s%s',ori_path,name);
    adv_path_name=sprintf('%s%s',path_name,name);
    ori = imread(ori_path_name);
    adv = imread(adv_path_name);
    [I Map] = MAD_index_april_2010( ori , adv, 16 );
    mad(i-2)=I.MAD;
    lo(i-2)=I.LO;
    hi(i-2)=I.HI;
end

save('./madpgd_1_16.mat','mad')

path_name = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/inceptionV3/BIM/l2_2/';

pic_list=dir(path_name);
p_n=size(pic_list);
mad = zeros(1000,1);
hi = zeros(1000,1);
lo = zeros(1000,1);
for i=3:p_n
    name=pic_list(i).name;
    
    ori_path_name=sprintf('%s%s',ori_path,name);
    adv_path_name=sprintf('%s%s',path_name,name);
    ori = imread(ori_path_name);
    adv = imread(adv_path_name);
    [I Map] = MAD_index_april_2010( ori , adv, 16 );
    mad(i-2)=I.MAD;
    lo(i-2)=I.LO;
    hi(i-2)=I.HI;
end

save('./madpgd_2_16.mat','mad')

path_name = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/inceptionV3/BIM/l2_3/';

pic_list=dir(path_name);
p_n=size(pic_list);
mad = zeros(1000,1);
hi = zeros(1000,1);
lo = zeros(1000,1);
for i=3:p_n
    name=pic_list(i).name;
    
    ori_path_name=sprintf('%s%s',ori_path,name);
    adv_path_name=sprintf('%s%s',path_name,name);
    ori = imread(ori_path_name);
    adv = imread(adv_path_name);
    [I Map] = MAD_index_april_2010( ori , adv, 16 );
    mad(i-2)=I.MAD;
    lo(i-2)=I.LO;
    hi(i-2)=I.HI;
end

save('./madpgd_3_16.mat','mad')

path_name = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/inceptionV3/BIM/l2_4/';

pic_list=dir(path_name);
p_n=size(pic_list);
mad = zeros(1000,1);
hi = zeros(1000,1);
lo = zeros(1000,1);
for i=3:p_n
    name=pic_list(i).name;
    
    ori_path_name=sprintf('%s%s',ori_path,name);
    adv_path_name=sprintf('%s%s',path_name,name);
    ori = imread(ori_path_name);
    adv = imread(adv_path_name);
    [I Map] = MAD_index_april_2010( ori , adv, 16 );
    mad(i-2)=I.MAD;
    lo(i-2)=I.LO;
    hi(i-2)=I.HI;
end

save('./madpgd_4_16.mat','mad')

path_name = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/inceptionV3/BIM/l2_5/';

pic_list=dir(path_name);
p_n=size(pic_list);
mad = zeros(1000,1);
hi = zeros(1000,1);
lo = zeros(1000,1);
for i=3:p_n
    name=pic_list(i).name;
    
    ori_path_name=sprintf('%s%s',ori_path,name);
    adv_path_name=sprintf('%s%s',path_name,name);
    ori = imread(ori_path_name);
    adv = imread(adv_path_name);
    [I Map] = MAD_index_april_2010( ori , adv, 16 );
    mad(i-2)=I.MAD;
    lo(i-2)=I.LO;
    hi(i-2)=I.HI;
end

save('./madpgd_5_16.mat','mad')