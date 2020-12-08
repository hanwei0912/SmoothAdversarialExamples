function show_perturbation_color()
ori_path = '/nfs/pyrex/raid6/hzhang/2017-nips/images/';
path_name = '/nfs/pyrex/raid6/hzhang/2017-nips/test/panda/';


pic_list=dir(path_name);
p_n=size(pic_list);
for i=3:p_n
    name=pic_list(i).name;
    
    ori_path_name=sprintf('%s%s',ori_path,name);
    adv_path_name=sprintf('%s%s',path_name,name);
    ori=im2double(imread(ori_path_name));
    adv=im2double(imread(adv_path_name));
    adv_p = adv-ori;
    [u_adv,l_adv]= max_p(adv_p);
    adv_np = reshape_img(adv_p,u_adv,l_adv);
    
    subplot(1,3,1)
    imshow(ori,[]);
    title('ori');
    subplot(1,3,2)
    imshow(adv,[])
    title('adv');
    subplot(1,3,3)
    imshow(adv_np,[]);
    title('perturb')
%     imwrite(adv,sprintf('image/c_p_sfs_%s'))
    imwrite(adv_np,sprintf('/udd/hzhang/moothAdversarialExamples/analyse/data/np-%s',name));
end
end




function [u_p,l_p]=max_p(p)
u_p=zeros(3,1);
l_p=zeros(3,1);
for i=1:3
    t=p(:,:,i);
    u_p(i)=max(t(:));
    l_p(i)=min(t(:));
end
end

function c_i=search_id(clip_d,cw_name_m)
c_i=0;
idx=all(ismember(clip_d.name,cw_name_m),2);
lidx=find(idx);
[num,~]=size(lidx);
for h=1:num
    if strcmp(clip_d.name(lidx(h),:),cw_name_m)==1
        c_i=lidx(h);
    end
end
end


function [img1,img2] = reshape_img(im1,u1,l1)
    img1=im1;
    de=zeros(3,1);
    for i=1:3
        de(i)=u1(i)-l1(i);
        img1(:,:,i)=(im1(:,:,i)-l1(i))/de(i);
    end
    
end
