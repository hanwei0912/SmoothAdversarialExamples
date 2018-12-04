function imgnet_generate()
path_name='/nfs/pyrex/raid6/hzhang/2017-nips/images/';
% path_name='/nfs/pyrex/raid6/hzhang/2017-nips/test/';
pic_list=dir(path_name);
p_n=size(pic_list);
for i=3:p_n
    name=pic_list(i).name;
    name = '7f045354f4577c16.png';
    ori_path=sprintf('%s%s',path_name,name);
    ori=imread(ori_path);
    ori=im2double(ori);
    [m1,m2,~]=size(ori);
    m=m1*m2;

    namuda=300;
    alpha=0.997;
    HA = zeros(4,m,3);
    for dim_i=1:3
        [image_feature,image_index]=similarity(ori(:,:,dim_i),m,namuda);
        %%  
        A_ = knngraph(image_index', image_feature');
        S=transition_matrix(A_);
        Aa=speye(size(S))-alpha*S;
        A= graph_matrix(Aa,m,image_index,299);
        HA(:,:,dim_i)=A';
    end
    A=reshape(HA,[4,299,299,3]);
    data_str=sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/A/SpA_%s_%f_%f.mat',name,namuda,alpha);
    save(data_str,'A');
end
end


function [ image_feature, image_index ] = similarity( x_img, m, namuda )
% Construct the similarity matrix of the image

image_feature = zeros(m,4);
image_index = zeros(m+1,4);
for i=1:299
    for j=1:299
        if i== 1
            image_feature((i-1)*299+j,1)=0;
            image_index((i-1)*299+j,1)=299*299+1;
        else
            image_feature((i-1)*299+j,1)=exp(-namuda*(x_img(i-1,j)-x_img(i,j)).^2);
            image_index((i-1)*299+j,1)=(i-2)*299+j;
        end
        if i== 299
            image_feature((i-1)*299+j,2)=0;
            image_index((i-1)*299+j,2)=299*299+1;
        else
            image_feature((i-1)*299+j,2)=exp(-namuda*(x_img(i+1,j)-x_img(i,j)).^2);
            image_index((i-1)*299+j,2)=i*299+j;
        end
        if j== 1
            image_feature((i-1)*299+j,3)=0;
            image_index((i-1)*299+j,3)=299*299+1;
        else
            image_feature((i-1)*299+j,3)=exp(-namuda*(x_img(i,j-1)-x_img(i,j)).^2);
            image_index((i-1)*299+j,3)=(i-1)*299+j-1;
        end
        if j== 299
            image_feature((i-1)*299+j,4)=0;
            image_index((i-1)*299+j,4)=299*299+1;
        else
            image_feature((i-1)*299+j,4)=exp(-namuda*(x_img(i,j+1)-x_img(i,j)).^2);
            image_index((i-1)*299+j,4)=(i-1)*299+j+1;
        end
    end
end

end
