function [ image_feature, image_index ] = calculate_similarity( x_img, m, namuda )
% Construct the similarity matrix of the image

image_feature = zeros(m,4);
image_index = zeros(m+1,4);
for i=1:28
    for j=1:28
        if i== 1
            image_feature((i-1)*28+j,1)=0;
            image_index((i-1)*28+j,1)=28*28+1;
        else
            image_feature((i-1)*28+j,1)=exp(-namuda*abs(x_img(i-1,j)-x_img(i,j)));
            image_index((i-1)*28+j,1)=(i-2)*28+j;
        end
        if i== 28
            image_feature((i-1)*28+j,2)=0;
            image_index((i-1)*28+j,2)=28*28+1;
        else
            image_feature((i-1)*28+j,2)=exp(-namuda*abs(x_img(i+1,j)-x_img(i,j)));
            image_index((i-1)*28+j,2)=i*28+j;
        end
        if j== 1
            image_feature((i-1)*28+j,3)=0;
            image_index((i-1)*28+j,3)=28*28+1;
        else
            image_feature((i-1)*28+j,3)=exp(-namuda*abs(x_img(i,j-1)-x_img(i,j)));
            image_index((i-1)*28+j,3)=(i-1)*28+j-1;
        end
        if j== 28
            image_feature((i-1)*28+j,4)=0;
            image_index((i-1)*28+j,4)=28*28+1;
        else
            image_feature((i-1)*28+j,4)=exp(-namuda*abs(x_img(i,j+1)-x_img(i,j)));
            image_index((i-1)*28+j,4)=(i-1)*28+j+1;
        end
    end
end

