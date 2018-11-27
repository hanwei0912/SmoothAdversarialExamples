function  image_feature= graph_matrix(A,m,indx,m_x)
% Construct the similarity matrix of the image

image_feature = zeros(m,4);
% image_index = zeros(m+1,4);
% m_x=299;
for i=1:m_x
    for j=1:m_x
        for k=1:4
            if indx((i-1)*m_x+j,k)<=m_x*m_x && indx((i-1)*m_x+j,k)>=1
                image_feature((i-1)*m_x+j,k)=A((i-1)*m_x+j,indx((i-1)*m_x+j,k));
            end
        end
    end
end
for i =1:4
a=vec2mat(image_feature(:,i),m_x);
image_feature(:,i)=reshape(a,[m,1]);
end
end

