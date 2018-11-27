dataset={'mnist'}; d_i=1;
NN={'BasicCnn','SimpleCnn'};n_i=1;
attack_method={'fgsm','cw','jsma','bim'};
a_i=1;

%% load data
x_path=sprintf('/nfs/pyrex/raid6/hzhang/number_one/data/%s_%s_%s_ori_x.csv',dataset{d_i},NN{n_i},attack_method{a_i});
x=csvread(x_path);

[n,m]=size(x);

for namuda=[0.1 1 5 10]
    for k=1:100
        u=zeros(100,28*28,300);
        v=zeros(100,300);
        for ki=1:100
            start_i = (k-1)*100;
            x_img=vec2mat(x(ki+start_i,:),28);
            %% construct the similarity matrix
            [image_feature, image_index]=calculate_similarity(x_img,m,namuda);
            A_ = knngraph(image_index', image_feature');
            S=transition_matrix(A_);
            [V,D]=eig(full(S),'nobalance');
            d=diag(D);
            u(ki,:,:)=V(:,1:300);
            v(ki,:)=d(1:300);
        end
        data_str=sprintf('/nfs/pyrex/raid6/hzhang/number_one/data/UV/mnist_adv_U_%d_%f.mat',k*100,namuda);
        save(data_str,'u');
        data_str=sprintf('/nfs/pyrex/raid6/hzhang/number_one/data/UV/mnist_adv_V_%d_%f.mat',k*100,namuda);
        save(data_str,'v');
    end
end