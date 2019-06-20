function our_way_for_suc_dis_rate()
attack_name={'sbim','pgd','ifgsm','fgsm','alp-pgd','0.08-ifgsm'};
eps_name = {'1.0','2.0','3.0','4.0','5.0'};
% eps_name = {'0.1','0.2','0.3','0.4','0.5'};
% eps_name = {'3.0','5.0'};
% eps_name = {'3.0','5.0','8.0','16.0','32.0','64.0'};;

alp_name = {'60.0','20.0','100.0'};
% adv=load('/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/inception/process/SQinEnd.mat');
% calculate(adv)
for i=1:length(attack_name)
    for j=1:length(alp_name)
        l2 = zeros(length(eps_name),10000);
        suc = zeros(length(eps_name),10000);
        for k=1:length(eps_name)
%             adv=load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/inception/process/%s%s-%s.mat',attack_name{i},eps_name{k},alp_name{j}));
%             adv=load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/inception/Adv/process/%s-%s.mat',attack_name{i},eps_name{k}));
            adv=load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/update/whole_image/mnist_BasicCnn_%s_%s_3.0.mat',attack_name{i},eps_name{k}));
%             adv=load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/mnist/adv_new/process/%s-BasicCnn-%s-%s-l2p.mat',attack_name{i},eps_name{k},alp_name{j}));
%             adv=load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/cifar10/adv_new/process/%s-BasicCnn%s-%s-l2p.mat',attack_name{i},eps_name{k},alp_name{j}));
%             adv=load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/cifar10/adv_new/process/%s-BasicCnn-%s.mat',attack_name{i},eps_name{k}));
%             adv=load(sprintf('/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/mnist/process/%s-pgdAdv-%s-%s-l2p.mat',attack_name{i},eps_name{k},alp_name{j}));
            adv.p(adv.ori_a==0) = 0;
            suc(k,:)=adv.p;
            l2(k,:)=adv.l2;
        end
        ind = find(suc);
        l2(ind) = 999999;
        l2_min = min(l2);
        suc_sum = sum(1-suc);
        p =length(find(suc_sum))/10000.0;
        ind = find(l2_min<999999);
        l2_min = l2_min(ind);
        l2_norm = mean(l2_min);
        
        
        
    end   
end
end