a_path = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/whole_data';
attack = {'bim','biml2','fgsm','sbiml2'};

for i=1:length(attack)
    l2 = zeros(5,1000);
    suc = zeros(5,1000);
    for j=1:5
        file_name = sprintf('%s/%s_inc_%s_p_l2.mat',a_path,attack{i},num2str(j));
        adv=load(file_name);
        adv.p(adv.ori_a==0) = 0;
        suc(j,:)=adv.p;
        l2(j,:)=adv.l2;
    end
    ind = find(suc);
    l2(ind) = 999999;
    [l2_min,ind_min] = min(l2);
    suc_sum = sum(1-suc);
    p =length(find(suc_sum))/10000.0;
    ind = find(l2_min<999999);
    l2_min = l2_min(ind);
end