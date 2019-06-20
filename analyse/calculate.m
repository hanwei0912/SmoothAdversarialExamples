function [p_suc,c_l2] = calculate(cw)
%%
d_total_num = sum(cw.ori_a==1);
cw_m = zeros(d_total_num,2);
cw_m(:,1)=cw.l2(cw.ori_a==1);
cw_m(:,2)=1-cw.p(cw.ori_a==1);
% cw_m(cw.ori_a==0,1) = 0;
% cw_m(cw.ori_a==0,2) = 0;
cw.p(cw.ori_a==0) = 1;

ind = find(cw_m(:,1)<30);
cw_m = cw_m(ind,:);

p_suc = sum(cw_m(:,2))/d_total_num;
c_l2  = mean(cw_m(cw_m(:,2)==1,1))
end