name_list1= {'cw','clip','bim','fgsm'};
name_list2= {'fgsm','cw','clip','bim'};
for i=1:length(name_list1)
    for j = 1:length(name_list2)
    name = sprintf('MnistSNN/mnistW%sM%sAdv.png',name_list1{i},name_list2{j});
    a=imread(name);
    A = imresize(a,[224 224],'nearest');
    imshow(A);
    imwrite(A,name);
    end
end
