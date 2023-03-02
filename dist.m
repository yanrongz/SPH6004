function distance=dist(a,b,p)
    if nargin==2 %只有两个变量时
        p=2;
        distance=dist(a,b,p);
    end

    %定义向量与矩阵之间的欧氏距离
    if size(a,2)~=size(b,2)
        disp('error:两样本列数不同')
    else
        distance=zeros(1,size(b,1));
        for j=1:size(b,1)
            distance(j)=(sum((a-b(j,:)).^2))^(1/p); %第j行每一维的平方之和开p次方
        end
    end
end

