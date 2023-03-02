function index=find_this_matrix(a,b)
	%a是矩阵，b是一个向量
	%函数是用来寻找矩阵a中与向量b相等的那一行
	k = size(a,1);
	i=0;
	for j = 1:k    %a中的第j行
    	if a(j,:)==b
       		i = i+1;
       		index(i) = j;
    	end
	end
	index = index';    
end
